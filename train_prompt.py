import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import cv2
import numpy as np
import json
from tqdm import tqdm

# 引入模块
from segment_anything import sam_model_registry
# 仍然复用 prompt_generator 里的模型结构和 loss
# 改为引入 v2 版本的函数
from prompt_generator import AutoBoxGenerator, build_target_v2, auto_box_loss_v2
# =================================================================================
# 1. 定义专属的 Dense Dataset (一次性返回所有框)
# =================================================================================
class DenseTrainingDataset(Dataset):
    def __init__(self, data_path, image_size=1024):
        self.image_size = image_size
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        
        # 加载训练集 JSON
        json_path = os.path.join(data_path, 'image2label_train.json')
        with open(json_path, 'r') as f:
            self.dataset = json.load(f)
        
        self.image_paths = []
        self.label_paths = []
        
        # 解析路径 (简化版逻辑)
        for img_p, label_p in self.dataset.items():
            # 兼容路径前缀
            if img_p.startswith('data_demo') or img_p.startswith('cpm17'): 
                # 这里假设你的 data_path 结构已经很标准，直接拼文件名
                img_name = os.path.basename(img_p)
                mask_name = os.path.basename(label_p) if isinstance(label_p, str) else os.path.basename(label_p[0])
            else:
                img_name = os.path.basename(img_p)
                mask_name = os.path.basename(label_p) if isinstance(label_p, str) else os.path.basename(label_p[0])

            # 尝试在标准目录找
            final_img_path = os.path.join(data_path, 'train/Images', img_name)
            final_mask_path = os.path.join(data_path, 'train/Labels', mask_name)
            
            if not os.path.exists(final_img_path):
                # 备用路径逻辑，防止找不到
                final_img_path = os.path.join(data_path, img_name)
            if not os.path.exists(final_mask_path):
                final_mask_path = os.path.join(data_path, mask_name)
                
            if os.path.exists(final_img_path) and os.path.exists(final_mask_path):
                self.image_paths.append(final_img_path)
                self.label_paths.append(final_mask_path)
        
        print(f"[DenseLoader] Loaded {len(self.image_paths)} training images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, -1) # 读取原始 Instance Mask
        
        # Resize
        original_size = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # =======================================================
        # [关键] 提取所有细胞的 Box
        # =======================================================
        boxes_list = []
        obj_ids = np.unique(label)
        for obj_id in obj_ids:
            if obj_id == 0: continue
            y_indices, x_indices = np.where(label == obj_id)
            if len(y_indices) < 3: continue # 过滤极小噪点
            
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            boxes_list.append([x_min, y_min, x_max, y_max])
            
        # 归一化图片
        image = (image - self.pixel_mean) / self.pixel_std
        image = torch.tensor(image).permute(2, 0, 1).float()
        
        # 转 Tensor
        if len(boxes_list) > 0:
            boxes_tensor = torch.tensor(boxes_list).float()
        else:
            boxes_tensor = torch.tensor([[0,0,1,1]]).float()

        return {
            "image": image,
            "all_boxes": boxes_tensor # 返回所有框
        }

def collate_fn_dense(batch):
    # 自定义 collate，因为 boxes 数量不一，不能 stack
    images = torch.stack([item['image'] for item in batch], dim=0)
    # boxes 保持为 list
    all_boxes = [item['all_boxes'] for item in batch]
    return {'image': images, 'all_boxes': all_boxes}

# =================================================================================
# 2. 训练主流程
# =================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/MoNuSeg_Processed')
    parser.add_argument('--sam_checkpoint', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='workdir/models/auto_box_head_dense')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    
    # SAM-Med2D args
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--encoder_adapter', action='store_true', default=True)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    # 加载 SAM (冻结)
    print("Loading SAM (Frozen)...")
    sam = sam_model_registry['vit_b'](args=args)
    sam.to(device)
    for param in sam.parameters():
        param.requires_grad = False
    sam.eval()

    # 初始化 Generator
    print("Initializing AutoBoxGenerator...")
    box_generator = AutoBoxGenerator(embed_dim=256).to(device)
    box_generator.train()
    
    optimizer = optim.AdamW(box_generator.parameters(), lr=args.lr)

    # 使用新的 Dataset
    dataset = DenseTrainingDataset(data_path=args.data_path, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_dense)

    print(f"Start training Dense Auto-Box Head for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            images = batch['image'].to(device)
            # 这是一个 list of tensors
            gt_boxes_list = [b.to(device) for b in batch['all_boxes']]
            
            with torch.no_grad():
                image_embedding = sam.image_encoder(images)
            
            # 预测
            pred_heatmap, pred_wh = box_generator(image_embedding)
            
            # 构建目标 (这次传入的是全量的框！)
            target_heatmap, target_wh, target_mask = build_target_v2( # 改为 v2
                gt_boxes_list, 
                feature_shape=(64, 64), 
                original_image_size=args.image_size,
                device=device
            )

        # 使用 v2 计算 Loss
            loss_hm, loss_wh = auto_box_loss_v2(pred_heatmap, pred_wh, target_heatmap, target_wh, target_mask) # 改为 v2

        # 重要：加大 WH Loss 权重，改为 1.0
            loss = loss_hm + 1.0 * loss_wh
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'hm': loss_hm.item()})
            
        # 每 10 轮保存
        if (epoch + 1) % 10 == 0:
            save_name = os.path.join(args.save_path, f'box_head_epoch{epoch+1}.pth')
            torch.save(box_generator.state_dict(), save_name)

if __name__ == "__main__":
    main()