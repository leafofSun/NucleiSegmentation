import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import cv2
import numpy as np
import json
import glob
from tqdm import tqdm

# 引入模块 
# (请确保 prompt_generator.py 是包含 build_target_v2 和 auto_box_loss_v2 的最新版)
from segment_anything import sam_model_registry
from prompt_generator import AutoBoxGenerator, build_target_v2, auto_box_loss_v2

# =================================================================================
# 1. SA-1B 标准格式数据集加载器
# =================================================================================
class SA1BDataset(Dataset):
    def __init__(self, data_root, image_size=1024):
        self.image_size = image_size
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        
        # 递归扫描所有 JSON 文件
        # SA-1B 格式的核心是 JSON，图片与 JSON 同名
        self.json_files = sorted(glob.glob(os.path.join(data_root, '**', '*.json'), recursive=True))
        
        # 过滤掉非标注文件（以防万一文件夹里有无关json）
        self.valid_files = []
        for jf in self.json_files:
            if "image2label" in jf: continue # 排除 MoNuSeg 旧版索引文件
            self.valid_files.append(jf)
            
        print(f"✅ [Dataset] Found {len(self.valid_files)} JSON annotation files in {data_root}")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, index):
        json_path = self.valid_files[index]
        
        # 1. 寻找对应的图片
        # 假设图片和 JSON 同名，尝试常见后缀
        base_path = os.path.splitext(json_path)[0]
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            if os.path.exists(base_path + ext):
                img_path = base_path + ext
                break
        
        if img_path is None:
            raise FileNotFoundError(f"❌ Image not found for JSON: {json_path}")

        # 2. 读取图片
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"❌ Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = image.shape[:2]
        
        # Resize 图片到 1024x1024
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        
        # 3. 解析 JSON 获取 BBox
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        boxes_list = []
        
        # 计算缩放比例
        scale_x = self.image_size / ori_w
        scale_y = self.image_size / ori_h
        
        # SA-1B 格式通常包含 'annotations' 列表
        annotations = data.get('annotations', [])
        
        for ann in annotations:
            if 'bbox' not in ann: continue
            
            # SA-1B 标准 bbox 格式: [x, y, w, h]
            x, y, w, h = ann['bbox']
            
            # 转换为我们需要的格式: [x1, y1, x2, y2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            
            # 执行坐标缩放
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            
            # 简单的边界保护和噪点过滤
            if (x2 - x1) < 2 or (y2 - y1) < 2: continue
            
            boxes_list.append([x1, y1, x2, y2])
            
        # 4. 归一化 & 转 Tensor
        image_tensor = (image_resized - self.pixel_mean) / self.pixel_std
        image_tensor = torch.tensor(image_tensor).permute(2, 0, 1).float()
        
        if len(boxes_list) > 0:
            boxes_tensor = torch.tensor(boxes_list).float()
        else:
            # 防止空图报错，给一个假的 0 面积框（Loss 计算时会自动忽略）
            boxes_tensor = torch.tensor([[0,0,1,1]]).float()

        return {
            "image": image_tensor,
            "all_boxes": boxes_tensor
        }

def collate_fn_dense(batch):
    images = torch.stack([item['image'] for item in batch], dim=0)
    all_boxes = [item['all_boxes'] for item in batch]
    return {'image': images, 'all_boxes': all_boxes}

# =================================================================================
# 2. 训练主流程
# =================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="SA-1B格式数据集根目录")
    parser.add_argument('--sam_checkpoint', type=str, required=True, help="SAM模型权重路径")
    parser.add_argument('--save_path', type=str, default='workdir/models/auto_box_sa1b')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--encoder_adapter', action='store_true', default=True)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    # 1. 加载 SAM (冻结参数，只做特征提取)
    print("Loading SAM (Frozen)...")
    sam = sam_model_registry['vit_b'](args=args)
    sam.to(device)
    for param in sam.parameters():
        param.requires_grad = False
    sam.eval()

    # 2. 初始化 AutoBoxGenerator
    print("Initializing AutoBoxGenerator...")
    box_generator = AutoBoxGenerator(embed_dim=256).to(device)
    box_generator.train()
    
    # 优化器
    optimizer = optim.AdamW(box_generator.parameters(), lr=args.lr)

    # 3. 加载数据集
    print(f"Initializing SA-1B Dataset from: {args.data_path}")
    dataset = SA1BDataset(data_root=args.data_path, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_dense)

    print(f"Start training Auto-Box Head for {args.epochs} epochs...")
    
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            images = batch['image'].to(device)
            gt_boxes_list = [b.to(device) for b in batch['all_boxes']]
            
            with torch.no_grad():
                image_embedding = sam.image_encoder(images)
            
            pred_heatmap, pred_wh = box_generator(image_embedding)
            
            # === 核心策略: V2 Target (高斯热力图) ===
            target_heatmap, target_wh, target_mask = build_target_v2(
                gt_boxes_list, 
                feature_shape=(64, 64), 
                original_image_size=args.image_size,
                device=device
            )
            
            # === 核心策略: V2 Loss (Focal Loss + L1 Loss) ===
            loss_hm, loss_wh = auto_box_loss_v2(pred_heatmap, pred_wh, target_heatmap, target_wh, target_mask)
            
            # === 核心策略: 加大 WH 权重 ===
            loss = loss_hm + 1.0 * loss_wh
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'hm': loss_hm.item(), 'wh': loss_wh.item()})
            
        # 学习率衰减 (可选，简单起见这里省略，AdamW 通常不需要太复杂的调度)
        
        # 保存最佳模型
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(box_generator.state_dict(), os.path.join(args.save_path, 'best_box_head.pth'))
            print(f"🔥 Best Model Saved (Loss: {best_loss:.4f})")

        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_name = os.path.join(args.save_path, f'box_head_epoch{epoch+1}.pth')
            torch.save(box_generator.state_dict(), save_name)

if __name__ == "__main__":
    main()