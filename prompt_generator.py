import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoBoxGenerator(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # 1. 共享卷积层：提取定位特征
        self.shared_convs = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 2. 热力图分支 (Heatmap Head): 预测细胞中心概率
        # 输出通道=1，值域 [0, 1]
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)
        
        # 3. 尺寸回归分支 (Size Head): 预测宽和高 (Width, Height)
        # 输出通道=2
        self.wh_head = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        """
        x: SAM Image Embedding [Batch, 256, 64, 64]
        return:
            pred_heatmap: [Batch, 1, 64, 64] (Sigmoid activated)
            pred_wh:      [Batch, 2, 64, 64] (ReLU activated, 必须为正)
        """
        features = self.shared_convs(x)
        
        # 热力图使用 Sigmoid 归一化到 0-1
        pred_heatmap = torch.sigmoid(self.heatmap_head(features))
        
        # 宽高必须大于0，使用 ReLU 或 Softplus
        pred_wh = F.relu(self.wh_head(features))
        
        return pred_heatmap, pred_wh

def build_target(boxes, feature_shape, original_image_size=1024, device='cuda'):
    """
    将真实的 Box 转换为训练用的 Heatmap 和 Size Map
    boxes: List of tensors or Tensor [Batch, N, 4] (x1, y1, x2, y2)
    feature_shape: (64, 64) - SAM embedding 的空间尺寸
    """
    batch_size = len(boxes)
    h_feat, w_feat = feature_shape
    stride = original_image_size / h_feat # 通常是 16
    
    # 初始化目标
    target_heatmap = torch.zeros((batch_size, 1, h_feat, w_feat), device=device)
    target_wh = torch.zeros((batch_size, 2, h_feat, w_feat), device=device)
    # 用于记录哪些位置有目标，方便计算 Loss (Mask)
    target_mask = torch.zeros((batch_size, 1, h_feat, w_feat), device=device)

    for b in range(batch_size):
        # 获取当前图片的所有 GT Box
        # 注意：DataLoader 里如果没目标是 [0,0,1,1]，这里可以过滤掉面积极小的框
        current_boxes = boxes[b] 
        
        for box in current_boxes:
            x1, y1, x2, y2 = box
            
            # 过滤无效框 (Dummy box)
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue

            # 1. 计算中心点和宽高 (在原图尺度)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # 2. 映射到 Feature Map 尺度 (除以 16)
            # 向下取整得到网格坐标
            cx_feat = int(cx / stride)
            cy_feat = int(cy / stride)

            # 边界检查
            if 0 <= cx_feat < w_feat and 0 <= cy_feat < h_feat:
                # 3. 设置热力图 (这里用简化的 Point，也可以用高斯核 Gaussian)
                target_heatmap[b, 0, cy_feat, cx_feat] = 1.0
                
                # 4. 设置宽高目标 (归一化到 Feature Map 尺度，方便回归)
                target_wh[b, 0, cy_feat, cx_feat] = w / stride
                target_wh[b, 1, cy_feat, cx_feat] = h / stride
                
                # 5. 标记该位置有目标
                target_mask[b, 0, cy_feat, cx_feat] = 1.0

    return target_heatmap, target_wh, target_mask

def auto_box_loss(pred_heatmap, pred_wh, target_heatmap, target_wh, target_mask):
    """
    计算检测头的 Loss
    """
    # 1. Heatmap Loss (Focal Loss 变体 或 简单的 MSE)
    # 这里为了简单稳健，使用 MSE，针对稀疏目标效果也不错
    loss_heatmap = F.mse_loss(pred_heatmap, target_heatmap)
    
    # 2. WH Loss (只在有目标的位置计算)
    # target_mask 只有在有中心点的地方是 1
    num_pos = target_mask.sum()
    if num_pos > 0:
        # 只取出有目标位置的预测值和真实值
        loss_wh = F.l1_loss(pred_wh * target_mask, target_wh * target_mask, reduction='sum')
        loss_wh = loss_wh / num_pos
    else:
        loss_wh = torch.tensor(0.0, device=pred_heatmap.device, requires_grad=True)

    return loss_heatmap, loss_wh