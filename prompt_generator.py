import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =====================================================================
# 1. 模型定义 (保持不变)
# =====================================================================
class AutoBoxGenerator(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # 共享卷积层
        self.shared_convs = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 热力图分支
        # 最后一层 bias 初始化为特定负值，有助于 Focal Loss 训练初期稳定
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)
        self.heatmap_head.bias.data.fill_(-2.19) 
        
        # 尺寸回归分支
        self.wh_head = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        features = self.shared_convs(x)
        # 热力图用 Sigmoid 激活到 [0,1]
        pred_heatmap = torch.sigmoid(self.heatmap_head(features))
        # 宽高用 Softplus 或 ReLU 保证为正数
        pred_wh = F.softplus(self.wh_head(features))
        return pred_heatmap, pred_wh

# =====================================================================
# 2. 工具函数：绘制高斯热力图 (核心升级!)
# =====================================================================
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    """在 heatmap 上绘制一个高斯圆点"""
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

# =====================================================================
# 3. 构建目标 (升级为高斯目标)
# =====================================================================
def build_target_v2(boxes_list, feature_shape=(64, 64), original_image_size=1024, device='cuda'):
    batch_size = len(boxes_list)
    h_feat, w_feat = feature_shape
    stride = original_image_size / h_feat
    
    # 初始化目标 (用 numpy 在 CPU 上生成高斯图更快，最后转 tensor)
    target_heatmap_np = np.zeros((batch_size, 1, h_feat, w_feat), dtype=np.float32)
    target_wh = torch.zeros((batch_size, 2, h_feat, w_feat), device=device)
    target_mask = torch.zeros((batch_size, 1, h_feat, w_feat), device=device)

    for b in range(batch_size):
        current_boxes = boxes_list[b]
        # 确保是 numpy 或 cpu tensor
        if isinstance(current_boxes, torch.Tensor):
            current_boxes = current_boxes.cpu().numpy()

        for box in current_boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w < 2 or h < 2: continue

            # 中心点和宽高 (归一化到 feature map 尺度)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cx_feat, cy_feat = cx / stride, cy / stride
            cx_int, cy_int = int(cx_feat), int(cy_feat)
            
            w_feat, h_feat_val = w / stride, h / stride

            # 边界检查
            if 0 <= cx_int < w_feat and 0 <= cy_int < h_feat:
                # A. 绘制高斯热力图
                # 半径根据目标大小动态决定，这里简化为固定半径或基于宽高
                radius = max(0, int(min(w_feat, h_feat_val) / 2))
                draw_umich_gaussian(target_heatmap_np[b, 0], (cx_int, cy_int), radius)
                
                # B. 设置宽高目标
                target_wh[b, 0, cy_int, cx_int] = w_feat
                target_wh[b, 1, cy_int, cx_int] = h_feat_val
                
                # C. 设置 Mask (只在中心点计算 WH loss)
                target_mask[b, 0, cy_int, cx_int] = 1.0

    target_heatmap = torch.from_numpy(target_heatmap_np).to(device)
    return target_heatmap, target_wh, target_mask

# =====================================================================
# 4. Loss 函数 (Focal Loss + L1 Loss)
# =====================================================================
def focal_loss(pred, target, alpha=2, beta=4):
    """CenterNet Modified Focal Loss"""
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, beta)

    loss = 0
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    # 正样本 Loss
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    # 负样本 Loss (加权，离中心越近权重越小)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def auto_box_loss_v2(pred_heatmap, pred_wh, target_heatmap, target_wh, target_mask):
    # 1. Heatmap Loss (Focal Loss)
    loss_hm = focal_loss(pred_heatmap, target_heatmap)
    
    # 2. WH Loss (L1 Loss)
    num_pos = target_mask.sum()
    if num_pos > 0:
        # 只在有目标中心点的位置计算 WH loss
        loss_wh = F.l1_loss(pred_wh * target_mask, target_wh * target_mask, reduction='sum')
        loss_wh = loss_wh / (num_pos + 1e-7)
    else:
        loss_wh = torch.tensor(0.0, device=pred_heatmap.device, requires_grad=True)

    return loss_hm, loss_wh