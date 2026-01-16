from albumentations.pytorch import ToTensorV2
import cv2
import albumentations as A
import torch
import numpy as np
from torch.nn import functional as F
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import random
import torch.nn as nn
import logging
import os

# ==========================================
# 辅助函数部分
# ==========================================

def get_boxes_from_mask(mask, box_num=1, std=0.1, max_pixel=5):
    """
    从 mask 中提取边界框，支持空 mask 的健壮处理
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # 如果 mask 包含 255 (Ignore)，我们在生成 Box 时只看 1 (Target)
    mask_for_box = np.zeros_like(mask)
    mask_for_box[mask == 1] = 1
    
    if mask_for_box.max() > 0:
        label_img = label(mask_for_box)
    else:
        return torch.tensor([[0, 0, 1, 1]] * box_num, dtype=torch.float)
        
    regions = regionprops(label_img)
    boxes = [tuple(region.bbox) for region in regions]

    if len(boxes) == 0:
        return torch.tensor([[0, 0, 1, 1]] * box_num, dtype=torch.float)

    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    noise_boxes = []
    for box in boxes:
        y0, x0, y1, x1 = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
        try:
            noise_x = np.random.randint(-max_noise, max_noise)
        except:
            noise_x = 0
        try:
            noise_y = np.random.randint(-max_noise, max_noise)
        except:
            noise_y = 0
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
        
    return torch.as_tensor(noise_boxes, dtype=torch.float)


def select_random_points(pr, gt, point_num = 9):
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    
    # 只在明确的 0/1 区域计算 Error，忽略 255
    valid_mask = (gt != 255)
    error[(pred != gt) & valid_mask] = 1

    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            h, w = one_pred.shape
            indices = np.random.randint(0, min(h, w), size=(point_num, 2))
            selected_indices = indices
            
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_gt[x,y] == 1:
                label = 1
            elif one_gt[x,y] == 0:
                label = 0
            else:
                label = -1 
            points.append((y, x))
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)


def init_point_sampling(mask, get_point=1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        if fg_size > 0:
            fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
            fg_coords_sel = fg_coords[fg_indices]
            fg_labels = np.ones(num_fg)
        else:
            fg_coords_sel = np.empty((0, 2))
            fg_labels = np.array([])
            num_bg = get_point

        if bg_size > 0:
            bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
            bg_coords_sel = bg_coords[bg_indices]
            bg_labels = np.zeros(num_bg)
        else:
            bg_coords_sel = np.empty((0, 2))
            bg_labels = np.array([])

        coords = np.concatenate([fg_coords_sel, bg_coords_sel], axis=0)
        labels = np.concatenate([fg_labels, bg_labels]).astype(int)
        
        if len(coords) < get_point:
             padding = get_point - len(coords)
             coords = np.pad(coords, ((0, padding), (0, 0)), mode='edge')
             labels = np.pad(labels, (0, padding), mode='edge')

        indices = np.random.permutation(len(coords))
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels
    

def get_transforms(img_size, ori_h, ori_w, mode='train'):
    transforms = []
    if mode == 'train':
        if ori_h < img_size or ori_w < img_size:
            transforms.append(A.PadIfNeeded(
                min_height=img_size, 
                min_width=img_size, 
                border_mode=cv2.BORDER_CONSTANT, 
                fill=0, 
                fill_mask=0
            ))
        transforms.append(A.RandomCrop(height=img_size, width=img_size))
    elif mode == 'test':
        transforms.append(A.PadIfNeeded(
            min_height=img_size, 
            min_width=img_size, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=0, 
            fill_mask=0
        ))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)


def train_transforms(img_size, ori_h, ori_w):
    return get_transforms(img_size, ori_h, ori_w, mode='train')


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def generate_point(masks, labels, low_res_masks, batched_input, point_num):
    masks_clone = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    masks_binary = (masks_sigmoid > 0.5).float()

    low_res_masks_clone = low_res_masks.clone()
    low_res_masks_logist = torch.sigmoid(low_res_masks_clone)

    points, point_labels = select_random_points(masks_binary, labels, point_num = point_num)
    batched_input["mask_inputs"] = low_res_masks_logist
    batched_input["point_coords"] = torch.as_tensor(points)
    batched_input["point_labels"] = torch.as_tensor(point_labels)
    batched_input["boxes"] = None
    return batched_input


def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    return batched_input


def draw_boxes(img, boxes):
    img_copy = np.copy(img)
    for box in boxes:
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(img_copy, pt1, pt2, (0, 255, 0), 2)
    return img_copy


def save_masks(preds, save_path, mask_name, image_size, original_size, pad=None,  boxes=None, points=None, visual_prompt=True):
    ori_h, ori_w = original_size

    preds = torch.sigmoid(preds)
    preds[preds > 0.5] = int(1)
    preds[preds <= 0.5] = int(0)

    mask = preds.squeeze().cpu().numpy()
    
    if mask.shape[0] != ori_h or mask.shape[1] != ori_w:
        mask = cv2.resize(mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
    
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    if visual_prompt: 
        if boxes is not None:
            boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
            if boxes_np.ndim == 1:
                boxes_np = boxes_np.reshape(1, -1)
            elif boxes_np.ndim > 2:
                boxes_np = boxes_np.reshape(-1, 4)
            
            boxes_ori = []
            for box in boxes_np:
                x0, y0, x1, y1 = box
                if pad is not None:
                    x0_ori = max(0, int(x0 - pad[1] + 0.5))
                    y0_ori = max(0, int(y0 - pad[0] + 0.5))
                    x1_ori = min(ori_w, int(x1 - pad[1] + 0.5))
                    y1_ori = min(ori_h, int(y1 - pad[0] + 0.5))
                else:
                    x0_ori = int(x0 * ori_w / image_size) 
                    y0_ori = int(y0 * ori_h / image_size) 
                    x1_ori = int(x1 * ori_w / image_size) 
                    y1_ori = int(y1 * ori_h / image_size)
                
                if x1_ori > x0_ori and y1_ori > y0_ori:
                    boxes_ori.append((x0_ori, y0_ori, x1_ori, y1_ori))
            
            if boxes_ori:
                mask = draw_boxes(mask, boxes_ori)

        if points is not None:
            point_coords, point_labels = points[0], points[1]
            if isinstance(point_coords, torch.Tensor):
                point_coords = point_coords.squeeze(0).cpu().numpy()
            if isinstance(point_labels, torch.Tensor):
                point_labels = point_labels.squeeze(0).cpu().numpy()
            
            if point_coords.ndim == 1:
                point_coords = point_coords.reshape(1, -1)
            elif point_coords.ndim > 2:
                point_coords = point_coords.reshape(-1, 2)
            
            if point_labels.ndim > 1:
                point_labels = point_labels.flatten()
            
            for (x, y), label in zip(point_coords, point_labels):
                if pad is not None:
                    x_ori = max(0, min(ori_w - 1, int(x - pad[1] + 0.5)))
                    y_ori = max(0, min(ori_h - 1, int(y - pad[0] + 0.5)))
                else:
                    x_ori = max(0, min(ori_w - 1, int(x * ori_w / image_size)))
                    y_ori = max(0, min(ori_h - 1, int(y * ori_h / image_size)))
                
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.drawMarker(mask, (int(x_ori), int(y_ori)), color, markerType=cv2.MARKER_CROSS, markerSize=7, thickness=2)  
    
    os.makedirs(save_path, exist_ok=True)
    mask_path = os.path.join(save_path, f"{mask_name}")
    cv2.imwrite(mask_path, np.uint8(mask))


# ==========================================
# Loss Classes
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, pred, mask):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        valid_mask = (mask != self.ignore_index)
        if valid_mask.sum() == 0:
            return pred.sum() * 0.0

        p = torch.sigmoid(pred)[valid_mask]   # [N]
        targets = mask[valid_mask]            # [N]
        
        num_pos = torch.sum(targets)
        num_neg = targets.numel() - num_pos
        
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * targets * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - targets) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, mask):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        valid_mask = (mask != self.ignore_index)
        if valid_mask.sum() == 0:
            return pred.sum() * 0.0

        p = torch.sigmoid(pred)[valid_mask]
        targets = mask[valid_mask]

        intersection = torch.sum(p * targets)
        union = torch.sum(p) + torch.sum(targets)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_loss


class MaskIoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(MaskIoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        assert pred_mask.shape == ground_truth_mask.shape
        valid = (ground_truth_mask != self.ignore_index)
        
        p = torch.sigmoid(pred_mask)
        gt = ground_truth_mask.clone()
        
        p = p * valid.float()
        gt = gt * valid.float()
        
        intersection = torch.sum(p * gt, dim=(1, 2, 3))
        union = torch.sum(p, dim=(1, 2, 3)) + torch.sum(gt, dim=(1, 2, 3)) - intersection
        
        gt_iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((gt_iou - pred_iou.squeeze()) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    def __init__(self, weight=20.0, iou_scale=1.0, ignore_index=255):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.ignore_index = ignore_index
        
        self.focal_loss = FocalLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.maskiou_loss = MaskIoULoss(ignore_index=ignore_index)

    def forward(self, pred, mask, pred_iou):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        
        loss1 = self.weight * focal_loss + dice_loss
        
        if pred_iou is not None:
            loss2 = self.maskiou_loss(pred, mask, pred_iou)
            loss = loss1 + loss2 * self.iou_scale
        else:
            loss = loss1
            
        return loss, {'focal': focal_loss.item(), 'dice': dice_loss.item()}

# ===  Point Guidance Loss ===
def point_guidance_loss(pred_heatmap, gt_nuclei):
    """
    计算热力图生成的 MSE Loss (支持 2通道 输出)
    pred_heatmap: [B, 2, H, W] (Ch0: Foreground, Ch1: Background)
    gt_nuclei:    [B, 1, H, W] (0/1 Mask)
    """
    # 1. 确保 GT 维度是 [B, 1, H, W]
    if gt_nuclei.ndim == 3:
        gt_nuclei = gt_nuclei.unsqueeze(1)
        
    # 2. 获取预测的 sigmoid 概率
    pred_prob = torch.sigmoid(pred_heatmap)

    # 情况 A: 模型输出 2 个通道 (前景 + 背景)
    if pred_heatmap.shape[1] == 2:
        # Channel 0 监督前景
        pred_fg = pred_prob[:, 0:1, :, :]
        loss_fg = F.mse_loss(pred_fg, gt_nuclei)
        
        # Channel 1 监督背景 (1 - GT)
        pred_bg = pred_prob[:, 1:2, :, :]
        gt_bg = 1.0 - gt_nuclei
        loss_bg = F.mse_loss(pred_bg, gt_bg)
        
        return loss_fg + loss_bg

    # 情况 B: 模型只输出 1 个通道
    else:
        return F.mse_loss(pred_prob, gt_nuclei)


# === 物理-语义一致性 Loss ===
def physical_semantic_consistency_loss(
    peak_counts: torch.Tensor,
    density_logits: torch.Tensor,
    margin_low: float = 10.0,
    margin_high: float = 30.0,
    temperature: float = 1.0
):
    """
    物理-语义一致性 Loss ($L_{consistency}$)
    
    比较热力图计数等级和分类头预测等级，使用 Margin 机制实现容差设计。
    
    Args:
        peak_counts: [B] 从热力图中统计的峰值数量 $N_{pred}$
        density_logits: [B, 3] 密度分类头的 logits（3类：低、中、高）
        margin_low: float 低密度阈值（默认 10）
        margin_high: float 高密度阈值（默认 30）
        temperature: float 温度参数，用于软化预测
    
    Returns:
        loss: scalar 一致性损失
    """
    device = peak_counts.device
    B = peak_counts.shape[0]
    
    # 获取预测的密度类别（使用 softmax 获取概率分布）
    density_probs = F.softmax(density_logits / temperature, dim=1)  # [B, 3]
    # 0: 低密度, 1: 中密度, 2: 高密度
    pred_class = torch.argmax(density_probs, dim=1)  # [B]
    
    # 定义每个类别对应的计数范围
    # 低密度: < margin_low (10)
    # 中密度: margin_low ~ margin_high (10-30)
    # 高密度: > margin_high (30)
    
    # 使用向量化操作保持梯度流
    # 将 peak_counts 转换为与 density_probs 兼容的形状
    n_pred = peak_counts.unsqueeze(1)  # [B, 1]
    
    # 为每个类别创建掩码
    is_low = (pred_class == 0).float().unsqueeze(1)  # [B, 1]
    is_mid = (pred_class == 1).float().unsqueeze(1)  # [B, 1]
    is_high = (pred_class == 2).float().unsqueeze(1)  # [B, 1]
    
    # 计算惩罚（向量化）
    # 低密度类别：如果计数 >= margin_low，惩罚
    penalty_low = torch.clamp((n_pred - margin_low) / margin_low, min=0.0) * is_low
    loss_low = (penalty_low * density_probs[:, 0:1]).sum()
    
    # 中密度类别：如果计数 < margin_low 或 > margin_high，惩罚
    penalty_mid_low = torch.clamp((margin_low - n_pred) / margin_low, min=0.0) * is_mid
    penalty_mid_high = torch.clamp((n_pred - margin_high) / margin_high, min=0.0) * is_mid
    loss_mid = ((penalty_mid_low + penalty_mid_high) * density_probs[:, 1:2]).sum()
    
    # 高密度类别：如果计数 < margin_high，惩罚
    penalty_high = torch.clamp((margin_high - n_pred) / margin_high, min=0.0) * is_high
    loss_high = (penalty_high * density_probs[:, 2:3]).sum()
    
    # 总损失
    loss = loss_low + loss_mid + loss_high
    
    # 平均化
    return loss / B if B > 0 else loss


def generate_density_map_from_mask(gt_mask: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """
    从 GT mask 生成密度图（基于点的高斯热力图）
    参考 DeNSe 论文：将每个细胞核中心点转换为高斯分布
    使用纯 PyTorch 实现，避免外部依赖
    
    Args:
        gt_mask: [H, W] 或 [1, H, W] 的 GT mask（值为 0 和 1，或 0 和 255）
        sigma: float 高斯核的标准差（控制密度扩散范围）
    
    Returns:
        density_map: [1, H, W] 的密度图（值域 [0, 1]）
    """
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.squeeze(0)
    H, W = gt_mask.shape
    device = gt_mask.device
    dtype = gt_mask.dtype
    
    # 将 mask 归一化到 [0, 1]
    if gt_mask.max() > 1:
        gt_mask = (gt_mask > 127).float()
    else:
        gt_mask = gt_mask.float()
    
    # 简化版本：使用最大池化找到局部最大值作为细胞核中心
    # 这比连通域分析更快，且是纯 PyTorch 实现
    kernel_size = 5
    padding = kernel_size // 2
    
    # 使用最大池化找到局部最大值
    gt_mask_padded = F.pad(gt_mask.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='constant', value=0)
    local_max = F.max_pool2d(gt_mask_padded, kernel_size=kernel_size, stride=1)
    local_max = local_max.squeeze(0).squeeze(0)[padding:H+padding, padding:W+padding]
    
    # 找到局部最大值的位置（作为细胞核中心）
    is_center = (gt_mask > 0.5) & (gt_mask == local_max)
    center_points = torch.nonzero(is_center, as_tuple=False)  # [N, 2] (y, x)
    
    # 生成高斯热力图
    y_coords = torch.arange(H, device=device, dtype=dtype).float().view(-1, 1)  # [H, 1]
    x_coords = torch.arange(W, device=device, dtype=dtype).float().view(1, -1)  # [1, W]
    
    density_map = torch.zeros((H, W), device=device, dtype=dtype)
    
    if len(center_points) > 0:
        # 为每个中心点生成高斯分布
        for center in center_points:
            cy, cx = center[0].float(), center[1].float()
            # 高斯分布: exp(-((y-cy)^2 + (x-cx)^2) / (2*sigma^2))
            gaussian = torch.exp(-((y_coords - cy) ** 2 + (x_coords - cx) ** 2) / (2 * sigma ** 2))
            density_map += gaussian
        
        # 归一化到 [0, 1]
        if density_map.max() > 0:
            density_map = density_map / density_map.max()
    
    return density_map.unsqueeze(0)  # [1, H, W]


def density_map_loss(
    pred_density_map: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    mse_weight: float = 1.0,
    iou_weight: float = 0.5
) -> torch.Tensor:
    """
    密度图损失函数（参考 DeNSe 论文）
    包含两部分：
    1. MSE Loss: 预测密度图 vs GT 密度图
    2. IoU Loss: 预测 mask 与密度图高响应区域的重叠
    
    Args:
        pred_density_map: [B, 1, H, W] 预测的密度图
        gt_mask: [B, 1, H, W] 或 [B, H, W] GT mask
        pred_mask: [B, 1, H, W] 或 [B, H, W] 预测的 mask（logits，未 sigmoid）
        mse_weight: float MSE 损失的权重
        iou_weight: float IoU 损失的权重
    
    Returns:
        loss: scalar 总损失
    """
    device = pred_density_map.device
    
    # 确保维度一致
    if pred_density_map.dim() == 3:
        pred_density_map = pred_density_map.unsqueeze(1)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)
    
    B, _, H, W = pred_density_map.shape
    
    # 调整大小以匹配
    if gt_mask.shape[-2:] != (H, W):
        gt_mask = F.interpolate(gt_mask.float(), size=(H, W), mode='nearest')
    if pred_mask.shape[-2:] != (H, W):
        pred_mask = F.interpolate(pred_mask.unsqueeze(0) if pred_mask.dim() == 2 else pred_mask, 
                                  size=(H, W), mode='bilinear', align_corners=False)
    
    # 生成 GT 密度图
    gt_density_list = []
    for b in range(B):
        gt_density = generate_density_map_from_mask(gt_mask[b, 0], sigma=2.0)
        gt_density_list.append(gt_density)
    gt_density_map = torch.cat(gt_density_list, dim=0).unsqueeze(1)  # [B, 1, H, W]
    
    # 1. MSE Loss: 预测密度图 vs GT 密度图
    mse_loss = F.mse_loss(pred_density_map, gt_density_map)
    
    # 2. IoU Loss: 预测 mask 与密度图高响应区域的重叠
    # 将预测 mask 转换为二值化
    pred_mask_binary = torch.sigmoid(pred_mask) > 0.5  # [B, 1, H, W]
    
    # 密度图的高响应区域（阈值可调）
    density_threshold = 0.3
    density_high_response = (pred_density_map > density_threshold).float()  # [B, 1, H, W]
    
    # 计算 IoU
    intersection = (pred_mask_binary.float() * density_high_response).sum(dim=[2, 3])  # [B, 1]
    union = (pred_mask_binary.float() + density_high_response - 
             pred_mask_binary.float() * density_high_response).sum(dim=[2, 3])  # [B, 1]
    
    iou = (intersection + 1e-6) / (union + 1e-6)  # [B, 1]
    iou_loss = 1.0 - iou.mean()  # 转换为损失（越小越好）
    
    # 总损失
    total_loss = mse_weight * mse_loss + iou_weight * iou_loss
    
    return total_loss