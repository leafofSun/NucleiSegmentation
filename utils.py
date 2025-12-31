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
# 辅助函数部分 (保持原逻辑，微调健壮性)
# ==========================================

def get_boxes_from_mask(mask, box_num=1, std=0.1, max_pixel=5):
    """
    从 mask 中提取边界框，支持空 mask 的健壮处理
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # 如果 mask 包含 255 (Ignore)，我们在生成 Box 时只看 1 (Target)
    # 忽略 255 区域
    mask_for_box = np.zeros_like(mask)
    mask_for_box[mask == 1] = 1
    
    if mask_for_box.max() > 0:
        label_img = label(mask_for_box)
    else:
        # 全是背景或忽略
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
    # 注意：如果 gt 含有 255，这里需要处理，但在生成点提示时
    # 通常我们只关心 1 (前景) 和 0 (背景)，255 区域不应该采样点
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
            # 如果没有错误，随机撒点 (注意这里范围改为了动态获取，防止写死 256)
            h, w = one_pred.shape
            indices = np.random.randint(0, min(h, w), size=(point_num, 2))
            selected_indices = indices
            
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            # 逻辑：只针对 0 和 1 产生提示
            if one_gt[x,y] == 1:
                label = 1
            elif one_gt[x,y] == 0:
                label = 0
            else:
                # 如果采样到了 255 区域，虽然理论上前面 error 过滤了，但做个兜底
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
        # 防止采样越界
        if fg_size > 0:
            fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
            fg_coords_sel = fg_coords[fg_indices]
            fg_labels = np.ones(num_fg)
        else:
            fg_coords_sel = np.empty((0, 2))
            fg_labels = np.array([])
            num_bg = get_point # 全采背景

        if bg_size > 0:
            bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
            bg_coords_sel = bg_coords[bg_indices]
            bg_labels = np.zeros(num_bg)
        else:
            bg_coords_sel = np.empty((0, 2))
            bg_labels = np.array([])

        coords = np.concatenate([fg_coords_sel, bg_coords_sel], axis=0)
        labels = np.concatenate([fg_labels, bg_labels]).astype(int)
        
        # 如果凑不够点，补齐
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
# 🔥 [关键修改] 支持 Ignore Index (255) 的 Loss
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W] Logits
        mask: [B, 1, H, W] 0, 1, 255
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        
        # 1. 筛选有效像素 (不等于 255)
        valid_mask = (mask != self.ignore_index)
        
        # 极端情况处理：全是 255
        if valid_mask.sum() == 0:
            return pred.sum() * 0.0

        # 2. 展平并取有效值
        p = torch.sigmoid(pred)[valid_mask]   # [N]
        targets = mask[valid_mask]            # [N] (只有 0 和 1)
        
        # 3. 计算 Focal Logic
        # Targets 此时应该是 0 或 1
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
        
        # 1. 筛选有效像素
        valid_mask = (mask != self.ignore_index)
        
        if valid_mask.sum() == 0:
            return pred.sum() * 0.0

        # 2. 取有效值
        p = torch.sigmoid(pred)[valid_mask]
        targets = mask[valid_mask]

        # 3. 计算 Dice
        intersection = torch.sum(p * targets)
        union = torch.sum(p) + torch.sum(targets)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(MaskIoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W] (含 255)
        pred_iou: [B, 1] 网络的 IoU Head 输出
        """
        assert pred_mask.shape == ground_truth_mask.shape
        
        # 计算真实 IoU (GT IoU)，必须忽略 255 区域
        
        # 创建 Valid Mask [B, 1, H, W]
        valid = (ground_truth_mask != self.ignore_index)
        
        # 为了按 Batch 计算，我们不能简单的 flatten 所有，需要保留 Batch 维度
        # 方法：把 255 区域在 pred 和 gt 中都置为 0，不参与 Intersection 和 Union 的贡献
        
        p = torch.sigmoid(pred_mask)
        gt = ground_truth_mask.clone()
        
        # 把 ignore 区域强制设为 0，防止影响 sum
        # 注意：p 也要在 ignore 区域设为 0，否则 union 会算多
        p = p * valid.float()
        gt = gt * valid.float()
        
        # 按 Batch 求和: [B, 1, H, W] -> [B, 1]
        intersection = torch.sum(p * gt, dim=(1, 2, 3))
        union = torch.sum(p, dim=(1, 2, 3)) + torch.sum(gt, dim=(1, 2, 3)) - intersection
        
        # 真实 IoU
        gt_iou = (intersection + 1e-7) / (union + 1e-7)
        
        # MSE Loss
        iou_loss = torch.mean((gt_iou - pred_iou.squeeze()) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    def __init__(self, weight=20.0, iou_scale=1.0, ignore_index=255):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.ignore_index = ignore_index
        
        # 初始化子 Loss
        self.focal_loss = FocalLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.maskiou_loss = MaskIoULoss(ignore_index=ignore_index)

    def forward(self, pred, mask, pred_iou):
        """
        pred: Mask Logits
        mask: GT Mask (0, 1, 255)
        pred_iou: Predicted IoU Score
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        
        loss1 = self.weight * focal_loss + dice_loss
        
        # 如果模型有 IoU Head 输出，则计算 IoU Loss
        if pred_iou is not None:
            loss2 = self.maskiou_loss(pred, mask, pred_iou)
            loss = loss1 + loss2 * self.iou_scale
        else:
            loss = loss1
            
        return loss, {'focal': focal_loss.item(), 'dice': dice_loss.item()}