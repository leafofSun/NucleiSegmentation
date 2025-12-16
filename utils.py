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


def get_boxes_from_mask(mask, box_num=1, std=0.1, max_pixel=5):
    """
    从 mask 中提取边界框，支持空 mask 的健壮处理
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    if mask.max() > 1:
        label_img = mask.astype(int)
    else:
        label_img = label(mask)
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
    error[pred != gt] = 1

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
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x,y] == 0 and one_gt[x,y] == 1:
                label = 1
            elif one_pred[x,y] == 1 and one_gt[x,y] == 0:
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
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
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
        # === [修复] 强制转换为 int，解决 OpenCV 类型报错 ===
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
                    # 减去 padding
                    x0_ori = max(0, int(x0 - pad[1] + 0.5))
                    y0_ori = max(0, int(y0 - pad[0] + 0.5))
                    x1_ori = min(ori_w, int(x1 - pad[1] + 0.5))
                    y1_ori = min(ori_h, int(y1 - pad[0] + 0.5))
                else:
                    # 缩放还原
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
                # mask[y_ori, x_ori] = color # 避免越界或类型错误，使用drawMarker代替
                cv2.drawMarker(mask, (int(x_ori), int(y_ori)), color, markerType=cv2.MARKER_CROSS, markerSize=7, thickness=2)  
    
    os.makedirs(save_path, exist_ok=True)
    mask_path = os.path.join(save_path, f"{mask_name}")
    cv2.imwrite(mask_path, np.uint8(mask))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):
    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss