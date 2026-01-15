import numpy as np
import cv2
import os
import json
from tqdm import tqdm

# ================= 配置区域 =================
DOWNLOAD_PATH = "data/PanNuke"  
OUTPUT_PATH = "data/PanNuke_SA1B"

# 划分逻辑
SPLITS = {
    "train": ["Fold 1", "Fold 2"],
    "test": ["Fold 3"]
}
# ===========================================

def mask_to_polygons(mask):
    """
    将二值 Mask 转换为多边形坐标 (COCO/SA-1B 格式)
    """
    # 查找轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotations = []
    idx = 1
    
    for contour in contours:
        # 忽略太小的噪点 (例如面积小于 10 像素)
        area = cv2.contourArea(contour)
        if area < 10:
            continue
            
        # 获取 Bounding Box [x, y, w, h]
        x, y, w, h = cv2.boundingRect(contour)
        
        # 展平坐标 [[x1, y1], [x2, y2]] -> [x1, y1, x2, y2, ...]
        poly = contour.flatten().tolist()
        
        # 只有坐标点数大于等于 6 (3个点) 才能构成多边形
        if len(poly) >= 6:
            annotations.append({
                "id": idx,
                "segmentation": [poly], # SA-1B/COCO 要求外层是 list
                "bbox": [x, y, w, h],
                "area": float(area),
                "category_id": 1, # 1 代表细胞核
                "iscrowd": 0
            })
            idx += 1
            
    return annotations

def convert_to_sa1b_json():
    print(f"🚀 Converting PanNuke to Strict SA-1B Format (Image + JSON)...")
    
    global_idx = 0
    
    for split_name, folders in SPLITS.items():
        # 创建 split 文件夹 (train/test)
        save_dir = os.path.join(OUTPUT_PATH, split_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"📂 Processing Split: [{split_name}] -> {save_dir}")
        
        for folder in folders:
            folder_path = os.path.join(DOWNLOAD_PATH, folder)
            
            # 加载数据
            try:
                images = np.load(os.path.join(folder_path, 'images.npy'))
                masks = np.load(os.path.join(folder_path, 'masks.npy'))
                types = np.load(os.path.join(folder_path, 'types.npy'))
            except Exception as e:
                print(f"   ❌ Skipping {folder}: {e}")
                continue
                
            # 遍历
            for i in tqdm(range(len(images)), desc=f"   {folder}"):
                # --- 1. 获取器官名 ---
                raw_type = types[i]
                organ_name = raw_type.decode('utf-8') if isinstance(raw_type, bytes) else str(raw_type)
                
                # --- 2. 处理图像 ---
                img = images[i].astype(np.uint8)
                # 注意：OpenCV 保存需要 BGR，如果原始是 RGB，需要转换
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # --- 3. 处理掩码并生成 Polygon ---
                # 合并 Channel 0-4 为前景
                mask_stack = masks[i]
                foreground = np.sum(mask_stack[..., :5], axis=-1)
                binary_mask = (foreground > 0).astype(np.uint8)
                
                # 如果没有细胞，跳过
                if np.sum(binary_mask) == 0:
                    continue
                
                # 核心步骤：转多边形
                anns = mask_to_polygons(binary_mask)
                
                if not anns: continue # 如果只有噪点，也跳过
                
                # --- 4. 生成文件名和路径 ---
                file_id = f"sa_{global_idx:07d}" # sa_0000001
                img_filename = f"{file_id}.png"
                json_filename = f"{file_id}.json"
                
                img_save_path = os.path.join(save_dir, img_filename)
                json_save_path = os.path.join(save_dir, json_filename)
                
                # --- 5. 构建 JSON 内容 ---
                json_content = {
                    "image_id": file_id,
                    "image_path": img_filename, # 相对路径
                    "organ_type": organ_name,   # 🔥 关键：写入器官类型
                    "width": 256,
                    "height": 256,
                    "annotations": anns
                }
                
                # --- 6. 保存到磁盘 ---
                cv2.imwrite(img_save_path, img_bgr)
                with open(json_save_path, 'w') as f:
                    json.dump(json_content, f)
                
                global_idx += 1
                
    print(f"\n✅ Conversion Complete! Total valid images: {global_idx}")

if __name__ == "__main__":
    convert_to_sa1b_json()