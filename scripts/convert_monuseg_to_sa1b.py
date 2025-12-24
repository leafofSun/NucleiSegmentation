import os
import glob
import shutil
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
# 必须引入 pycocotools 来生成 RLE Mask
from pycocotools import mask as mask_utils

def parse_xml_to_annotations(xml_path, img_height, img_width):
    """
    解析 MoNuSeg XML -> 多边形 -> Binary Mask -> RLE
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        
        # 查找所有 Region (每个 Region 代表一个细胞)
        regions = root.findall('.//Region')
        
        for region in regions:
            vertices = region.findall('.//Vertex')
            coords = []
            for v in vertices:
                x = float(v.get('X'))
                y = float(v.get('Y'))
                coords.append([x, y])
            
            if len(coords) < 3: continue # 忽略不成形的点
                
            # 1. 生成多边形 Mask
            # 创建一个全黑的底图
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            poly_points = np.array(coords, dtype=np.int32)
            # 填充多边形区域为 1
            cv2.fillPoly(mask, [poly_points], 1)
            
            # 2. 计算 Bounding Box
            x_min = np.min(poly_points[:, 0])
            x_max = np.max(poly_points[:, 0])
            y_min = np.min(poly_points[:, 1])
            y_max = np.max(poly_points[:, 1])
            w = x_max - x_min
            h = y_max - y_min
            
            # 过滤极小框
            if w < 2 or h < 2: continue
            
            # 3. 编码为 RLE (Run-Length Encoding)
            # RLE 需要列优先 (Fortran-style)
            mask_fortran = np.asfortranarray(mask)
            rle = mask_utils.encode(mask_fortran)
            # 将 bytes 解码为 string 以存入 JSON
            rle['counts'] = rle['counts'].decode('utf-8')
            
            annotations.append({
                "bbox": [int(x_min), int(y_min), int(w), int(h)],
                "area": int(mask_utils.area(rle)),
                "segmentation": rle,  # 这里的 mask 是精确的形状
                "iscrowd": 0,
                "category_id": 1
            })
            
        return annotations
    except Exception as e:
        print(f"❌ Error parsing XML {xml_path}: {e}")
        return []

def process_monuseg_pair(img_path, xml_path, out_dir):
    filename = os.path.basename(img_path)
    file_id = os.path.splitext(filename)[0]
    
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Error reading image: {img_path}")
        return
    h, w = img.shape[:2]
    
    # 2. 解析标注 (包含 RLE Mask)
    annotations = parse_xml_to_annotations(xml_path, h, w)
    
    if len(annotations) == 0:
        print(f"⚠️ No valid annotations for {filename}")
    
    # 3. 构建 JSON
    json_data = {
        "image": {
            "file_name": filename,
            "height": h,
            "width": w,
            "id": file_id
        },
        "annotations": annotations
    }
    
    # 4. 保存
    shutil.copy2(img_path, os.path.join(out_dir, filename))
    with open(os.path.join(out_dir, file_id + '.json'), 'w') as f:
        json.dump(json_data, f)

def convert_monuseg(src_root, dst_root):
    # 准备目录
    train_out = os.path.join(dst_root, 'train')
    test_out = os.path.join(dst_root, 'test')
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)
    
    print(f"🚀 开始转换 MoNuSeg (带RLE掩码) 到: {dst_root}")
    
    # --- 1. 处理 Train Set ---
    print("\nProcessing Train Set...")
    train_img_dir = os.path.join(src_root, 'Train', 'Tissue Images')
    train_xml_dir = os.path.join(src_root, 'Train', 'Annotations')
    
    train_images = glob.glob(os.path.join(train_img_dir, '*.tif'))
    for img_path in tqdm(train_images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(train_xml_dir, stem + '.xml')
        
        if os.path.exists(xml_path):
            process_monuseg_pair(img_path, xml_path, train_out)
        else:
            print(f"⚠️ Missing XML: {stem}")

    # --- 2. 处理 Test Set ---
    print("\nProcessing Test Set...")
    test_dir = os.path.join(src_root, 'Test')
    test_images = glob.glob(os.path.join(test_dir, '*.tif'))
    
    for img_path in tqdm(test_images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(test_dir, stem + '.xml')
        
        if os.path.exists(xml_path):
            process_monuseg_pair(img_path, xml_path, test_out)
        else:
            print(f"⚠️ Missing XML: {stem}")
            
    print(f"\n✅ MoNuSeg 转换完成！")

if __name__ == '__main__':
    # 你的路径
    SRC_ROOT = 'data/MoNuSeg'
    DST_ROOT = 'data/MoNuSeg_SA1B_RLE'
    
    convert_monuseg(SRC_ROOT, DST_ROOT)