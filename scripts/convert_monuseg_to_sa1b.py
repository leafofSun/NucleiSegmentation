import os
import cv2
import json
import glob
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 你的原始数据路径 (根据你的描述设置)
# 自动定位到项目根目录，再拼出 data/MoNuSeg 的绝对路径，避免重复前缀
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_ROOT = os.path.join(PROJECT_ROOT, "data", "MoNuSeg")
TRAIN_IMG_DIR = os.path.join(RAW_ROOT, "Train", "images")       # 训练集图片
TRAIN_XML_DIR = os.path.join(RAW_ROOT, "Train", "Annotations") # 训练集标注
TEST_DIR = os.path.join(RAW_ROOT, "Test")                      # 测试集 (混在一起)

# 2. 输出路径 (处理好的数据将保存在这里，建议用新名字以免混淆)
OUTPUT_ROOT = "data/MoNuSeg_Processed"

# 3. 通用属性描述 (PNuRL 需要)
COMMON_ATTRS = {
    "prompt": "Microscopy image of H&E stained tissue. The image contains deep purple, rounded or ellipsoidal cell nuclei that are densely distributed against a pink background.",
    "color": "deep purple",
    "shape": "rounded",
    "density": "dense"
}

# ===========================================

def parse_xml_to_mask(xml_path, shape):
    """解析 MoNuSeg XML 生成实例掩码"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return np.zeros(shape[:2], dtype=np.int32)

    # 创建空 Mask
    mask = np.zeros(shape[:2], dtype=np.int32)
    
    # 遍历每个 Region (每个细胞核)
    # MoNuSeg XML 结构: Annotations -> Annotation -> Regions -> Region -> Vertices
    count = 0
    for i, region in enumerate(root.findall(".//Region"), start=1):
        vertices = []
        for vertex in region.findall(".//Vertex"):
            x = float(vertex.get("X"))
            y = float(vertex.get("Y"))
            vertices.append([x, y])
        
        if len(vertices) > 2: # 至少3个点才能围成多边形
            pts = np.array(vertices, np.int32)
            # 填充多边形，使用 i 作为实例 ID (Instance ID)
            cv2.fillPoly(mask, [pts], i)
            count += 1
            
    return mask

def process_subset(image_source_dir, xml_source_dir, mode):
    """
    image_source_dir: 图片所在的文件夹
    xml_source_dir:   XML所在的文件夹 (如果是Test，这两个是同一个路径)
    mode:             'train' 或 'test'
    """
    print(f"\nProcessing {mode} set...")
    print(f"  - Images from: {image_source_dir}")
    print(f"  - XMLs from:   {xml_source_dir}")

    # 创建输出目录
    out_img_dir = os.path.join(OUTPUT_ROOT, mode, 'Images')
    out_lbl_dir = os.path.join(OUTPUT_ROOT, mode, 'Labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 寻找所有图片 (.tif)
    img_files = glob.glob(os.path.join(image_source_dir, "*.tif"))
    
    if len(img_files) == 0:
        print("  ! No .tif images found! Check path.")
        return

    image2label = {}
    attribute_info = {}

    for img_path in tqdm(img_files):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # 1. 寻找对应的 XML
        # 逻辑：在 xml_source_dir 下找 同名.xml
        xml_path = os.path.join(xml_source_dir, name_no_ext + ".xml")
        
        if not os.path.exists(xml_path):
            print(f"  ! Warning: XML not found for {filename}, skipping.")
            continue

        # 2. 读取图片 (TIF -> RGB)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ! Error reading image {img_path}")
            continue

        # 3. 生成 Mask
        mask = parse_xml_to_mask(xml_path, img.shape)

        # 4. 保存为标准格式 (.png)
        save_name = name_no_ext + ".png"
        
        # 保存图片
        cv2.imwrite(os.path.join(out_img_dir, save_name), img)
        
        # 保存 Mask (必须是 uint16 以支持>255个细胞)
        cv2.imwrite(os.path.join(out_lbl_dir, save_name), mask.astype(np.uint16))

        # 5. 记录元数据
        image2label[save_name] = save_name
        attribute_info[save_name] = COMMON_ATTRS.copy()

    # 保存 JSON
    with open(os.path.join(OUTPUT_ROOT, f"image2label_{mode}.json"), 'w') as f:
        json.dump(image2label, f, indent=4)
    
    with open(os.path.join(OUTPUT_ROOT, f"attribute_info_{mode}.json"), 'w') as f:
        json.dump(attribute_info, f, indent=4)
        
    print(f"  ✅ {mode} set done. Saved to {OUTPUT_ROOT}/{mode}")

if __name__ == "__main__":
    # 1. 处理 Train 集
    # 你的结构: data/MoNuSeg/Train/images 和 data/MoNuSeg/Train/Annotations
    if os.path.exists(TRAIN_IMG_DIR) and os.path.exists(TRAIN_XML_DIR):
        process_subset(TRAIN_IMG_DIR, TRAIN_XML_DIR, 'train')
    else:
        print(f"❌ Train paths not found:\n {TRAIN_IMG_DIR}\n {TRAIN_XML_DIR}")

    # 2. 处理 Test 集
    # 你的结构: data/MoNuSeg/Test (里面既有tif也有xml)
    if os.path.exists(TEST_DIR):
        # 此时图片和xml在同一个文件夹，所以传两次同一个路径
        process_subset(TEST_DIR, TEST_DIR, 'test')
    else:
        print(f"❌ Test path not found: {TEST_DIR}")
        
    print("\nProcessing Complete! New dataset is at:", OUTPUT_ROOT)