# check_sa1b_format.py
import os
import json

def check_sa1b_structure(data_path):
    """检查SA-1B格式的数据结构"""
    print("检查SA-1B格式数据结构...")
    
    # 检查必要的文件和目录
    required_items = [
        "images/",
        "annotations/",
        "annotations/sa1b_annotations.json"  # 或类似的标注文件
    ]
    
    for item in required_items:
        item_path = os.path.join(data_path, item)
        if os.path.exists(item_path):
            print(f"✅ 找到: {item}")
            
            # 如果是目录，统计文件数量
            if item.endswith('/'):
                files = os.listdir(item_path)
                print(f"   包含 {len(files)} 个文件/子目录")
        else:
            print(f"❌ 缺失: {item}")
    
    # 检查标注文件内容
    annotation_file = os.path.join(data_path, "annotations/sa1b_annotations.json")
    if os.path.exists(annotation_file):
        try:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            if isinstance(annotations, list):
                print(f"标注文件包含 {len(annotations)} 个标注项")
            elif isinstance(annotations, dict):
                images_count = len(annotations.get('images', []))
                annotations_count = len(annotations.get('annotations', []))
                print(f"标注文件包含 {images_count} 个图像, {annotations_count} 个标注")
            else:
                print("标注文件格式不明确")
                
        except Exception as e:
            print(f"读取标注文件出错: {e}")

if __name__ == "__main__":
    data_path = r"data\monuseg_sa1b"
    check_sa1b_structure(data_path)