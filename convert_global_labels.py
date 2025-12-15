"""
转换 global_label_*.json 为 DataLoader.py 期望的 attribute_info 格式

将 PromptNu 风格的全局标签转换为 PNuRL 需要的格式：
- 从列表格式转换为字典格式
- 提取属性文本作为 attribute_prompts
- 根据文本映射到类别索引，生成 attribute_labels
"""

import json
import os
from typing import Dict, List, Any

# 属性类别映射（根据 PNuRL 的默认配置：3, 5, 4, 3, 3）
# 顺序：[颜色, 形状, 排列, 大小, 分布]
ATTRIBUTE_MAPPINGS = {
    # Color (3 classes)
    'color': {
        'deep purple': 0,
        'light pink': 1,
        'other': 2,  # 默认类别
    },
    # Shape (5 classes)
    'shape': {
        'elliptical/oval': 0,
        'spindle': 1,
        'irregular': 2,
        'round': 3,
        'other': 4,  # 默认类别
    },
    # Arrangement (4 classes)
    'arrange': {
        'scattered': 0,
        'clustered': 1,
        'linear': 2,
        'other': 3,  # 默认类别
    },
    # Size (3 classes)
    'size': {
        'small': 0,
        'medium': 1,
        'large': 2,
    },
    # Distribution/Density (3 classes)
    'density': {
        'sparsely distributed': 0,
        'moderately dense': 1,
        'densely packed': 2,
    },
}


def text_to_class_index(text: str, attr_type: str) -> int:
    """
    将属性文本转换为类别索引
    
    Args:
        text: 属性文本（如 "deep purple"）
        attr_type: 属性类型（"color", "shape", "arrange", "size", "density"）
    
    Returns:
        类别索引
    """
    text_lower = text.lower().strip()
    mapping = ATTRIBUTE_MAPPINGS.get(attr_type, {})
    
    # 精确匹配
    if text_lower in mapping:
        return mapping[text_lower]
    
    # 模糊匹配（包含关键词）
    for key, idx in mapping.items():
        if key in text_lower or text_lower in key:
            return idx
    
    # 返回默认类别（通常是最后一个）
    if mapping:
        return max(mapping.values())
    return 0


def class_index_to_onehot(idx: int, num_classes: int) -> List[int]:
    """
    将类别索引转换为 one-hot 编码
    
    Args:
        idx: 类别索引
        num_classes: 类别数量
    
    Returns:
        one-hot 编码列表
    """
    onehot = [0] * num_classes
    if 0 <= idx < num_classes:
        onehot[idx] = 1
    return onehot


def convert_global_labels(
    input_path: str,
    output_path: str,
    num_classes_per_attr: List[int] = [3, 5, 4, 3, 3]
) -> None:
    """
    转换 global_label JSON 文件为 attribute_info 格式
    
    Args:
        input_path: 输入的 global_label JSON 文件路径
        output_path: 输出的 attribute_info JSON 文件路径
        num_classes_per_attr: 每个属性的类别数量 [颜色, 形状, 排列, 大小, 分布]
    """
    # 读取原始文件
    with open(input_path, 'r', encoding='utf-8') as f:
        global_labels = json.load(f)
    
    # 转换后的字典
    attribute_info = {}
    
    # 属性顺序：[颜色, 形状, 排列, 大小, 分布]
    attr_order = ['color', 'shape', 'arrange', 'size', 'density']
    
    for item in global_labels:
        # 获取图像ID
        image_ids = item.get('id', [])
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        
        # 提取属性文本
        color_text = item.get('color', ['other'])[0] if isinstance(item.get('color'), list) else item.get('color', 'other')
        shape_text = item.get('shape', ['other'])[0] if isinstance(item.get('shape'), list) else item.get('shape', 'other')
        arrange_text = item.get('arrange', ['other'])[0] if isinstance(item.get('arrange'), list) else item.get('arrange', 'other')
        size_text = item.get('size', ['medium'])[0] if isinstance(item.get('size'), list) else item.get('size', 'medium')
        density_text = item.get('density', ['moderately dense'])[0] if isinstance(item.get('density'), list) else item.get('density', 'moderately dense')
        
        # 构建 attribute_prompts（按顺序：[颜色, 形状, 排列, 大小, 分布]）
        attribute_prompts = [
            color_text,
            shape_text,
            arrange_text,
            size_text,
            density_text
        ]
        
        # 转换为类别索引并生成 one-hot 编码
        attribute_labels = []
        attr_texts = [color_text, shape_text, arrange_text, size_text, density_text]
        attr_types = ['color', 'shape', 'arrange', 'size', 'density']
        
        for i, (text, attr_type, num_classes) in enumerate(zip(attr_texts, attr_types, num_classes_per_attr)):
            class_idx = text_to_class_index(text, attr_type)
            onehot = class_index_to_onehot(class_idx, num_classes)
            attribute_labels.append(onehot)
        
        # 为每个图像ID创建条目
        for img_id in image_ids:
            # 移除可能的扩展名
            img_id = img_id.split('.')[0] if '.' in img_id else img_id
            
            attribute_info[img_id] = {
                'attribute_prompts': attribute_prompts,
                'attribute_labels': attribute_labels
            }
    
    # 保存转换后的文件
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(attribute_info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 成功转换 {len(attribute_info)} 个图像")
    print(f"  输入文件: {input_path}")
    print(f"  输出文件: {output_path}")
    print(f"  属性类别数: {num_classes_per_attr}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='转换 global_label JSON 为 attribute_info 格式')
    parser.add_argument('--input', type=str, required=True, help='输入的 global_label JSON 文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出的 attribute_info JSON 文件路径')
    parser.add_argument('--num_classes', nargs=5, type=int, default=[3, 5, 4, 3, 3],
                        help='每个属性的类别数量 [颜色, 形状, 排列, 大小, 分布]')
    
    args = parser.parse_args()
    
    convert_global_labels(
        input_path=args.input,
        output_path=args.output,
        num_classes_per_attr=args.num_classes
    )





