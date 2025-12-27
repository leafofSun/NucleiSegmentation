import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from DataLoader import TrainingDataset
import torch

def check_data():
    # 初始化数据集
    dataset = TrainingDataset(
        data_dir="data/MoNuSeg_SA1B", 
        image_size=256, 
        mode='train', 
        requires_name=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # 随机抽查 5 张图
    count = 0
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
        except Exception as e:
            print(f"Error loading index {i}: {e}")
            continue
            
        mask = sample['label'][0].numpy()
        image = sample['image'].permute(1, 2, 0).numpy().astype(np.uint8)
        
        # 统计像素点
        pixel_count = mask.sum()
        print(f"Checking Sample {i}: Name={sample['name']}, Mask Pixels={pixel_count}")
        
        if pixel_count > 0:
            # 画图看看
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f"Image {sample['name']}")
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Mask (Pixels: {pixel_count})")
            
            plt.savefig(f"debug_data_{count}.png")
            print(f"✅ Saved debug_data_{count}.png")
            
            count += 1
            if count >= 3: # 只要找到 3 张有东西的就停止
                break
        else:
            print("❌ Empty Mask! Skipping...")

    if count == 0:
        print("\n😱 致命错误：遍历了数据集，没有找到任何一张带有前景的 Mask！")
        print("请检查 DataLoader.py 里的 JSON 解析逻辑或文件路径匹配逻辑。")

if __name__ == "__main__":
    check_data()