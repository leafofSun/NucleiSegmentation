import cv2
import numpy as np
import glob
import os

# 配置
LABEL_DIR = "data/MoNuSeg_Processed/test/Labels"
SAVE_DIR = "vis_check"
os.makedirs(SAVE_DIR, exist_ok=True)

files = glob.glob(os.path.join(LABEL_DIR, "*.png"))

print(f"正在生成可视化预览图到 {SAVE_DIR} ...")

for f in files[:3]: # 只看前3张
    # 读取原始 Mask (16位)
    mask = cv2.imread(f, -1)
    
    # 归一化到 0-255 以便显示
    # 将 ID 映射到 0-255: (mask * 20) 让 ID=1 变成 20, ID=10 变成 200，看起来就亮了
    vis_mask = (mask * 30).astype(np.uint8) 
    
    # 或者用伪彩色看起来更清楚
    vis_color = cv2.applyColorMap(vis_mask, cv2.COLORMAP_JET)
    # 把背景变黑
    vis_color[mask == 0] = [0, 0, 0]
    
    save_path = os.path.join(SAVE_DIR, "vis_" + os.path.basename(f))
    cv2.imwrite(save_path, vis_color)
    print(f"已保存: {save_path}")

print("去 vis_check 文件夹中查看可视化预览图")