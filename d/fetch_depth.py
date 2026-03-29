import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def visualize_depth_map(npz_file, save_dir):
    # 加载数据
    data = np.load(npz_file)
    label = data['label']
    frame_id = data['frame_id']
    
    # 打印深度图尺寸
    print(f"深度图尺寸: {label.shape}")
    
    # 处理inf
    valid_mask = np.isfinite(label)
    label_vis = np.copy(label)
    if np.any(valid_mask):
        max_val = label_vis[valid_mask].max()
        label_vis[~valid_mask] = max_val if max_val > 0 else 0
    else:
        label_vis[:] = 0
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_path_noted = os.path.join(save_dir, f'frame_depth_noted.png')
    save_path_plain = os.path.join(save_dir, f'frame_depth.png')
    
    # 带注解的深度图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(label_vis, cmap='viridis', aspect='auto')
    plt.colorbar(im)
    plt.title(f'Frame {frame_id} Depth Map')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.tight_layout()
    plt.savefig(save_path_noted, dpi=150)
    plt.close()
    
    # 无注解的深度图，透明显示无效区域
    norm = (label_vis - label_vis.min()) / (label_vis.max() - label_vis.min() + 1e-8)
    cmap = plt.get_cmap('viridis')
    rgba_img = cmap(norm)
    rgba_img[..., 3] = valid_mask.astype(float)  # 有效区域alpha=1，无效区域alpha=0
    plt.imsave(save_path_plain, rgba_img)
    
    print(f"带注解深度图已保存到: {save_path_noted}")
    print(f"无注解深度图已保存到: {save_path_plain}")

def main():
    parser = argparse.ArgumentParser(description='生成NPZ文件中的深度图PNG')
    parser.add_argument('--file', type=str, default='./dataset/02000.npz', help='NPZ文件路径')
    parser.add_argument('--save_dir', type=str, default='.', help='保存目录')
    args = parser.parse_args()
    visualize_depth_map(args.file, args.save_dir)

if __name__ == "__main__":
    main()
