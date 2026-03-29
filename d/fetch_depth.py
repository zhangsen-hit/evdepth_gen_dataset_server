import numpy as np
import numpy.ma as ma
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
    
    valid_mask = np.isfinite(label)
    h, w = label.shape
    # 对数域显示：log1p(max(d,0))，再线性映射到 colormap（红近蓝远）
    log_field = np.empty((h, w), dtype=np.float64)
    log_field.fill(np.nan)
    if np.any(valid_mask):
        d_safe = np.maximum(label[valid_mask].astype(np.float64), 0.0)
        log_field[valid_mask] = np.log1p(d_safe)
        z_min = float(np.min(log_field[valid_mask]))
        z_max = float(np.max(log_field[valid_mask]))
    else:
        z_min, z_max = 0.0, 1.0
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_path_noted = os.path.join(save_dir, f'frame_depth_noted.png')
    save_path_plain = os.path.join(save_dir, f'frame_depth.png')
    
    cmap = plt.get_cmap('jet_r').copy()
    cmap.set_bad(color='k')
    
    # 带注解的深度图（jet_r：红近蓝远；无效为黑）
    masked = ma.masked_where(~valid_mask, log_field)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(masked, cmap=cmap, vmin=z_min, vmax=z_max, aspect='auto')
    cbar = plt.colorbar(im)
    cbar.set_label('log1p(depth)')
    plt.title(f'Frame {frame_id} Depth Map')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.tight_layout()
    plt.savefig(save_path_noted, dpi=150)
    plt.close()
    
    # 无注解深度图：同一着色；无效为黑色不透明
    norm = np.zeros((h, w), dtype=np.float64)
    if np.any(valid_mask):
        rng = z_max - z_min
        if rng < 1e-12:
            norm[valid_mask] = 0.5
        else:
            norm[valid_mask] = (log_field[valid_mask] - z_min) / rng
    rgba_img = cmap(norm)
    rgba_img[~valid_mask, :3] = 0.0
    rgba_img[..., 3] = 1.0
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
