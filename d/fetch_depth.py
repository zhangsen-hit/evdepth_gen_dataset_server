import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import cv2
import numpy.ma as ma
import matplotlib.pyplot as plt
# import os
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
    plt.savefig(save_path_noted, dpi=800)
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

    # 深度增密填充：将 max-dilation 改为 7x7 的平均值滤波（仅对有效像素做 masked mean）。
    # 仅将滤波结果写回下方 1/3；上方 2/3 保持原始深度。
    split_row = int((1.8 * h) / 3.0)  # [0, split_row) 为上方 2/3

    vals = np.zeros((h, w), dtype=np.float32)
    vals[valid_mask] = log_field[valid_mask].astype(np.float32, copy=False)
    weights = valid_mask.astype(np.float32)

    kernel = np.ones((7, 7), dtype=np.float32)
    sum_vals = cv2.filter2D(vals, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    sum_w = cv2.filter2D(weights, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)

    dense_log = np.zeros((h, w), dtype=np.float32)
    dense_valid_mask = sum_w > 0.0
    dense_log[dense_valid_mask] = sum_vals[dense_valid_mask] / sum_w[dense_valid_mask]

    dense_log_out = log_field.copy()
    dense_valid_mask_out = valid_mask.copy()

    dense_log_out[split_row:, :] = dense_log[split_row:, :]
    dense_valid_mask_out[split_row:, :] = dense_valid_mask[split_row:, :]

    norm_dense = np.zeros((h, w), dtype=np.float64)
    if np.any(dense_valid_mask_out):
        rng = z_max - z_min
        if rng < 1e-12:
            norm_dense[dense_valid_mask_out] = 0.5
        else:
            norm_dense[dense_valid_mask_out] = (dense_log_out[dense_valid_mask_out] - z_min) / rng

    rgba_dense = cmap(norm_dense)
    rgba_dense[~dense_valid_mask_out, :3] = 0.0
    rgba_dense[..., 3] = 1.0
    save_path_dense = os.path.join(save_dir, 'frame_depth_dense.png')
    plt.imsave(save_path_dense, rgba_dense)
    
    print(f"带注解深度图已保存到: {save_path_noted}")
    print(f"无注解深度图已保存到: {save_path_plain}")
    print(f"增密(下方1/3 + 7x7 mean filter)深度图已保存到: {save_path_dense}")

def main():
    parser = argparse.ArgumentParser(description='生成NPZ文件中的深度图PNG')
    parser.add_argument('--file', type=str, default='./dataset/02000.npz', help='NPZ文件路径')
    parser.add_argument('--save_dir', type=str, default='.', help='保存目录')
    args = parser.parse_args()
    visualize_depth_map(args.file, args.save_dir)

if __name__ == "__main__":
    main()
