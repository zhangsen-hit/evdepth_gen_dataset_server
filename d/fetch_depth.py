import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import cv2
import numpy.ma as ma
import matplotlib.pyplot as plt
# import os
import argparse
import tempfile

def visualize_depth_map(npz_file, save_dir):
    # 加载数据
    data = np.load(npz_file)
    label = data['label']
    frame_id = data['frame_id']
    
    # 打印深度图尺寸
    print(f"深度图尺寸: {label.shape}")
    
    valid_mask = np.isfinite(label)
    h, w = label.shape
    # 对数域显示：在给定深度范围 [0.5, 80] 上做对数归一化，再线性映射到 colormap（红近蓝远）
    depth_min, depth_max = 0.5, 80.0
    log_min = np.log(depth_min)
    log_max = np.log(depth_max)
    log_range = log_max - log_min
    log_field = np.empty((h, w), dtype=np.float64)
    log_field.fill(np.nan)
    if np.any(valid_mask):
        # 将深度裁剪到 [depth_min, depth_max]，然后做对数归一化：
        # log_normalized = (ln(d) - ln(depth_min)) / (ln(depth_max) - ln(depth_min))
        d_safe = label[valid_mask].astype(np.float64)
        d_clipped = np.clip(d_safe, depth_min, depth_max)
        log_d = np.log(d_clipped)
        if log_range < 1e-12:
            log_field[valid_mask] = 0.5
        else:
            log_field[valid_mask] = (log_d - log_min) / log_range
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
    cbar.set_label('log-normalized depth (0.5–80)')
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

def write_back_dense_depth_npz(npz_in_file: str, label_out: np.ndarray, npz_out_file: str) -> None:
    with np.load(npz_in_file, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    out["label"] = label_out

    out_dir = os.path.dirname(os.path.abspath(npz_out_file)) or "."
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_depth_", suffix=".npz", dir=out_dir)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **out)
        os.replace(tmp_path, npz_out_file)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def main():
    parser = argparse.ArgumentParser(description='生成NPZ文件中的深度图PNG')
    parser.add_argument('--file', type=str, default='./dataset/02000.npz', help='NPZ文件路径')
    parser.add_argument('--save_dir', type=str, default='.', help='保存目录')
    parser.add_argument('--overwrite_npz', action='store_true', help='将增密+均值滤波后的深度写回并覆盖原npz')
    parser.add_argument('--out_npz', type=str, default=None, help='将结果另存为新的npz路径（不覆盖原文件）')
    args = parser.parse_args()

    # 先生成 PNG 可视化
    visualize_depth_map(args.file, args.save_dir)

    # 可选：写回 npz（覆盖或另存）
    if args.overwrite_npz or args.out_npz:
        with np.load(args.file, allow_pickle=True) as data:
            label = data["label"]
            valid_mask = np.isfinite(label)
            h, w = label.shape

            depth_min, depth_max = 0.5, 80.0
            log_min = np.log(depth_min)
            log_max = np.log(depth_max)
            log_range = log_max - log_min

            log_field = np.empty((h, w), dtype=np.float64)
            log_field.fill(np.nan)
            if np.any(valid_mask):
                d_safe = label[valid_mask].astype(np.float64)
                d_clipped = np.clip(d_safe, depth_min, depth_max)
                log_d = np.log(d_clipped)
                if log_range < 1e-12:
                    log_field[valid_mask] = 0.5
                else:
                    log_field[valid_mask] = (log_d - log_min) / log_range

            split_row = int((1.8 * h) / 3.0)
            vals = np.zeros((h, w), dtype=np.float32)
            vals[valid_mask] = log_field[valid_mask].astype(np.float32, copy=False)
            weights = valid_mask.astype(np.float32)

            kernel = np.ones((7, 7), dtype=np.float32)
            sum_vals = cv2.filter2D(vals, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
            sum_w = cv2.filter2D(weights, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)

            dense_log = np.zeros((h, w), dtype=np.float32)
            dense_valid_mask = sum_w > 0.0
            dense_log[dense_valid_mask] = sum_vals[dense_valid_mask] / sum_w[dense_valid_mask]

            dense_log_out = log_field.astype(np.float32, copy=True)
            dense_valid_mask_out = valid_mask.copy()
            dense_log_out[split_row:, :] = dense_log[split_row:, :]
            dense_valid_mask_out[split_row:, :] = dense_valid_mask[split_row:, :]

            # 将“归一化对数域”转回真实深度：ln(d)=ln(dmin)+u*(ln(dmax)-ln(dmin))
            label_out = np.full((h, w), np.nan, dtype=label.dtype if np.issubdtype(label.dtype, np.floating) else np.float32)
            if log_range < 1e-12:
                label_out[dense_valid_mask_out] = depth_min
            else:
                ln_d = log_min + dense_log_out.astype(np.float64) * log_range
                label_out[dense_valid_mask_out] = np.exp(ln_d[dense_valid_mask_out]).astype(label_out.dtype, copy=False)

        target = args.file if args.overwrite_npz else args.out_npz
        write_back_dense_depth_npz(args.file, label_out, target)
        print(f"已将增密+均值滤波后的深度写回到: {target}")

if __name__ == "__main__":
    main()
