#!/usr/bin/env python3
import argparse
import glob
import os
import tempfile
from typing import Dict, Tuple

import numpy as np
import cv2


def _atomic_save_npz(npz_out_path: str, arrays: Dict[str, np.ndarray]) -> None:
    out_dir = os.path.dirname(os.path.abspath(npz_out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_npz_", suffix=".npz", dir=out_dir)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, npz_out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _densify_label_like_fetch_depth(
    label: np.ndarray,
    *,
    depth_min: float = 0.5,
    depth_max: float = 80.0,
    kernel_size: int = 7,
) -> np.ndarray:
    """
    规则对齐 d/fetch_depth.py：
    - 在给定深度范围 [depth_min, depth_max] 上做对数归一化；
    - 对归一化对数域做 7x7 masked mean（只用有效像素参与平均）；
    - 仅将结果写回下方区域：split_row=int((1.8*h)/3)，即上方约 1.8/3 保持不变；
    - 最终把归一化对数域再还原成深度值，输出到 label。
    """
    if label.ndim != 2:
        raise ValueError(f"label must be 2D (H,W), got shape={label.shape}")

    valid_mask = np.isfinite(label)
    h, w = label.shape

    log_min = float(np.log(depth_min))
    log_max = float(np.log(depth_max))
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

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    sum_vals = cv2.filter2D(vals, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    sum_w = cv2.filter2D(weights, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)

    dense_log = np.zeros((h, w), dtype=np.float32)
    dense_valid_mask = sum_w > 0.0
    dense_log[dense_valid_mask] = sum_vals[dense_valid_mask] / sum_w[dense_valid_mask]

    dense_log_out = log_field.astype(np.float32, copy=True)
    dense_valid_mask_out = valid_mask.copy()
    dense_log_out[split_row:, :] = dense_log[split_row:, :]
    dense_valid_mask_out[split_row:, :] = dense_valid_mask[split_row:, :]

    # 还原深度：ln(d)=ln(dmin)+u*(ln(dmax)-ln(dmin))
    out_dtype = label.dtype if np.issubdtype(label.dtype, np.floating) else np.float32
    label_out = np.full((h, w), np.nan, dtype=out_dtype)

    if log_range < 1e-12:
        label_out[dense_valid_mask_out] = depth_min
    else:
        ln_d = log_min + dense_log_out.astype(np.float64) * log_range
        label_out[dense_valid_mask_out] = np.exp(ln_d[dense_valid_mask_out]).astype(out_dtype, copy=False)

    # 上方区域严格保留原始 label（包括 NaN/inf）
    label_out[:split_row, :] = label[:split_row, :].astype(out_dtype, copy=False)
    return label_out


def process_one(npz_path: str, *, inplace: bool) -> Tuple[str, int, int]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "label" not in data.files:
            raise KeyError(f"{npz_path} missing key 'label'")
        arrays = {k: data[k] for k in data.files}

    label = arrays["label"]
    label_new = _densify_label_like_fetch_depth(label)
    arrays["label"] = label_new

    out_path = npz_path if inplace else (os.path.splitext(npz_path)[0] + "_dense.npz")
    _atomic_save_npz(out_path, arrays)
    return out_path, int(label.shape[0]), int(label.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch densify NPZ depth labels with 7x7 masked mean filter")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="包含 npz 的目录")
    parser.add_argument("--pattern", type=str, default="*.npz", help="匹配文件名的 glob（默认 *.npz）")
    parser.add_argument("--inplace", action="store_true", help="原地覆盖写回（默认：否，另存 *_dense.npz）")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    paths = sorted(glob.glob(os.path.join(dataset_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No npz files found in {dataset_dir} with pattern {args.pattern!r}")

    ok = 0
    for p in paths:
        out_path, h, w = process_one(p, inplace=args.inplace)
        ok += 1
        print(f"[{ok}/{len(paths)}] {os.path.relpath(p)} -> {os.path.relpath(out_path)}  (label {h}x{w})")


if __name__ == "__main__":
    main()

