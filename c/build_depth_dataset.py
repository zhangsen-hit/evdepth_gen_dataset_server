#!/usr/bin/env python3
"""
合并 depth.npz 与 events_tensor.npz，按时间戳对齐，生成单目深度估计数据集。
- 标签：depth.npz 的 depth 字段 (12677, 260, 346)
- 输入：events_tensor.npz 的 event_frames (12786, 2, 260, 346)
- 通过 timestamps 匹配，剔除无法对齐或时间戳差距过大的样本

运行：需在已安装 numpy 的环境中执行，例如：
  python build_depth_dataset.py
  python build_depth_dataset.py --max-gap-ms 7.5   # 放宽到 7.5ms
"""

import argparse
import os
from pathlib import Path

import numpy as np

# 时间戳对齐阈值（秒）。设计为每帧约 5ms，允许的匹配误差不超过此值
MAX_TIMESTAMP_GAP_SEC = 0.005  # 5ms


def load_timestamps_only(depth_npz_path, events_npz_path):
    """仅加载两个 npz 的时间戳到内存，用于匹配。"""
    depth_data = np.load(depth_npz_path, allow_pickle=True)
    events_data = np.load(events_npz_path, allow_pickle=True)
    # depth 里是 'timestamp'，events 里是 'timestamps'
    depth_ts = np.asarray(depth_data["timestamp"], dtype=np.float64)
    event_ts = np.asarray(events_data["timestamps"], dtype=np.float64)
    depth_data.close()
    events_data.close()
    return depth_ts, event_ts


def match_timestamps(depth_ts, event_ts, max_gap_sec=MAX_TIMESTAMP_GAP_SEC):
    """
    为每个 depth 帧找到时间戳最近的 event 帧。
    若多个 depth 匹配到同一 event，只保留时间差最小的那一对。
    返回: list of (depth_idx, event_idx, time_diff_sec)
    """
    n_depth = len(depth_ts)
    n_event = len(event_ts)
    # 为每个 depth 找最近的 event 及时间差
    depth_to_best = []  # (depth_idx, event_idx, abs_diff)
    for i in range(n_depth):
        diffs = np.abs(event_ts - depth_ts[i])
        j = np.argmin(diffs)
        d = float(diffs[j])
        if d <= max_gap_sec:
            depth_to_best.append((i, j, d))

    # 每个 event 最多对应一个 depth：若同一 event 被多个 depth 选中，只保留时间差最小的
    event_to_pair = {}  # event_idx -> (depth_idx, time_diff)
    for depth_idx, event_idx, time_diff in depth_to_best:
        if event_idx not in event_to_pair or time_diff < event_to_pair[event_idx][1]:
            event_to_pair[event_idx] = (depth_idx, time_diff)

    # 得到唯一匹配对 (depth_idx, event_idx, time_diff)，按 depth_idx 排序
    matched = [
        (depth_idx, event_idx, time_diff)
        for event_idx, (depth_idx, time_diff) in event_to_pair.items()
    ]
    matched.sort(key=lambda x: x[0])
    return matched


def build_dataset(
    depth_npz_path="depth.npz",
    events_npz_path="events_tensor.npz",
    out_dir="depth_dataset",
    max_gap_sec=MAX_TIMESTAMP_GAP_SEC,
):
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir)

    print("加载时间戳...")
    depth_ts, event_ts = load_timestamps_only(depth_npz_path, events_npz_path)
    n_depth, n_event = len(depth_ts), len(event_ts)
    print(f"  depth 帧数: {n_depth},  event 帧数: {n_event}")

    print("按时间戳匹配...")
    matched = match_timestamps(depth_ts, event_ts, max_gap_sec)
    n_matched = len(matched)
    if n_matched == 0:
        print("没有在阈值内匹配到任何样本，请放宽 max_gap_sec 或检查时间戳单位。")
        return

    time_diffs = np.array([m[2] for m in matched])
    mean_gap = float(np.mean(time_diffs))
    max_gap_matched = float(np.max(time_diffs))
    print(f"匹配成功数量: {n_matched}")
    print(f"平均时间戳对齐误差: {mean_gap*1000:.4f} ms")
    print(f"匹配对中最大时间差: {max_gap_matched*1000:.4f} ms")

    # 使用 mmap 打开大 npz，避免一次性读入内存
    depth_data = np.load(depth_npz_path, mmap_mode="r", allow_pickle=True)
    events_data = np.load(events_npz_path, mmap_mode="r", allow_pickle=True)
    depth_arr = depth_data["depth"]
    event_arr = events_data["event_frames"]

    index_lines = []
    for seq_id, (depth_idx, event_idx, time_diff) in enumerate(matched, start=1):
        # 按 00001, 00002 命名
        fname = f"{seq_id:05d}.npz"
        npz_path = out_path / fname
        # 输入：event 一帧 (2, 260, 346)；标签：depth 一帧 (260, 346)
        input_frame = np.asarray(event_arr[event_idx], dtype=np.float32)
        label_frame = np.asarray(depth_arr[depth_idx], dtype=np.float32)
        # 时间戳取输入的（event）时间戳
        timestamp = float(events_data["timestamps"][event_idx])
        frame_id = seq_id
        np.savez_compressed(
            npz_path,
            input=input_frame,
            label=label_frame,
            timestamp=np.array(timestamp, dtype=np.float64),
            frame_id=np.int32(frame_id),
        )
        index_lines.append(f"{frame_id} {timestamp} {fname}\n")
        if seq_id % 500 == 0:
            print(f"  已写入 {seq_id}/{n_matched}")

    depth_data.close()
    events_data.close()

    index_file = out_path / "index.txt"
    with open(index_file, "w", encoding="utf-8") as f:
        f.writelines(index_lines)
    print(f"已写入 {n_matched} 个 npz 与 {index_file}")

    # 统计汇总
    print("\n======== 统计汇总 ========")
    print(f"原始 depth 帧数: {n_depth}")
    print(f"原始 event 帧数: {n_event}")
    print(f"时间戳正确匹配数量: {n_matched}")
    print(f"平均时间戳对齐误差: {mean_gap*1000:.4f} ms")
    print(f"输出目录: {out_path.absolute()}")
    print(f"索引文件: {index_file.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按时间戳对齐 depth 与 events，生成单目深度数据集")
    parser.add_argument("--depth", default="depth.npz", help="depth.npz 路径")
    parser.add_argument("--events", default="events_tensor.npz", help="events_tensor.npz 路径")
    parser.add_argument("--out-dir", default="./dataset", help="输出目录")
    parser.add_argument("--max-gap-ms", type=float, default=3.0, help="最大允许时间戳误差(ms)，默认 5")
    args = parser.parse_args()
    max_gap_sec = args.max_gap_ms / 1000.0
    build_dataset(
        depth_npz_path=args.depth,
        events_npz_path=args.events,
        out_dir=args.out_dir,
        max_gap_sec=max_gap_sec,
    )
