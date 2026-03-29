#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成多个位姿的深度图
"""

from generate_depth_map import (
    prepare_depth_batch_context,
    set_batch_context_for_workers,
    batch_worker_pose,
)
import os
import argparse
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def _collect_with_progress(result_iter, n_total):
    """
    消费迭代器并收集结果；按约 1% 打印一行进度（无第三方库）。
    与 ex.map 顺序一致时，表示「已按序完成多少项」，非真实 wall-clock 比例。
    """
    raw_results = []
    if n_total <= 0:
        return raw_results
    last_pct = -1
    for i, r in enumerate(result_iter, start=1):
        raw_results.append(r)
        approx_pct = min(100, (i * 100) // n_total)
        if approx_pct != last_pct:
            print(f"[进度] 约 {approx_pct}%  ({i}/{n_total})", flush=True)
            last_pct = approx_pct
    return raw_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量生成深度图")
    parser.add_argument("--pcd", type=str, default="scans.pcd", help="点云文件路径")
    parser.add_argument("--odom", type=str, default="odometry.txt", help="位姿文件路径")
    parser.add_argument("--calib", type=str, default="single_calib_result.txt", help="相机标定文件路径")
    parser.add_argument(
        "--extrinsic",
        type=str,
        default="multi_calib_result_modified.txt",
        help="外参标定文件路径（激光雷达到相机，默认：multi_calib_result.txt）",
    )
    parser.add_argument("--start", type=int, default=0, help="起始位姿索引")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="结束位姿索引（不包含），默认到最后一个",
    )
    parser.add_argument("--step", type=int, default=1, help="步长（每隔几个位姿生成一个）")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="并行进程数，默认使用 min(CPU 核数, 待处理帧数)；设为 1 则禁用并行",
    )
    parser.add_argument(
        "--extrinsic_pcl_in_lidar_frame",
        action="store_true",
        help="若标定中 Pcl 为雷达系下平移则加此选项（与单帧脚本一致）",
    )

    args = parser.parse_args()

    for f in [args.pcd, args.odom, args.calib]:
        if not os.path.exists(f):
            print(f"错误: 文件不存在: {f}")
            exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("加载点云、位姿与标定（仅一次）...")
    ctx = prepare_depth_batch_context(
        args.pcd,
        args.odom,
        args.calib,
        extrinsic_calib_file=args.extrinsic,
        extrinsic_pcl_in_camera_frame=not args.extrinsic_pcl_in_lidar_frame,
    )
    poses = ctx.poses
    print(f"共找到 {len(poses)} 个位姿")

    end_idx = args.end if args.end is not None else len(poses)
    pose_indices = range(args.start, min(end_idx, len(poses)), args.step)
    n_total = len(pose_indices)

    max_workers = args.jobs
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, max(n_total, 1))
    max_workers = max(1, min(max_workers, max(n_total, 1)))

    print(f"\n将生成 {n_total} 个深度图")
    print(f"位姿范围: {args.start} 到 {end_idx - 1}，步长: {args.step}")
    print(f"输出目录: {args.output_dir}")
    print(f"并行进程数: {max_workers}\n")

    success_count = 0
    fail_count = 0

    depth_list = []
    timestamp_list = []
    frame_id_list = []
    pose_index_list = []

    try:
        mp_fork = None
        if max_workers > 1:
            try:
                mp_fork = mp.get_context("fork")
            except ValueError:
                mp_fork = None
            # 避免每个子进程内的 OpenBLAS/MKL 再开多线程，导致 CPU 过度抢占
            for _k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ.setdefault(_k, "1")

        set_batch_context_for_workers(ctx)

        if max_workers > 1 and mp_fork is not None:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_fork) as ex:
                raw_results = _collect_with_progress(
                    ex.map(batch_worker_pose, pose_indices), n_total
                )
        else:
            if max_workers > 1 and mp_fork is None:
                print("提示: 当前环境不支持 fork，已回退为单进程。\n")
            raw_results = _collect_with_progress(
                (batch_worker_pose(i) for i in pose_indices), n_total
            )
    finally:
        set_batch_context_for_workers(None)

    by_idx = {r[0]: r for r in raw_results}

    for k, idx in enumerate(pose_indices):
        print(f"\n[{k + 1}/{n_total}] 位姿 {idx}...")
        tup = by_idx.get(idx)
        if tup is None:
            fail_count += 1
            print("  ❌ 缺少结果")
            continue
        _pi, depth_map, err = tup
        if err is not None:
            fail_count += 1
            print(f"  ❌ 错误: {err}")
            continue
        if depth_map is None:
            fail_count += 1
            print("  ❌ 失败（无有效投影）")
            continue

        success_count += 1
        print("  ✅ 成功")
        depth_list.append(depth_map.astype(np.float32))

        pose_meta = poses[idx] if idx < len(poses) else {}
        secs = pose_meta.get("stamp_secs", None)
        nsecs = pose_meta.get("stamp_nsecs", None)
        if (secs is not None) and (nsecs is not None):
            timestamp = float(secs) + float(nsecs) * 1e-9
        else:
            timestamp = None
        timestamp_list.append(timestamp)

        frame_id = pose_meta.get("frame_id", None)
        frame_id_list.append(frame_id)
        pose_index_list.append(idx)

    if len(depth_list) > 0:
        depth_array = np.stack(depth_list, axis=0)
        timestamp_array = np.array(
            [np.nan if t is None else t for t in timestamp_list],
            dtype=np.float64,
        )
        frame_id_array = np.array(
            ["" if fid is None else str(fid) for fid in frame_id_list],
            dtype=np.str_,
        )
        pose_index_array = np.array(pose_index_list, dtype=np.int32)

        npz_path = os.path.join(args.output_dir, "depth.npz")
        np.savez_compressed(
            npz_path,
            depth=depth_array,
            timestamp=timestamp_array,
            frame_id=frame_id_array,
            pose_index=pose_index_array,
        )

        print(f"\n深度tensor及元信息已保存到: {npz_path}")
        print(f"  depth 形状: {depth_array.shape}")
        print(f"  timestamp 数量: {timestamp_array.shape[0]}")
        print(f"  frame_id 数量: {frame_id_array.shape[0]}")

    print("\n" + "=" * 60)
    print("批量处理完成！")
    print(f"成功: {success_count}, 失败: {fail_count}")
    print("=" * 60)
