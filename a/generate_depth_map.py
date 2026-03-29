#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从全局点云地图生成深度图
根据给定的相机位姿，截取视场角内的点云，并舍弃被遮挡的点云

外参约定（本文件统一）：
  Rcl / Pcl 表示 LiDAR -> Camera，即 p_cam = Rcl @ p_lidar + Pcl。
  位姿由雷达位姿经此外参转换为相机位姿后，再将世界点变换到相机系并投影。
"""

import numpy as np
import open3d as o3d
import cv2
import re
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Optional


def parse_camera_params(calib_file):
    """解析相机内参文件（含畸变 cam_d0..cam_d3，对应 OpenCV 的 k1,k2,p1,p2）"""
    params = {}
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    
    return params


def parse_odometry(odom_file):
    """解析odometry.txt文件，提取所有位姿及其时间戳等元信息

    返回的每个元素为字典，至少包含:
      - 'position': np.array([x, y, z])
      - 'orientation': np.array([qx, qy, qz, qw])

    如果文件中存在对应字段，还会额外包含:
      - 'seq': 序号（int）
      - 'stamp_secs': 时间戳秒（int）
      - 'stamp_nsecs': 时间戳纳秒（int）
      - 'frame_id': 帧ID（str）
    """
    poses = []
    
    with open(odom_file, 'r') as f:
        content = f.read()
    
    # 使用正则表达式提取每个位姿块（以---分隔）
    pose_blocks = re.split(r'^---$', content, flags=re.MULTILINE)
    
    for block_idx, block in enumerate(pose_blocks):
        if 'pose:' not in block or 'position:' not in block:
            continue

        # 可选: 解析header中的seq、时间戳和frame_id
        seq = None
        stamp_secs = None
        stamp_nsecs = None
        frame_id = None

        # seq
        seq_match = re.search(r'seq:\s*([-\d.eE+-]+)', block)
        if seq_match:
            try:
                seq = int(float(seq_match.group(1)))
            except ValueError:
                seq = None

        # 时间戳
        secs_match = re.search(r'secs:\s*([-\d.eE+-]+)', block)
        nsecs_match = re.search(r'nsecs:\s*([-\d.eE+-]+)', block)
        if secs_match and nsecs_match:
            try:
                stamp_secs = int(float(secs_match.group(1)))
                stamp_nsecs = int(float(nsecs_match.group(1)))
            except ValueError:
                stamp_secs = None
                stamp_nsecs = None

        # frame_id: "xxxx"
        frame_id_match = re.search(r'frame_id:\s*\"([^\"]*)\"', block)
        if frame_id_match:
            frame_id = frame_id_match.group(1)
        
        # 提取position（支持科学计数法）
        pos_match = re.search(
            r'position:\s*x:\s*([-\d.eE+-]+)\s*y:\s*([-\d.eE+-]+)\s*z:\s*([-\d.eE+-]+)',
            block,
            re.MULTILINE
        )
        
        # 提取orientation（支持科学计数法）
        ori_match = re.search(
            r'orientation:\s*x:\s*([-\d.eE+-]+)\s*y:\s*([-\d.eE+-]+)\s*z:\s*([-\d.eE+-]+)\s*w:\s*([-\d.eE+-]+)',
            block,
            re.MULTILINE
        )
        
        if pos_match and ori_match:
            try:
                position = np.array([
                    float(pos_match.group(1)),
                    float(pos_match.group(2)),
                    float(pos_match.group(3))
                ])
                
                orientation = np.array([
                    float(ori_match.group(1)),  # x
                    float(ori_match.group(2)),  # y
                    float(ori_match.group(3)),  # z
                    float(ori_match.group(4))  # w
                ])

                pose_dict = {
                    'position': position,
                    'orientation': orientation
                }

                # 只有在成功解析时才附加可选字段
                if seq is not None:
                    pose_dict['seq'] = seq
                if (stamp_secs is not None) and (stamp_nsecs is not None):
                    pose_dict['stamp_secs'] = stamp_secs
                    pose_dict['stamp_nsecs'] = stamp_nsecs
                if frame_id is not None:
                    pose_dict['frame_id'] = frame_id

                poses.append(pose_dict)
            except ValueError as e:
                print(f"  警告: 跳过第 {block_idx} 个位姿块（解析错误: {e}）")
                continue
    
    return poses


def quaternion_to_rotation_matrix(quat):
    """
    将四元数转换为旋转矩阵
    quat: [x, y, z, w]
    返回: 3x3旋转矩阵
    """
    r = R.from_quat(quat)
    return r.as_matrix()


def parse_extrinsic_calib(calib_file):
    """
    解析外参标定文件（FAST-LIVO2 格式）。
    约定：Rcl/Pcl 为 LiDAR -> Camera，即 p_cam = Rcl @ p_lidar + Pcl。
    返回: (Rcl, Pcl)
    Rcl: 激光雷达到相机的旋转矩阵 (3x3)
    Pcl: 平移向量 (3,)，相机系下雷达原点相对相机的位置
    """
    Rcl = None
    Pcl = None
    
    with open(calib_file, 'r') as f:
        content = f.read()
    
    # 解析Rcl（旋转矩阵，按行存储）
    rcl_match = re.search(
        r'Rcl:\s*\[\s*([-\d.eE+-]+),\s*([-\d.eE+-]+),\s*([-\d.eE+-]+),\s*'
        r'([-\d.eE+-]+),\s*([-\d.eE+-]+),\s*([-\d.eE+-]+),\s*'
        r'([-\d.eE+-]+),\s*([-\d.eE+-]+),\s*([-\d.eE+-]+)\s*\]',
        content,
        re.MULTILINE
    )
    
    if rcl_match:
        Rcl = np.array([
            [float(rcl_match.group(1)), float(rcl_match.group(2)), float(rcl_match.group(3))],
            [float(rcl_match.group(4)), float(rcl_match.group(5)), float(rcl_match.group(6))],
            [float(rcl_match.group(7)), float(rcl_match.group(8)), float(rcl_match.group(9))]
        ])
    else:
        raise ValueError(f"无法从 {calib_file} 中解析Rcl")
    
    # 解析Pcl（平移向量）
    pcl_match = re.search(
        r'Pcl:\s*\[\s*([-\d.eE+-]+),\s*([-\d.eE+-]+),\s*([-\d.eE+-]+)\s*\]',
        content,
        re.MULTILINE
    )
    
    if pcl_match:
        Pcl = np.array([
            float(pcl_match.group(1)),
            float(pcl_match.group(2)),
            float(pcl_match.group(3))
        ])
    else:
        raise ValueError(f"无法从 {calib_file} 中解析Pcl")
    
    return Rcl, Pcl


def lidar_pose_to_camera_pose(lidar_position, lidar_orientation, Rcl, Pcl, pcl_in_camera_frame=True):
    """
    将激光雷达位姿转换为相机位姿
    
    参数:
    lidar_position: 激光雷达在世界坐标系下的位置 [x, y, z]
    lidar_orientation: 激光雷达在世界坐标系下的四元数 [x, y, z, w]
    Rcl: 从激光雷达到相机的旋转矩阵 (3x3)
    Pcl: 从激光雷达到相机的平移向量 (3,)
    pcl_in_camera_frame: 若 True，约定 p_cam = Rcl@p_lidar + Pcl（Pcl 在相机系，常见）；
                         若 False，约定 Pcl 为雷达系下雷达到相机的向量，即 t_cam = t_lidar + R_lidar@Pcl
    
    返回:
    camera_position: 相机在世界坐标系下的位置 [x, y, z]
    camera_orientation: 相机在世界坐标系下的四元数 [x, y, z, w]
    """
    # 激光雷达在世界坐标系下的旋转矩阵
    R_lidar_world = quaternion_to_rotation_matrix(lidar_orientation)
    
    # 相机在世界坐标系下的旋转矩阵
    # R_camera_world = R_lidar_world * Rcl^T（相机轴在世界系下的表示）
    R_camera_world = R_lidar_world @ Rcl.T
    
    # 相机在世界坐标系下的位置（两种标定约定）
    if pcl_in_camera_frame:
        # 标准约定：p_cam = Rcl @ p_lidar + Pcl，Pcl 在相机系 → t_cam = t_lidar - R_camera_world @ Pcl
        camera_position = lidar_position - R_camera_world @ Pcl
    else:
        # Pcl 为雷达系下从雷达到相机的向量 → t_cam = t_lidar + R_lidar_world @ Pcl
        camera_position = lidar_position + R_lidar_world @ Pcl
    
    # 将旋转矩阵转换为四元数
    r_camera = R.from_matrix(R_camera_world)
    camera_orientation = r_camera.as_quat()  # 返回 [x, y, z, w]
    
    return camera_position, camera_orientation


def world_to_camera(points_world, position, orientation):
    """
    将世界坐标系下的点转换到相机坐标系
    points_world: Nx3 numpy数组
    position: 相机在世界坐标系下的位置 [x, y, z]
    orientation: 四元数 [x, y, z, w]
    返回: Nx3 相机坐标系下的点
    """
    # 四元数转旋转矩阵
    R_world_to_cam = quaternion_to_rotation_matrix(orientation).T  # 转置，因为我们要world->camera
    
    # 平移向量（相机在世界坐标系下的位置）
    t_world = position
    
    # 转换：P_cam = R^T * (P_world - t)
    points_cam = (points_world - t_world) @ R_world_to_cam.T
    # points_cam = (points_world - t_world) @ R_world_to_cam
    
    return points_cam


def project_to_image(points_cam, camera_matrix, dist_coeffs):
    """
    使用 OpenCV 将相机坐标系下的 3D 点投影到 2D 图像平面（含畸变，投影到原始图像空间）。
    points_cam: Nx3 numpy 数组 (X, Y, Z)，相机坐标系
    camera_matrix: 3x3 内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]
    dist_coeffs: 畸变参数，形状 (1,4) 或 (1,5)，如 [[k1, k2, p1, p2]]（OpenCV 格式）
    返回: (uv_depth Nx3 [u,v,depth], valid_mask, depths)，无效时 uv_depth/valid_mask/depths 为 None
    """
    # 只保留 Z>0 的点（在相机前方）
    valid_mask = points_cam[:, 2] > 0
    if not np.any(valid_mask):
        return None, None, None

    points_valid = points_cam[valid_mask].astype(np.float64)
    # cv2.projectPoints 要求 Nx1x3
    object_pts = np.expand_dims(points_valid, axis=1)
    # 点已在相机系下，物体系即相机系，故 rvec=0, tvec=0
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    image_pts, _ = cv2.projectPoints(object_pts, rvec, tvec, camera_matrix, dist_coeffs)
    # image_pts 形状 (N, 1, 2)
    uv = image_pts.reshape(-1, 2)
    depths = points_valid[:, 2]
    uv_depth = np.column_stack([uv[:, 0], uv[:, 1], depths])
    return uv_depth, valid_mask, depths


@dataclass
class DepthBatchContext:
    """批量深度生成：一次加载点云与位姿列表后的共享上下文（供多进程 fork 共享只读页）。"""
    points_world: np.ndarray
    poses: list
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    width: int
    height: int
    Rcl: Optional[np.ndarray]
    Pcl: Optional[np.ndarray]
    has_extrinsic: bool
    extrinsic_pcl_in_camera_frame: bool


def build_camera_intrinsics_from_calib_file(calib_file):
    """读取标定文件，返回 (camera_matrix, dist_coeffs, width, height)。"""
    cam_params = parse_camera_params(calib_file)
    fx = cam_params['cam_fx']
    fy = cam_params['cam_fy']
    cx = cam_params['cam_cx']
    cy = cam_params['cam_cy']
    width = int(cam_params['cam_width'])
    height = int(cam_params['cam_height'])
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    if 'cam_d0' in cam_params and 'cam_d1' in cam_params:
        dist_coeffs = np.array([[
            cam_params.get('cam_d0', 0),
            cam_params.get('cam_d1', 0),
            cam_params.get('cam_d2', 0),
            cam_params.get('cam_d3', 0)
        ]], dtype=np.float64)
    else:
        dist_coeffs = np.zeros((1, 4), dtype=np.float64)
    return camera_matrix, dist_coeffs, width, height


def prepare_depth_batch_context(
    pcd_file,
    odom_file,
    calib_file,
    extrinsic_calib_file=None,
    extrinsic_pcl_in_camera_frame=True,
):
    """
    一次性读取点云、位姿与内外参，供批量/多进程深度生成复用。
    """
    camera_matrix, dist_coeffs, width, height = build_camera_intrinsics_from_calib_file(
        calib_file
    )
    pcd = o3d.io.read_point_cloud(pcd_file)
    if len(pcd.points) == 0:
        raise ValueError(f"无法读取点云文件: {pcd_file}")
    points_world = np.asarray(pcd.points)
    poses = parse_odometry(odom_file)
    if len(poses) == 0:
        raise ValueError(f"无法从 {odom_file} 中解析到位姿")
    Rcl, Pcl = None, None
    has_extrinsic = bool(extrinsic_calib_file)
    if has_extrinsic:
        Rcl, Pcl = parse_extrinsic_calib(extrinsic_calib_file)
    return DepthBatchContext(
        points_world=points_world,
        poses=poses,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        width=width,
        height=height,
        Rcl=Rcl,
        Pcl=Pcl,
        has_extrinsic=has_extrinsic,
        extrinsic_pcl_in_camera_frame=extrinsic_pcl_in_camera_frame,
    )


def compute_depth_map_core(
    points_world,
    camera_position,
    camera_orientation,
    camera_matrix,
    dist_coeffs,
    width,
    height,
    output_visualization=True,
    quiet=False,
):
    """
    世界点云 + 相机位姿 -> 深度图（与原先 generate_depth_map 中投影与 Z-buffer 语义一致）。
    Z-buffer 使用 np.minimum.at 向量化，等价于原 for 循环取最近深度。
    """
    points_cam = world_to_camera(points_world, camera_position, camera_orientation)
    uv_depth, _valid_mask, depths = project_to_image(
        points_cam, camera_matrix, dist_coeffs
    )
    if uv_depth is None:
        if not quiet:
            print("  错误: 没有点在相机前方！")
        return None, None

    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    u_coords = np.round(uv_depth[:, 0]).astype(np.int64)
    v_coords = np.round(uv_depth[:, 1]).astype(np.int64)
    in_bounds = (
        (u_coords >= 0)
        & (u_coords < width)
        & (v_coords >= 0)
        & (v_coords < height)
    )
    u_coords = u_coords[in_bounds]
    v_coords = v_coords[in_bounds]
    depths = depths[in_bounds]
    depths_f = depths.astype(np.float32, copy=False)
    np.minimum.at(depth_map, (v_coords, u_coords), depths_f)

    if not output_visualization:
        if not quiet:
            print("\n" + "=" * 60)
            print("完成！")
            print("=" * 60)
        return depth_map, None

    depth_map_vis = depth_map.copy()
    depth_map_vis[~np.isfinite(depth_map_vis)] = 0

    if np.any(np.isfinite(depth_map)):
        depth_min = np.nanmin(depth_map[np.isfinite(depth_map)])
        depth_max = np.nanmax(depth_map[np.isfinite(depth_map)])
        depth_map_normalized = (depth_map_vis - depth_min) / (depth_max - depth_min + 1e-6)
        depth_map_normalized = (depth_map_normalized * 255).astype(np.uint8)
        depth_map_normalized = 255 - depth_map_normalized
    else:
        depth_map_normalized = np.zeros((height, width), dtype=np.uint8)

    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    invalid_mask = ~np.isfinite(depth_map)
    depth_map_colored[invalid_mask] = [0, 0, 0]

    if not quiet:
        print("\n" + "=" * 60)
        print("完成！")
        print("=" * 60)

    return depth_map, depth_map_colored


def depth_map_for_batch_pose(
    ctx: DepthBatchContext,
    pose_index: int,
    output_visualization=False,
    quiet=True,
):
    """在已加载的 DepthBatchContext 上生成单个位姿的深度图（不重复读盘）。"""
    poses = ctx.poses
    if len(poses) == 0:
        raise ValueError("位姿列表为空")
    pi = pose_index
    if pi >= len(poses):
        if not quiet:
            print(f"  警告: pose_index={pose_index} 超出范围，使用最后一个位姿")
        pi = len(poses) - 1
    lidar_pose = poses[pi]
    if ctx.has_extrinsic:
        camera_position, camera_orientation = lidar_pose_to_camera_pose(
            lidar_pose["position"],
            lidar_pose["orientation"],
            ctx.Rcl,
            ctx.Pcl,
            pcl_in_camera_frame=ctx.extrinsic_pcl_in_camera_frame,
        )
    else:
        if not quiet:
            print("\n[4/6] 警告: 未提供外参标定文件，假设位姿即为相机位姿")
        camera_position = lidar_pose["position"]
        camera_orientation = lidar_pose["orientation"]
    return compute_depth_map_core(
        ctx.points_world,
        camera_position,
        camera_orientation,
        ctx.camera_matrix,
        ctx.dist_coeffs,
        ctx.width,
        ctx.height,
        output_visualization=output_visualization,
        quiet=quiet,
    )


# 多进程 worker 通过 fork 继承父进程中的只读点云；由 prepare 之后在父进程赋值
_BATCH_CTX_FOR_WORKERS: Optional[DepthBatchContext] = None


def set_batch_context_for_workers(ctx: Optional[DepthBatchContext]):
    global _BATCH_CTX_FOR_WORKERS
    _BATCH_CTX_FOR_WORKERS = ctx


def batch_worker_pose(pose_index: int):
    """多进程池调用的顶层函数：使用 set_batch_context_for_workers 注入的上下文。"""
    global _BATCH_CTX_FOR_WORKERS
    ctx = _BATCH_CTX_FOR_WORKERS
    if ctx is None:
        return (pose_index, None, "batch context is not set")
    try:
        dm, _vis = depth_map_for_batch_pose(
            ctx, pose_index, output_visualization=False, quiet=True
        )
        return (pose_index, dm, None)
    except Exception as e:
        return (pose_index, None, str(e))


def generate_depth_map(pcd_file, odom_file, calib_file, pose_index=0, 
                       output_depth_file=None, output_visualization=True,
                       extrinsic_calib_file=None, extrinsic_pcl_in_camera_frame=True):
    """
    生成深度图
    
    参数:
    pcd_file: PCD点云文件路径
    odom_file: odometry.txt文件路径
    calib_file: 相机标定文件路径
    pose_index: 使用第几个位姿（默认0，即第一个）
    output_depth_file: 输出深度图文件路径（可选）
    output_visualization: 是否显示可视化结果
    extrinsic_calib_file: 外参标定文件路径（激光雷达到相机，可选）
    extrinsic_pcl_in_camera_frame: True=标定中 Pcl 在相机系（p_cam=Rcl@p_lidar+Pcl）；False=Pcl 在雷达系
    """
    camera_matrix, dist_coeffs, width, height = build_camera_intrinsics_from_calib_file(
        calib_file
    )

    pcd = o3d.io.read_point_cloud(pcd_file)
    if len(pcd.points) == 0:
        raise ValueError(f"无法读取点云文件: {pcd_file}")
    points_world = np.asarray(pcd.points)

    poses = parse_odometry(odom_file)
    if len(poses) == 0:
        raise ValueError(f"无法从 {odom_file} 中解析到位姿")

    pi = pose_index
    if pi >= len(poses):
        print(f"  警告: pose_index={pose_index} 超出范围，使用最后一个位姿")
        pi = len(poses) - 1

    lidar_pose = poses[pi]

    if extrinsic_calib_file:
        Rcl, Pcl = parse_extrinsic_calib(extrinsic_calib_file)
        camera_position, camera_orientation = lidar_pose_to_camera_pose(
            lidar_pose["position"],
            lidar_pose["orientation"],
            Rcl,
            Pcl,
            pcl_in_camera_frame=extrinsic_pcl_in_camera_frame,
        )
    else:
        print("\n[4/6] 警告: 未提供外参标定文件，假设位姿即为相机位姿")
        camera_position = lidar_pose["position"]
        camera_orientation = lidar_pose["orientation"]

    depth_map, depth_map_colored = compute_depth_map_core(
        points_world,
        camera_position,
        camera_orientation,
        camera_matrix,
        dist_coeffs,
        width,
        height,
        output_visualization=output_visualization,
        quiet=False,
    )
    if depth_map is None:
        return None
    return depth_map, depth_map_colored


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='从点云生成深度图')
    parser.add_argument('--pcd', type=str, default='GlobalMap.pcd',
                        help='点云文件路径')
    parser.add_argument('--odom', type=str, default='odometry.txt',
                        help='位姿文件路径')
    parser.add_argument('--calib', type=str, default='single_calib_result.txt',
                        help='相机标定文件路径')
    parser.add_argument('--extrinsic', type=str, default='multi_calib_result_modified.txt',
                        help='外参标定文件路径（激光雷达到相机，默认：multi_calib_result.txt）')
    parser.add_argument('--pose_idx', type=int, default=1,
                        help='使用第几个位姿（默认0）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出深度图文件路径（可选）')
    parser.add_argument('--no_vis', action='store_true',
                        help='不显示可视化结果')
    parser.add_argument('--extrinsic_pcl_in_lidar_frame', action='store_true',
                        help='若深度图仍偏移，可尝试：标定中 Pcl 为雷达系下平移时使用此选项')
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，自动生成
    if args.output is None:
        args.output = f'depth_map_pose{args.pose_idx}.jpg'
    try:
        depth_map, depth_vis = generate_depth_map(
            args.pcd,
            args.odom,
            args.calib,
            pose_index=args.pose_idx,
            output_depth_file=args.output,
            output_visualization=not args.no_vis,
            extrinsic_calib_file=args.extrinsic,
            extrinsic_pcl_in_camera_frame=not args.extrinsic_pcl_in_lidar_frame
        )
        cv2.imwrite(args.output, depth_vis)
        print(f"\n  彩色深度图已保存到: {args.output}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
