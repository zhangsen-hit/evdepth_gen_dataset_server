import numpy as np
from typing import Dict, List, Tuple
import time
import warnings
import os
import sys
import yaml

warnings.filterwarnings('ignore')


def load_odom_timestamps(txt_path):
    timestamps = []

    with open(txt_path, 'r') as f:
        docs = yaml.safe_load_all(f)  # 自动按 --- 分块

        for doc in docs:
            if doc is None:
                continue
            secs = doc['header']['stamp']['secs']
            nsecs = doc['header']['stamp']['nsecs']
            t = secs + nsecs * 1e-9
            timestamps.append(t)

    return np.array(timestamps)




class EventOdomProcessor:
    def __init__(self, events_path: str, odom_path: str):
        """
        初始化事件相机和odom数据处理器
        
        参数:
            events_path: 事件数据npz文件路径
            odom_path: odom数据npz文件路径
        """
        print("加载数据...")
        
        # 使用mmap模式加载以减少内存使用
        self.events_data = np.load(events_path, allow_pickle=True, mmap_mode='r')
        
        
        # 获取事件数据
        self.events = self.events_data['events']
        print(f"事件数据形状: {self.events.shape}")
        print(f"事件总数: {self.events.shape[0]}")
        

        
        # 获取odom时间戳
        self.odom_timestamps = load_odom_timestamps(odom_path)
        print(f"odom时间戳数量: {len(self.odom_timestamps)}")
        # 传感器分辨率

        self.height = 260   # 高度
        self.width = 346    # 宽度
        self.output_height = 260  # 输出高度保持不变
        self.output_width = 346   # 输出宽度保持不变
        
        # 时间窗口设置 (5ms = 0.005秒)
        self.window_half = 0.0025  # 2.5ms
        self.window_size = 0.005   # 5ms
        
        # 验证时间戳范围
        self._validate_timestamps()
        
    def _validate_timestamps(self):
        """验证事件和odom时间戳的范围"""
        event_min_time = self.events[:, 0].min()
        event_max_time = self.events[:, 0].max()
        odom_min_time = self.odom_timestamps.min()
        odom_max_time = self.odom_timestamps.max()
        
        print(f"事件时间范围: [{event_min_time:.6f}, {event_max_time:.6f}]")
        print(f"odom时间范围: [{odom_min_time:.6f}, {odom_max_time:.6f}]")
        print(f"时间范围重叠: {max(event_min_time, odom_min_time):.6f} to {min(event_max_time, odom_max_time):.6f}")
        
    def process_all_frames(self, output_path: str = "event_frames.npz"):
        """
        处理所有帧并保存结果（内存高效版本）
        事件帧形状将调整为 (2, 260, 346) 即 (channels, height, width)
        
        参数:
            output_path: 输出npz文件路径
        """
        print(f"\n开始处理 {len(self.odom_timestamps)} 个odom样本...")
        print(f"输出事件帧形状: (batch_size, 2, {self.output_height}, {self.output_width})")
        
        # 预计算事件时间戳以提高速度
        event_timestamps = self.events[:, 0]
        
        # 查找事件时间的开始和结束索引
        start_idx = 0
        
        # 先计数有效帧，然后预分配内存
        print("计算有效帧数量...")
        valid_count = 0
        for i, odom_time in enumerate(self.odom_timestamps):
            if i % 2000 == 0 and i > 0:
                print(f"  计数进度: {i}/{len(self.odom_timestamps)}")
            
            # 定义时间窗口
            time_start = odom_time - self.window_half
            time_end = odom_time + self.window_half
            
            # 快速查找事件索引范围
            while start_idx < len(event_timestamps) and event_timestamps[start_idx] < time_start:
                start_idx += 1
            
            window_start_idx = start_idx
            end_idx = window_start_idx
            while end_idx < len(event_timestamps) and event_timestamps[end_idx] < time_end:
                end_idx += 1
            
            if window_start_idx < end_idx:
                valid_count += 1
        
        print(f"找到 {valid_count} 个有效时间窗口")
        
        if valid_count == 0:
            print("警告：没有找到有效的时间窗口！")
            return None, None
        
        # 预分配内存
        print(f"预分配内存: {valid_count} 帧")
        all_frames = np.zeros((valid_count, 2, self.output_height, self.output_width), dtype=np.float32)
        valid_timestamps = np.zeros(valid_count, dtype=np.float64)
        
        # 重新处理
        print("\n生成事件帧...")
        start_idx = 0
        frame_idx = 0
        
        for i, odom_time in enumerate(self.odom_timestamps):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{len(self.odom_timestamps)}")
                if frame_idx > 0:
                    print(f"  已生成 {frame_idx} 帧")
            
            # 定义时间窗口
            time_start = odom_time - self.window_half
            time_end = odom_time + self.window_half
            
            # 快速查找事件索引范围
            while start_idx < len(event_timestamps) and event_timestamps[start_idx] < time_start:
                start_idx += 1
            
            window_start_idx = start_idx
            end_idx = window_start_idx
            while end_idx < len(event_timestamps) and event_timestamps[end_idx] < time_end:
                end_idx += 1
            
            if window_start_idx >= end_idx:
                continue
            
            # 获取窗口内的事件
            window_events = self.events[window_start_idx:end_idx]
            
            # 检查是否在时间窗口内
            valid_mask = (window_events[:, 0] >= time_start) & (window_events[:, 0] <= time_end)
            window_events = window_events[valid_mask]
            
            if len(window_events) == 0:
                continue
            
            # 创建事件帧
            event_frame = self._create_event_frame(window_events)
            
            if event_frame is not None:
                all_frames[frame_idx] = event_frame
                valid_timestamps[frame_idx] = odom_time
                frame_idx += 1
        
        print(f"\n处理完成！有效帧数: {frame_idx}")
        print(f"有效帧比例: {frame_idx/len(self.odom_timestamps)*100:.2f}%")
        
        # 如果有提前结束的情况，截断数组
        if frame_idx < valid_count:
            print(f"实际生成 {frame_idx} 帧，截断数组")
            all_frames = all_frames[:frame_idx]
            valid_timestamps = valid_timestamps[:frame_idx]
        
        # 保存结果
        self._save_results(output_path, all_frames, valid_timestamps)
        
        # 清理内存
        del window_events
        del event_frame
        
        return all_frames, valid_timestamps


    def _create_event_frame(self, window_events: np.ndarray) -> np.ndarray:
        """
        正确的处理方法 (260×346)
        """
        if len(window_events) == 0:
            return None
        
        height, width = 260, 346
        pos_frame = np.zeros((height, width), dtype=np.float32)
        neg_frame = np.zeros((height, width), dtype=np.float32)
        
        # 提取坐标
        x_coords = window_events[:, 1].astype(int)
        y_coords = window_events[:, 2].astype(int)
        polarities = window_events[:, 3]
        
        # 边界检查
        mask = (x_coords >= 0) & (x_coords < width) & \
            (y_coords >= 0) & (y_coords < height)
        
        if not mask.any():
            return None
        
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        polarities = polarities[mask]
        
        # 分离正负事件
        pos_mask = polarities == 1  # 根据你的数据，1是正事件
        neg_mask = polarities == 0  # 0是负事件
        
        # 处理正事件
        if pos_mask.any():
            pos_x = x_coords[pos_mask]
            pos_y = y_coords[pos_mask]
            # 直接使用y作为行，x作为列
            for x, y in zip(pos_x, pos_y):
                pos_frame[y, x] += 1
        
        # 处理负事件
        if neg_mask.any():
            neg_x = x_coords[neg_mask]
            neg_y = y_coords[neg_mask]
            for x, y in zip(neg_x, neg_y):
                neg_frame[y, x] += 1
        
        return np.stack([pos_frame, neg_frame], axis=0)



    
    def _save_results(self, output_path: str, frames: np.ndarray, timestamps: np.ndarray):
        """
        保存处理结果到npz文件
        
        参数:
            output_path: 输出文件路径
            frames: 事件帧数组，形状为 (N, 2, 260, 346)
            timestamps: 对应的时间戳
        """
        print(f"\n保存结果到 {output_path}...")
        
        metadata = {
            "original_event_file": "events_data.npz",
            "original_odom_file": "odometry.txt",
            "window_size_seconds": float(self.window_size),
            "window_half_seconds": float(self.window_half),
            "frame_shape": list(frames.shape),
            "resolution": f"{self.output_height}x{self.output_width}",
            "total_frames": int(len(frames)),
            "total_odom_samples": int(len(self.odom_timestamps)),
            "valid_ratio": float(len(frames)/len(self.odom_timestamps)*100),
            "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "2-channel event frames (positive, negative) centered at odom timestamps"
        }
        
        # 分批保存以节省内存
        try:
            np.savez_compressed(
                output_path,
                event_frames=frames.astype(np.float32),
                timestamps=timestamps.astype(np.float64),
                metadata=np.array([metadata], dtype=object)
            )
            
            print(f"✓ 保存完成！")
            print(f"  文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            print(f"  事件帧形状: {frames.shape}")
            print(f"  帧数: {len(frames)}")
            print(f"  时间戳数量: {len(timestamps)}")
            
        except MemoryError as e:
            print(f"内存不足，尝试分批保存...")
            # 分批保存
            self._save_in_batches(output_path, frames, timestamps, metadata)
    
    def _save_in_batches(self, output_path: str, frames: np.ndarray, timestamps: np.ndarray, metadata: dict):
        """分批保存数据以节省内存"""
        batch_size = 1000  # 每批保存1000帧
        total_frames = len(frames)
        batches = (total_frames + batch_size - 1) // batch_size
        
        print(f"  分批保存: {total_frames} 帧分为 {batches} 批")
        
        # 创建临时目录
        temp_dir = "temp_save"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # 分批保存到临时文件
            batch_files = []
            for i in range(batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_frames)
                
                batch_frames = frames[start_idx:end_idx]
                batch_timestamps = timestamps[start_idx:end_idx]
                
                batch_file = os.path.join(temp_dir, f"batch_{i:04d}.npz")
                np.savez_compressed(
                    batch_file,
                    event_frames=batch_frames.astype(np.float32),
                    timestamps=batch_timestamps.astype(np.float64)
                )
                batch_files.append(batch_file)
                print(f"    保存批次 {i+1}/{batches}: {batch_file}")
            
            # 合并所有批次
            print(f"  合并批次文件...")
            all_frames = []
            all_timestamps = []
            
            for batch_file in batch_files:
                batch_data = np.load(batch_file, allow_pickle=True)
                all_frames.append(batch_data['event_frames'])
                all_timestamps.append(batch_data['timestamps'])
                batch_data.close()
            
            # 合并数据
            combined_frames = np.concatenate(all_frames, axis=0)
            combined_timestamps = np.concatenate(all_timestamps, axis=0)
            
            # 保存最终文件
            np.savez_compressed(
                output_path,
                event_frames=combined_frames.astype(np.float32),
                timestamps=combined_timestamps.astype(np.float64),
                metadata=np.array([metadata], dtype=object)
            )
            
            print(f"✓ 合并完成！最终文件: {output_path}")
            print(f"  文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            
        finally:
            # 清理临时文件
            for batch_file in batch_files:
                if os.path.exists(batch_file):
                    os.remove(batch_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)


# 使用示例
if __name__ == "__main__":
    print("Jetson事件帧生成器 (内存优化版本)")
    print("=" * 60)
    
    # 文件路径
    events_file = "events_data.npz"
    odom_file = "odometry.txt"
    output_file = "events_tensor.npz"
    
    try:
        # 检查输入文件是否存在
        print("检查输入文件...")
        if not os.path.exists(events_file):
            print(f"错误: 事件文件 '{events_file}' 不存在！")
            sys.exit(1)
        if not os.path.exists(odom_file):
            print(f"错误: odom文件 '{odom_file}' 不存在！")
            sys.exit(1)
            
        print(f"✓ 找到事件文件: {events_file}")
        print(f"✓ 找到odom文件: {odom_file}")
        
        # 创建处理器
        print("\n初始化处理器...")
        processor = EventOdomProcessor(events_file, odom_file)
        
        # 处理所有帧
        print(f"\n开始处理...")
        print(f"输出文件: {output_file}")
        frames, timestamps = processor.process_all_frames(output_file)
        
        if frames is not None:
            # 检查是否生成了输出文件
            if os.path.exists(output_file):
                print(f"\n" + "=" * 60)
                print("处理结果总结:")
                print("=" * 60)
                print(f"✓ 成功生成输出文件: {output_file}")
                print(f"  文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
                print(f"  事件帧形状: {frames.shape}")
                print(f"  帧数: {len(frames)}")
                print(f"  时间戳数量: {len(timestamps)}")
                print(f"  每帧平均事件数: {frames.sum(axis=(1,2,3)).mean():.1f}")
                
                # 快速验证
                print(f"\n快速验证...")
                result = np.load(output_file, allow_pickle=True)
                print(f"  文件包含的数据: {list(result.files)}")
                if 'event_frames' in result:
                    loaded_frames = result['event_frames']
                    print(f"  加载的事件帧形状: {loaded_frames.shape}")
                    print(f"  数据一致性检查: {np.allclose(frames, loaded_frames)}")
                result.close()
            else:
                print(f"\n✗ 未生成输出文件: {output_file}")
        else:
            print("\n✗ 未能生成事件帧")
                
    except KeyboardInterrupt:
        print("\n\n用户中断处理")
    except MemoryError as e:
        print(f"\n内存不足错误: {e}")
        print("建议:")
        print("1. 增加Jetson的交换空间")
        print("2. 减少时间窗口大小")
        print("3. 分批处理数据")
    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

