import rosbag
import numpy as np
from tqdm import tqdm

def extract_event_camera_data(bag_file, event_topic, output_npz):
    """
    完整提取事件相机数据并保存为npz文件
    
    Args:
        bag_file: rosbag文件路径
        event_topic: 事件话题名称（如 /dvs/events）
        output_npz: 输出npz文件路径
    """
    
    print(f"处理文件: {bag_file}")
    print(f"事件话题: {event_topic}")
    
    # 用于存储所有事件
    all_events = []
    
    try:
        bag = rosbag.Bag(bag_file, 'r')
        
        # 获取消息数量
        msg_count = bag.get_message_count(topic_filters=[event_topic])
        print(f"事件消息数量: {msg_count}")
        
        if msg_count == 0:
            print("未找到事件数据")
            return
        
        # 处理每条消息
        total_events = 0
        event_batches = []
        
        for topic, msg, t in tqdm(bag.read_messages(topics=[event_topic]), 
                                 total=msg_count, desc="提取事件"):
            
            # 检查消息类型
            msg_type = msg._type
            
            # 处理不同的事件相机消息格式
            if msg_type == 'dvs_msgs/EventArray':
                # DVS格式
                batch_size = len(msg.events)
                if batch_size > 0:
                    batch_data = np.zeros((batch_size, 4), dtype=np.float64)
                    
                    for i, event in enumerate(msg.events):
                        # print("ts",(event.ts))
                        # print("ts.to_sec",(event.ts.to_sec()))
                        batch_data[i, 0] = event.ts.to_sec()  # 时间戳
                        batch_data[i, 1] = event.x            # x坐标
                        batch_data[i, 2] = event.y            # y坐标
                        batch_data[i, 3] = event.polarity     # 极性
                        # print("batch_data",batch_data[i,0])
                    
                    event_batches.append(batch_data)
                    total_events += batch_size
                    
            # elif msg_type == 'prophesee_event_msgs/EventArray':
            #     # Prophesee格式
            #     batch_size = len(msg.events)
            #     if batch_size > 0:
            #         batch_data = np.zeros((batch_size, 4), dtype=np.float32)
                    
            #         for i, event in enumerate(msg.events):
            #             batch_data[i, 0] = event.t.to_sec()   # 时间戳
            #             batch_data[i, 1] = event.x            # x坐标
            #             batch_data[i, 2] = event.y            # y坐标
            #             batch_data[i, 3] = event.p            # 极性
                    
            #         event_batches.append(batch_data)
            #         total_events += batch_size
                    
            # elif hasattr(msg, 'events'):
            #     # 通用事件数组格式
            #     batch_size = len(msg.events)
            #     if batch_size > 0:
            #         batch_data = np.zeros((batch_size, 4), dtype=np.float32)
                    
            #         for i, event in enumerate(msg.events):
            #             # 尝试获取时间戳
            #             if hasattr(event, 'ts'):
            #                 ts = event.ts.to_sec()
            #             elif hasattr(event, 't'):
            #                 ts = event.t.to_sec()
            #             elif hasattr(event, 'timestamp'):
            #                 ts = event.timestamp.to_sec()
            #             else:
            #                 ts = t.to_sec()
                        
            #             batch_data[i, 0] = ts
            #             batch_data[i, 1] = getattr(event, 'x', 0)
            #             batch_data[i, 2] = getattr(event, 'y', 0)
            #             batch_data[i, 3] = getattr(event, 'polarity', 
            #                                       getattr(event, 'p', 0))
                    
            #         event_batches.append(batch_data)
            #         total_events += batch_size
        
        bag.close()
        
        print(f"总共提取事件数量: {total_events}")
        
        if total_events == 0:
            print("没有提取到任何事件")
            return
        
        # 合并所有批次数据
        print("合并之前形状 ",len(event_batches))
        print("合并数据...")
        all_events = np.vstack(event_batches)
        print("合并后形状 ",len(all_events))
        
        # # 按时间戳排序
        # print("按时间戳排序...")
        # sort_idx = np.argsort(all_events[:, 0])
        # all_events = all_events[sort_idx]
        
        # 保存为npz文件
        print(f"保存数据到 {output_npz}...")
        np.savez_compressed(
            output_npz,
            events=all_events,
            columns=['timestamp', 'x', 'y', 'polarity'],
            total_events=total_events,
            time_range=[all_events[0, 0], all_events[-1, 0]],
            topic_name=event_topic,
            original_bag=bag_file
        )
        
        print(f"完成！事件数据已保存")
        print(f"时间范围: {all_events[0, 0]:.3f} 到 {all_events[-1, 0]:.3f}")
        print(f"事件统计:")
        print(f"  正事件: {np.sum(all_events[:, 3] > 0)}")
        print(f"  负事件: {np.sum(all_events[:, 3] <= 0)}")
        
        return all_events
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

# 使用示例
if __name__ == "__main__":
    extract_event_camera_data(
        bag_file="liosam.bag",
        event_topic="dvs/events",
        output_npz="events_data.npz"
    )
