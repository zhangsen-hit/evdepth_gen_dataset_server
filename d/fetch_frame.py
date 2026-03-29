import cv2
import os

def extract_frame_by_time(video_path, target_time, output_path):
    """
    根据时间（秒）抽取视频中最接近的一帧并保存为jpg
    
    :param video_path: 视频路径
    :param target_time: 目标时间（单位：秒，例如4.23）
    :param output_path: 输出图片路径，例如 output.jpg
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("无法打开视频文件")

    # 获取视频FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("无法获取FPS，视频可能损坏")

    # 计算目标帧号
    frame_index = int(round(target_time * fps))

    # 设置视频读取位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    if not ret:
        raise ValueError("无法读取目标帧，可能时间超出视频长度")

    # 保存为jpg
    cv2.imwrite(output_path, frame)
    cap.release()

    print(f"成功提取第 {target_time} 秒附近的帧")
    print(f"FPS: {fps}")
    print(f"对应帧号: {frame_index}")
    print(f"保存路径: {output_path}")


if __name__ == "__main__":
    video_path = "output_video.mp4"
    target_time = 10.0
    output_path = "frame.jpg"

    extract_frame_by_time(video_path, target_time, output_path)
