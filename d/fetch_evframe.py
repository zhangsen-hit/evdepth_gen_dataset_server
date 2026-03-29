import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


from PIL import Image

def save_overlay(overlay, output, title=None, legend=None):
    # overlay为float，0~1，转uint8
    arr = (overlay * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(output)

def main():
    parser = argparse.ArgumentParser(description='按频率累加NPZ文件像素值')
    parser.add_argument('--dir', default='./dataset', help='数据目录')
    parser.add_argument('--start', type=int, default=1995, help='起始编号')
    parser.add_argument('--end', type=int, default=2005, help='结束编号')
    parser.add_argument('--output', default='event_tensor_noted.png', help='输出文件（带注解）')
    args = parser.parse_args()
    
    print(f"按频率累加 {args.start:05d}.npz 到 {args.end:05d}.npz")
    ch0_sum = None
    ch1_sum = None
    count = 0
    for i in range(args.start, args.end + 1):
        file_path = os.path.join(args.dir, f"{i:05d}.npz")
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                if 'input' in data:
                    input_data = data['input']
                    print(f"{file_path} 图像尺寸: {input_data.shape}")  # 新增打印尺寸
                    if ch0_sum is None:
                        ch0_sum = np.zeros_like(input_data[0])
                        ch1_sum = np.zeros_like(input_data[1])
                    ch0_sum += input_data[0]
                    ch1_sum += input_data[1]
                    count += 1
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
    if count == 0:
        print("没有找到文件")
        return
    ch0_display = np.log1p(np.abs(ch0_sum))
    ch1_display = np.log1p(np.abs(ch1_sum))
    ch0_norm = (ch0_display - ch0_display.min()) / (ch0_display.max() - ch0_display.min() + 1e-8)
    ch1_norm = (ch1_display - ch1_display.min()) / (ch1_display.max() - ch1_display.min() + 1e-8)
    overlay = np.zeros(ch0_sum.shape + (3,))
    overlay[:, :, 2] = ch0_norm
    overlay[:, :, 0] = ch1_norm

    save_overlay(overlay, 'event_tensor.png')

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='red', label=f'Ch0: {ch0_sum.sum():.0f}'),
        Patch(facecolor='blue', label=f'Ch1: {ch1_sum.sum():.0f}'),
        Patch(facecolor='purple', label='both')
    ]
    title = f'stack: {args.start:05d}-{args.end:05d} ({count}files)'
    save_overlay(overlay, args.output, title=title, legend=legend)

    print(f"无注解图已保存: event_tensor.png")
    print(f"带注解图已保存: {args.output}")

    height, width = overlay.shape[:2]
    fig_width = 12
    fig_height = fig_width * (height / width)
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(overlay, aspect='equal')
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.legend(handles=legend, loc='upper right', fontsize=11)
    # plt.show()

if __name__ == "__main__":
    main()
