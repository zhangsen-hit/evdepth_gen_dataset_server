import cv2
import numpy as np

def overlay_with_alpha_channel(background_path, overlay_path, output_path='overlay_result.png', 
                               alpha_bg=0.5, alpha_fg=0.8, remove_black_bg=True, black_threshold=10):
    """
    将两张图片叠加，可选择去除前景图的黑色背景
    
    参数:
        background_path: 背景图片路径
        overlay_path: 前景图片路径
        output_path: 输出图片路径
        alpha_bg: 背景透明度
        alpha_fg: 前景透明度
        remove_black_bg: 是否去除前景图的黑色背景
        black_threshold: 黑色阈值，小于该值的像素被视为黑色背景
    """
    frame = cv2.imread('frame.jpg')
    
    
    background = cv2.imread(background_path)
    if background is None:
        print(f"错误：无法读取背景图片 {background_path}")
        return False

    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        print(f"错误：无法读取叠加图片 {overlay_path}")
        return False

    # 调整前景图尺寸与背景图匹配
    if background.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))
        print(f"已调整前景图尺寸至: {overlay.shape}")

    # 背景黄色mask（容忍一定误差）
    yellow_mask = np.all(np.abs(background - np.array([255,255,0])) < 20, axis=2)
    bg_alpha = np.ones(background.shape[:2], dtype=np.float32) * alpha_bg
    bg_alpha[yellow_mask] = 0.0

    # 处理前景图
    if overlay.shape[2] == 4:
        overlay_rgb = overlay[:, :, :3].astype(np.float32)
        overlay_alpha = overlay[:, :, 3] / 255.0
    else:
        overlay_rgb = overlay.astype(np.float32)
        overlay_alpha = np.ones((overlay.shape[0], overlay.shape[1]), dtype=np.float32)
    
    # 去除黑色背景 - 改进部分
    if remove_black_bg:
        # 计算每个像素到黑色的距离
        black_distance = np.sqrt(np.sum(overlay_rgb ** 2, axis=2))
        
        # 创建黑色背景掩码
        black_mask = black_distance <= black_threshold
        
        # 将黑色背景区域的alpha设为0
        overlay_alpha[black_mask] = 0.0
        
        # 可选：将黑色像素转换为非常暗的灰色，以保留一些结构信息
        # overlay_rgb[black_mask] = [1, 1, 1]  # 极暗灰色
        
        # 统计信息
        non_black_pixels = np.sum(~black_mask)
        total_pixels = black_mask.size
        print(f"黑色背景像素: {np.sum(black_mask)} ({np.sum(black_mask)/total_pixels*100:.1f}%)")
        print(f"非黑色前景像素: {non_black_pixels} ({non_black_pixels/total_pixels*100:.1f}%)")
    
    # 应用前景透明度
    overlay_alpha = overlay_alpha * alpha_fg

    # 扩展通道
    overlay_alpha_3 = np.repeat(overlay_alpha[:, :, np.newaxis], 3, axis=2)
    bg_alpha_3 = np.repeat(bg_alpha[:, :, np.newaxis], 3, axis=2)

    # 混合计算 - 使用正常的alpha混合公式
    # 结果 = 前景 * 前景透明度 + 背景 * (1 - 前景透明度) * 背景透明度
    result = overlay_rgb * overlay_alpha_3 + background.astype(np.float32) * bg_alpha_3 * (1 - overlay_alpha_3)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"重叠结果已保存到: {output_path}")

    # 显示结果
    cv2.imshow('Overlay Result - No Black Background', result)
    
    # 可选：也显示原始前景图以便对比
    overlay_display = overlay_rgb.astype(np.uint8) if overlay_rgb.max() <= 255 else overlay_rgb.astype(np.uint8)
    cv2.imshow('Original Overlay', overlay_display)
    
    cv2.imshow('Background', background)
    cv2.imshow('Frame', frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


def overlay_with_alpha_channel_advanced(background_path, overlay_path, output_path='overlay_result_advanced.png',
                                       alpha_bg=0.3, alpha_fg=0.8, black_threshold=10, 
                                       preserve_edges=True, edge_enhance=False):
    """
    高级版本：提供更多选项来处理黑色背景
    
    参数:
        preserve_edges: 是否保留边缘结构（将黑色转换为极暗灰色）
        edge_enhance: 是否增强边缘对比度
    """
    background = cv2.imread(background_path)
    if background is None:
        print(f"错误：无法读取背景图片 {background_path}")
        return False

    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        print(f"错误：无法读取叠加图片 {overlay_path}")
        return False

    # 调整前景图尺寸
    if background.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))

    # 背景黄色mask
    yellow_mask = np.all(np.abs(background - np.array([255,255,0])) < 20, axis=2)
    bg_alpha = np.ones(background.shape[:2], dtype=np.float32) * alpha_bg
    bg_alpha[yellow_mask] = 0.0

    # 处理前景图
    if overlay.shape[2] == 4:
        overlay_rgb = overlay[:, :, :3].astype(np.float32)
        orig_overlay_alpha = overlay[:, :, 3] / 255.0
    else:
        overlay_rgb = overlay.astype(np.float32)
        orig_overlay_alpha = np.ones((overlay.shape[0], overlay.shape[1]), dtype=np.float32)
    
    # 计算黑色距离
    black_distance = np.sqrt(np.sum(overlay_rgb ** 2, axis=2))
    black_mask = black_distance <= black_threshold
    
    # 创建渐变alpha（边缘附近保持一定透明度）
    if preserve_edges:
        # 对黑色区域进行膨胀操作，找到边缘区域
        kernel = np.ones((3,3), np.uint8)
        black_mask_dilated = cv2.dilate(black_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        edge_mask = black_mask_dilated & ~black_mask
        
        # 边缘区域保持较低的透明度，以保留结构信息
        overlay_alpha = orig_overlay_alpha.copy()
        overlay_alpha[black_mask] = 0.0
        overlay_alpha[edge_mask] = overlay_alpha[edge_mask] * 0.3  # 边缘半透明
    else:
        overlay_alpha = orig_overlay_alpha.copy()
        overlay_alpha[black_mask] = 0.0
    
    # 应用前景透明度
    overlay_alpha = overlay_alpha * alpha_fg
    
    # 边缘增强（可选）
    if edge_enhance:
        # 使用Sobel算子检测边缘
        gray = cv2.cvtColor(overlay_rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_magnitude = (edge_magnitude / edge_magnitude.max() * 0.5)  # 归一化
        
        # 增强边缘区域的alpha值
        overlay_alpha = overlay_alpha * (1 + edge_magnitude)
        overlay_alpha = np.clip(overlay_alpha, 0, 1)

    # 扩展通道
    overlay_alpha_3 = np.repeat(overlay_alpha[:, :, np.newaxis], 3, axis=2)
    bg_alpha_3 = np.repeat(bg_alpha[:, :, np.newaxis], 3, axis=2)

    # 混合
    result = overlay_rgb * overlay_alpha_3 + background.astype(np.float32) * bg_alpha_3
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"高级版本重叠结果已保存到: {output_path}")

    # 显示结果
    cv2.imshow('Advanced Overlay Result', result)
    
    # 显示alpha通道以便调试
    alpha_display = (overlay_alpha * 255).astype(np.uint8)
    cv2.imshow('Overlay Alpha Mask', alpha_display)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    # 使用基本版本
    overlay_with_alpha_channel(
        background_path="frame_depth.png",
        overlay_path="event_tensor.png",
        output_path="overlay_result.png",
        alpha_bg=1.0,      # 背景透明度
        alpha_fg=1.0,       # 前景透明度
        remove_black_bg=True,
        black_threshold=10  # 黑色阈值，可根据需要调整
    )
