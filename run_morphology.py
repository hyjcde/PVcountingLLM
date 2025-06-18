#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import cv2
import numpy as np


def morphology_pv_detection(image_path):
    print('开始形态学光伏板检测...')
    
    # 加载图像
    rgb_image = cv2.imread(image_path)
    height, width = rgb_image.shape[:2]
    print(f'图像尺寸: {width} x {height}')
    
    # 创建调试目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_dir = f'morphology_debug_{timestamp}'
    os.makedirs(debug_dir, exist_ok=True)
    
    # 1. HSV色彩空间提取光伏板区域
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    
    # 多个蓝色范围
    lower_blue1 = np.array([100, 50, 50])
    upper_blue1 = np.array([130, 255, 200])
    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    
    lower_blue2 = np.array([80, 30, 30])
    upper_blue2 = np.array([120, 255, 150])
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])
    mask3 = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # 合并掩码
    combined_mask = cv2.bitwise_or(mask1, mask2)
    combined_mask = cv2.bitwise_or(combined_mask, mask3)
    
    # 2. 形态学闭运算修复轮廓
    print('应用形态学闭运算修复轮廓...')
    
    # 小核：连接细小断裂
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed_small = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, small_kernel)
    
    # 中等核：连接水平断裂
    medium_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed_medium = cv2.morphologyEx(closed_small, cv2.MORPH_CLOSE, medium_kernel)
    
    # 垂直核：连接垂直断裂
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    closed_vertical = cv2.morphologyEx(closed_medium, cv2.MORPH_CLOSE, vertical_kernel)
    
    # 最终矩形核
    final_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    final_closed = cv2.morphologyEx(closed_vertical, cv2.MORPH_CLOSE, final_kernel)
    
    # 3. 检测轮廓
    contours, _ = cv2.findContours(final_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'找到 {len(contours)} 个轮廓')
    
    # 4. 筛选光伏板
    panels = []
    panel_id = 1
    
    # 统计面积和长宽比
    areas = []
    aspect_ratios = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 0:
                aspect_ratio = w / float(h)
                areas.append(area)
                aspect_ratios.append(aspect_ratio)
    
    if areas:
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        mean_aspect = np.mean(aspect_ratios)
        std_aspect = np.std(aspect_ratios)
        
        print(f'面积统计: 均值={mean_area:.1f}, 标准差={std_area:.1f}')
        print(f'长宽比统计: 均值={mean_aspect:.2f}, 标准差={std_aspect:.2f}')
        
        # 动态阈值
        min_area = max(1000, mean_area - 2 * std_area)
        max_area = mean_area + 3 * std_area
        min_aspect = max(0.8, mean_aspect - 1.5 * std_aspect)
        max_aspect = min(6.0, mean_aspect + 1.5 * std_aspect)
        
        print(f'动态阈值 - 面积: [{min_area:.1f}, {max_area:.1f}]')
        print(f'动态阈值 - 长宽比: [{min_aspect:.2f}, {max_aspect:.2f}]')
    else:
        min_area, max_area = 1000, 20000
        min_aspect, max_aspect = 0.8, 4.0
    
    # 筛选轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w > 30 and h > 30:
                aspect_ratio = w / float(h) if h > 0 else 0
                
                if min_aspect <= aspect_ratio <= max_aspect:
                    # 计算矩形度
                    rect_area = w * h
                    rectangularity = area / rect_area if rect_area > 0 else 0
                    
                    if rectangularity > 0.6:
                        panels.append({
                            'id': f'PV{panel_id:03d}',
                            'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                            'area': float(area),
                            'aspect_ratio': float(aspect_ratio),
                            'rectangularity': float(rectangularity)
                        })
                        panel_id += 1
    
    # 5. 可视化
    result_img = rgb_image.copy()
    for i, panel in enumerate(panels):
        pos = panel['position']
        x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
        
        rectangularity = panel['rectangularity']
        if rectangularity > 0.8:
            color = (0, 255, 0)  # 绿色
        elif rectangularity > 0.7:
            color = (0, 255, 255)  # 黄色
        else:
            color = (255, 0, 0)  # 蓝色
        
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(result_img, str(i + 1), (x + w//2 - 15, y + h//2 + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(result_img, str(i + 1), (x + w//2 - 15, y + h//2 + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 保存结果
    cv2.imwrite(f'{debug_dir}/01_original.jpg', rgb_image)
    cv2.imwrite(f'{debug_dir}/02_combined_mask.jpg', combined_mask)
    cv2.imwrite(f'{debug_dir}/03_closed_small.jpg', closed_small)
    cv2.imwrite(f'{debug_dir}/04_closed_medium.jpg', closed_medium)
    cv2.imwrite(f'{debug_dir}/05_closed_vertical.jpg', closed_vertical)
    cv2.imwrite(f'{debug_dir}/06_final_closed.jpg', final_closed)
    cv2.imwrite(f'{debug_dir}/07_final_result.jpg', result_img)
    
    # 统计质量
    high_quality = [p for p in panels if p['rectangularity'] > 0.8]
    medium_quality = [p for p in panels if 0.7 < p['rectangularity'] <= 0.8]
    low_quality = [p for p in panels if p['rectangularity'] <= 0.7]
    
    print(f'检测到 {len(panels)} 个光伏板')
    print(f'高质量: {len(high_quality)} 个')
    print(f'中等质量: {len(medium_quality)} 个')
    print(f'低质量: {len(low_quality)} 个')
    print(f'结果保存在: {debug_dir}')
    
    return panels, debug_dir

if __name__ == "__main__":
    # 运行检测
    panels, debug_dir = morphology_pv_detection('DJI_0518_W.JPG') 