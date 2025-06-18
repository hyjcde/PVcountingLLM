#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime

import cv2
import numpy as np


def improved_detection(image_path):
    print('开始改进的全面光伏板检测...')
    
    # 加载图像
    rgb_image = cv2.imread(image_path)
    height, width = rgb_image.shape[:2]
    print(f'图像尺寸: {width} x {height}')
    
    # 创建调试目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_dir = f'improved_debug_{timestamp}'
    os.makedirs(debug_dir, exist_ok=True)
    
    # 转换色彩空间
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    # 更宽松的蓝色检测
    lower_blue = np.array([80, 20, 20])
    upper_blue = np.array([150, 255, 250])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 深色区域检测
    _, dark_mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    
    # 合并掩码
    combined_mask = cv2.bitwise_or(blue_mask, dark_mask)
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # 边缘检测
    edges = cv2.Canny(gray, 30, 100)
    
    # 霍夫线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                           minLineLength=100, maxLineGap=50)
    
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length > 80:
                if abs(angle) < 20 or abs(angle) > 160:
                    horizontal_lines.append((y1 + y2) // 2)
                elif abs(abs(angle) - 90) < 20:
                    vertical_lines.append((x1 + x2) // 2)
    
    # 聚类线条
    def cluster_lines(lines, tolerance=30):
        if not lines:
            return []
        lines_sorted = sorted(set(lines))
        clustered = []
        current_cluster = [lines_sorted[0]]
        
        for i in range(1, len(lines_sorted)):
            if lines_sorted[i] - current_cluster[-1] <= tolerance:
                current_cluster.append(lines_sorted[i])
            else:
                clustered.append(sum(current_cluster) // len(current_cluster))
                current_cluster = [lines_sorted[i]]
        
        if current_cluster:
            clustered.append(sum(current_cluster) // len(current_cluster))
        return clustered
    
    horizontal_lines = cluster_lines(horizontal_lines)
    vertical_lines = cluster_lines(vertical_lines)
    
    # 如果线条不足，生成均匀网格
    if len(horizontal_lines) < 15:
        horizontal_lines = list(range(0, height, height//25))
    if len(vertical_lines) < 40:
        vertical_lines = list(range(0, width, width//60))
    
    print(f'检测到 {len(horizontal_lines)} 条水平线, {len(vertical_lines)} 条垂直线')
    
    # 分割面板
    panels = []
    panel_id = 1
    
    for r in range(len(horizontal_lines) - 1):
        for c in range(len(vertical_lines) - 1):
            y1, y2 = horizontal_lines[r], horizontal_lines[r + 1]
            x1, x2 = vertical_lines[c], vertical_lines[c + 1]
            
            w, h = x2 - x1, y2 - y1
            
            if 20 < w < 500 and 20 < h < 500:
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    roi = combined_mask[y1:y2, x1:x2]
                    if roi.size > 0:
                        pixel_ratio = np.sum(roi > 0) / roi.size
                        if pixel_ratio > 0.05:  # 只需5%的像素
                            panels.append({
                                'id': f'PV{panel_id:03d}',
                                'grid_position': f'R{r+1}-C{c+1}',
                                'position': {
                                    'x': int(x1), 
                                    'y': int(y1), 
                                    'width': int(w), 
                                    'height': int(h)
                                },
                                'pixel_ratio': float(pixel_ratio),
                                'aspect_ratio': float(aspect_ratio)
                            })
                            panel_id += 1
    
    # 可视化
    result_img = rgb_image.copy()
    for i, panel in enumerate(panels):
        pos = panel['position']
        x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
        
        # 根据像素比例选择颜色
        ratio = panel['pixel_ratio']
        if ratio > 0.3:
            color = (0, 255, 0)  # 绿色
        elif ratio > 0.15:
            color = (0, 255, 255)  # 黄色
        else:
            color = (255, 0, 0)  # 蓝色
        
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        
        # 显示编号
        cv2.putText(result_img, str(i + 1), 
                   (x + w//2 - 10, y + h//2 + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result_img, str(i + 1), 
                   (x + w//2 - 10, y + h//2 + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 绘制网格线
    grid_img = rgb_image.copy()
    for y in horizontal_lines:
        cv2.line(grid_img, (0, y), (width, y), (0, 255, 0), 2)
    for x in vertical_lines:
        cv2.line(grid_img, (x, 0), (x, height), (255, 0, 0), 2)
    
    # 保存结果
    cv2.imwrite(f'{debug_dir}/01_original.jpg', rgb_image)
    cv2.imwrite(f'{debug_dir}/02_combined_mask.jpg', combined_mask)
    cv2.imwrite(f'{debug_dir}/03_grid_lines.jpg', grid_img)
    cv2.imwrite(f'{debug_dir}/04_final_result.jpg', result_img)
    
    # 统计信息
    high_ratio = [p for p in panels if p['pixel_ratio'] > 0.3]
    medium_ratio = [p for p in panels if 0.15 < p['pixel_ratio'] <= 0.3]
    low_ratio = [p for p in panels if p['pixel_ratio'] <= 0.15]
    
    result_data = {
        'method': 'improved_comprehensive_detection',
        'total_panels': len(panels),
        'detection_time': datetime.now().isoformat(),
        'image_size': {'width': width, 'height': height},
        'statistics': {
            'high_ratio_panels': len(high_ratio),
            'medium_ratio_panels': len(medium_ratio),
            'low_ratio_panels': len(low_ratio)
        },
        'panels': panels
    }
    
    with open(f'{debug_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f'检测到 {len(panels)} 个光伏板')
    print(f'高像素比例: {len(high_ratio)} 个')
    print(f'中等像素比例: {len(medium_ratio)} 个') 
    print(f'低像素比例: {len(low_ratio)} 个')
    print(f'结果保存在: {debug_dir}')
    
    return panels, debug_dir

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'DJI_0518_W.JPG'
    
    panels, debug_dir = improved_detection(image_path) 