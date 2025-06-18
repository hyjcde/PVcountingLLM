#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime

import cv2
import numpy as np


def optimized_morphology_detection(image_path):
    print('开始优化的形态学光伏板检测...')
    
    # 加载图像
    rgb_image = cv2.imread(image_path)
    height, width = rgb_image.shape[:2]
    print(f'图像尺寸: {width} x {height}')
    
    # 创建调试目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_dir = f'optimized_morphology_{timestamp}'
    os.makedirs(debug_dir, exist_ok=True)
    
    # 1. 更精确的HSV色彩空间提取
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    
    # 针对光伏板的精确蓝色范围
    lower_pv_blue = np.array([95, 40, 40])
    upper_pv_blue = np.array([125, 255, 180])
    blue_mask = cv2.inRange(hsv, lower_pv_blue, upper_pv_blue)
    
    # 深色光伏板范围
    lower_dark_pv = np.array([0, 0, 0])
    upper_dark_pv = np.array([180, 255, 90])
    dark_mask = cv2.inRange(hsv, lower_dark_pv, upper_dark_pv)
    
    # 合并掩码
    combined_mask = cv2.bitwise_or(blue_mask, dark_mask)
    
    # 去除小噪声
    noise_kernel = np.ones((2, 2), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, noise_kernel)
    
    # 2. 更精细的形态学闭运算
    print('应用优化的形态学闭运算...')
    
    # 第一步：连接细小断裂
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    closed1 = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel1)
    
    # 第二步：连接水平方向的断裂（重要！）
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 2))
    closed2 = cv2.morphologyEx(closed1, cv2.MORPH_CLOSE, kernel2)
    
    # 第三步：连接垂直方向的断裂
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
    closed3 = cv2.morphologyEx(closed2, cv2.MORPH_CLOSE, kernel3)
    
    # 第四步：形成完整矩形
    kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    final_closed = cv2.morphologyEx(closed3, cv2.MORPH_CLOSE, kernel4)
    
    # 最后填充小孔洞
    kernel_fill = np.ones((4, 4), np.uint8)
    final_closed = cv2.morphologyEx(final_closed, cv2.MORPH_CLOSE, kernel_fill)
    
    # 3. 检测轮廓
    contours, _ = cv2.findContours(final_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'找到 {len(contours)} 个轮廓')
    
    # 4. 智能筛选光伏板
    panels = []
    panel_id = 1
    
    # 预筛选：收集合理大小的轮廓
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 2000 <= area <= 50000:  # 基于图像尺寸的合理面积范围
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 30:  # 最小尺寸要求
                aspect_ratio = w / float(h)
                if 0.8 <= aspect_ratio <= 4.0:  # 合理的长宽比
                    valid_contours.append((contour, area, x, y, w, h, aspect_ratio))
    
    print(f'预筛选后剩余 {len(valid_contours)} 个候选轮廓')
    
    # 计算统计信息用于进一步筛选
    if valid_contours:
        areas = [item[1] for item in valid_contours]
        aspects = [item[6] for item in valid_contours]
        
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        mean_aspect = np.mean(aspects)
        std_aspect = np.std(aspects)
        
        print(f'有效轮廓统计:')
        print(f'  面积: 均值={mean_area:.1f}, 标准差={std_area:.1f}')
        print(f'  长宽比: 均值={mean_aspect:.2f}, 标准差={std_aspect:.2f}')
        
        # 更严格的阈值
        area_lower = max(2000, mean_area - 1.5 * std_area)
        area_upper = mean_area + 2 * std_area
        aspect_lower = max(0.8, mean_aspect - 1 * std_aspect)
        aspect_upper = min(4.0, mean_aspect + 1 * std_aspect)
        
        print(f'最终阈值:')
        print(f'  面积: [{area_lower:.1f}, {area_upper:.1f}]')
        print(f'  长宽比: [{aspect_lower:.2f}, {aspect_upper:.2f}]')
        
        # 最终筛选
        for contour, area, x, y, w, h, aspect_ratio in valid_contours:
            if area_lower <= area <= area_upper and aspect_lower <= aspect_ratio <= aspect_upper:
                # 计算形状质量指标
                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0
                
                # 计算轮廓的凸性
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # 只接受足够矩形和凸的形状
                if rectangularity > 0.65 and solidity > 0.85:
                    panel = {
                        'id': f'PV{panel_id:03d}',
                        'position': {
                            'x': int(x), 
                            'y': int(y), 
                            'width': int(w), 
                            'height': int(h),
                            'center_x': int(x + w//2),
                            'center_y': int(y + h//2)
                        },
                        'area': float(area),
                        'aspect_ratio': float(aspect_ratio),
                        'rectangularity': float(rectangularity),
                        'solidity': float(solidity),
                        'quality_score': float((rectangularity + solidity) / 2)
                    }
                    panels.append(panel)
                    panel_id += 1
    
    # 5. 按质量排序
    panels.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # 6. 可视化结果
    result_img = rgb_image.copy()
    
    for i, panel in enumerate(panels):
        pos = panel['position']
        x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
        
        # 根据质量分数选择颜色
        quality = panel['quality_score']
        if quality > 0.85:
            color = (0, 255, 0)  # 绿色 - 高质量
        elif quality > 0.75:
            color = (0, 255, 255)  # 黄色 - 中等质量
        else:
            color = (255, 0, 0)  # 蓝色 - 低质量
        
        # 绘制矩形框
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 4)
        
        # 显示编号和质量分数
        label = f'{i+1}'
        cv2.putText(result_img, label, (x + w//2 - 20, y + h//2 + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
        cv2.putText(result_img, label, (x + w//2 - 20, y + h//2 + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # 显示质量分数
        score_text = f'{quality:.2f}'
        cv2.putText(result_img, score_text, (x + 5, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_img, score_text, (x + 5, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 保存所有调试图像
    cv2.imwrite(f'{debug_dir}/01_original.jpg', rgb_image)
    cv2.imwrite(f'{debug_dir}/02_hsv.jpg', hsv)
    cv2.imwrite(f'{debug_dir}/03_blue_mask.jpg', blue_mask)
    cv2.imwrite(f'{debug_dir}/04_dark_mask.jpg', dark_mask)
    cv2.imwrite(f'{debug_dir}/05_combined_mask.jpg', combined_mask)
    cv2.imwrite(f'{debug_dir}/06_closed1.jpg', closed1)
    cv2.imwrite(f'{debug_dir}/07_closed2.jpg', closed2)
    cv2.imwrite(f'{debug_dir}/08_closed3.jpg', closed3)
    cv2.imwrite(f'{debug_dir}/09_final_closed.jpg', final_closed)
    cv2.imwrite(f'{debug_dir}/10_final_result.jpg', result_img)
    
    # 统计质量分布
    high_quality = [p for p in panels if p['quality_score'] > 0.85]
    medium_quality = [p for p in panels if 0.75 < p['quality_score'] <= 0.85]
    low_quality = [p for p in panels if p['quality_score'] <= 0.75]
    
    # 保存详细结果
    result_data = {
        'method': 'optimized_morphology_detection',
        'total_panels': len(panels),
        'detection_time': datetime.now().isoformat(),
        'image_size': {'width': width, 'height': height},
        'quality_distribution': {
            'high_quality': len(high_quality),
            'medium_quality': len(medium_quality),
            'low_quality': len(low_quality)
        },
        'panels': panels
    }
    
    with open(f'{debug_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f'\n=== 优化形态学检测结果 ===')
    print(f'检测到 {len(panels)} 个光伏板')
    print(f'高质量 (>0.85): {len(high_quality)} 个')
    print(f'中等质量 (0.75-0.85): {len(medium_quality)} 个')
    print(f'低质量 (≤0.75): {len(low_quality)} 个')
    print(f'平均质量分数: {np.mean([p["quality_score"] for p in panels]):.3f}')
    print(f'结果保存在: {debug_dir}')
    
    return panels, debug_dir

if __name__ == "__main__":
    # 运行优化检测
    panels, debug_dir = optimized_morphology_detection('DJI_0518_W.JPG') 