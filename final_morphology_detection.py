#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime

import cv2
import numpy as np


def final_morphology_detection(image_path):
    """
    最终的形态学光伏板检测算法
    结合HSV色彩空间和精细的形态学闭运算
    """
    print('开始最终形态学光伏板检测...')
    
    # 加载图像
    rgb_image = cv2.imread(image_path)
    height, width = rgb_image.shape[:2]
    print(f'图像尺寸: {width} x {height}')
    
    # 创建调试目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_dir = f'final_morphology_{timestamp}'
    os.makedirs(debug_dir, exist_ok=True)
    
    # 1. 多重HSV色彩空间提取
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    
    # 方法1: 标准蓝色光伏板
    lower_blue_std = np.array([95, 40, 40])
    upper_blue_std = np.array([125, 255, 180])
    blue_mask1 = cv2.inRange(hsv, lower_blue_std, upper_blue_std)
    
    # 方法2: 深蓝色光伏板
    lower_blue_dark = np.array([100, 60, 30])
    upper_blue_dark = np.array([130, 255, 120])
    blue_mask2 = cv2.inRange(hsv, lower_blue_dark, upper_blue_dark)
    
    # 方法3: 黑色/深色光伏板
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # 方法4: 灰度阈值检测
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    _, gray_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 合并所有掩码
    combined_mask = cv2.bitwise_or(blue_mask1, blue_mask2)
    combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
    combined_mask = cv2.bitwise_or(combined_mask, gray_mask)
    
    # 2. 精细的形态学闭运算序列
    print('应用精细的形态学闭运算序列...')
    
    # 步骤1: 去除小噪声
    noise_kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, noise_kernel)
    
    # 步骤2: 连接细小断裂 (小核)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    closed_small = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_small)
    
    # 步骤3: 连接水平断裂 (水平长核 - 关键!)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    closed_horizontal = cv2.morphologyEx(closed_small, cv2.MORPH_CLOSE, kernel_horizontal)
    
    # 步骤4: 连接垂直断裂 (垂直长核)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed_vertical = cv2.morphologyEx(closed_horizontal, cv2.MORPH_CLOSE, kernel_vertical)
    
    # 步骤5: 形成完整矩形
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    closed_rect = cv2.morphologyEx(closed_vertical, cv2.MORPH_CLOSE, kernel_rect)
    
    # 步骤6: 填充内部孔洞
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    final_closed = cv2.morphologyEx(closed_rect, cv2.MORPH_CLOSE, kernel_fill)
    
    # 步骤7: 最终平滑
    kernel_smooth = np.ones((3, 3), np.uint8)
    final_closed = cv2.morphologyEx(final_closed, cv2.MORPH_OPEN, kernel_smooth)
    
    # 3. 轮廓检测
    contours, _ = cv2.findContours(final_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'找到 {len(contours)} 个轮廓')
    
    # 4. 多级筛选系统
    panels = []
    panel_id = 1
    
    # 第一级筛选: 基本尺寸和面积
    stage1_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1500 <= area <= 100000:  # 合理的面积范围
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 40 and h >= 25:  # 最小尺寸
                aspect_ratio = w / float(h)
                if 0.5 <= aspect_ratio <= 5.0:  # 宽松的长宽比
                    stage1_contours.append((contour, area, x, y, w, h, aspect_ratio))
    
    print(f'第一级筛选: {len(stage1_contours)} 个候选')
    
    if not stage1_contours:
        print('第一级筛选后无候选轮廓')
        return [], debug_dir
    
    # 第二级筛选: 统计分析
    areas = [item[1] for item in stage1_contours]
    aspects = [item[6] for item in stage1_contours]
    
    # 计算统计信息
    area_median = np.median(areas)
    area_std = np.std(areas)
    aspect_median = np.median(aspects)
    aspect_std = np.std(aspects)
    
    print(f'统计信息:')
    print(f'  面积中位数: {area_median:.1f}, 标准差: {area_std:.1f}')
    print(f'  长宽比中位数: {aspect_median:.2f}, 标准差: {aspect_std:.2f}')
    
    # 动态阈值 (基于中位数而非均值，更鲁棒)
    area_lower = max(1500, area_median - 2 * area_std)
    area_upper = area_median + 3 * area_std
    aspect_lower = max(0.5, aspect_median - 2 * aspect_std)
    aspect_upper = min(5.0, aspect_median + 2 * aspect_std)
    
    print(f'动态阈值:')
    print(f'  面积: [{area_lower:.1f}, {area_upper:.1f}]')
    print(f'  长宽比: [{aspect_lower:.2f}, {aspect_upper:.2f}]')
    
    # 第三级筛选: 形状质量分析
    for contour, area, x, y, w, h, aspect_ratio in stage1_contours:
        if area_lower <= area <= area_upper and aspect_lower <= aspect_ratio <= aspect_upper:
            
            # 计算多个形状质量指标
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # 凸包分析
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 轮廓周长分析
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 椭圆拟合分析
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
                ellipse_ratio = area / ellipse_area if ellipse_area > 0 else 0
            else:
                ellipse_ratio = 0
            
            # 综合质量评分
            quality_score = (
                rectangularity * 0.4 +  # 矩形度权重最高
                solidity * 0.3 +        # 凸性
                min(circularity * 2, 1) * 0.2 +  # 圆形度 (适度)
                ellipse_ratio * 0.1     # 椭圆拟合
            )
            
            # 质量阈值筛选
            if (rectangularity > 0.6 and 
                solidity > 0.8 and 
                quality_score > 0.65):
                
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
                    'metrics': {
                        'area': float(area),
                        'aspect_ratio': float(aspect_ratio),
                        'rectangularity': float(rectangularity),
                        'solidity': float(solidity),
                        'circularity': float(circularity),
                        'ellipse_ratio': float(ellipse_ratio),
                        'quality_score': float(quality_score)
                    }
                }
                panels.append(panel)
                panel_id += 1
    
    print(f'第三级筛选: {len(panels)} 个高质量光伏板')
    
    # 5. 按质量分数排序
    panels.sort(key=lambda x: x['metrics']['quality_score'], reverse=True)
    
    # 6. 高级可视化
    result_img = rgb_image.copy()
    
    for i, panel in enumerate(panels):
        pos = panel['position']
        metrics = panel['metrics']
        x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
        quality = metrics['quality_score']
        
        # 质量颜色编码
        if quality > 0.8:
            color = (0, 255, 0)      # 绿色 - 优秀
            thickness = 5
        elif quality > 0.7:
            color = (0, 255, 255)    # 黄色 - 良好
            thickness = 4
        else:
            color = (255, 165, 0)    # 橙色 - 一般
            thickness = 3
        
        # 绘制主矩形
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, thickness)
        
        # 绘制角点标记
        corner_size = 15
        cv2.rectangle(result_img, (x-corner_size//2, y-corner_size//2), 
                     (x+corner_size//2, y+corner_size//2), color, -1)
        cv2.rectangle(result_img, (x+w-corner_size//2, y-corner_size//2), 
                     (x+w+corner_size//2, y+corner_size//2), color, -1)
        cv2.rectangle(result_img, (x-corner_size//2, y+h-corner_size//2), 
                     (x+corner_size//2, y+h+corner_size//2), color, -1)
        cv2.rectangle(result_img, (x+w-corner_size//2, y+h-corner_size//2), 
                     (x+w+corner_size//2, y+h+corner_size//2), color, -1)
        
        # 显示编号
        label = f'{i+1}'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = x + w//2 - text_size[0]//2
        text_y = y + h//2 + text_size[1]//2
        
        cv2.putText(result_img, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5)
        cv2.putText(result_img, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        
        # 显示质量分数
        score_text = f'{quality:.2f}'
        cv2.putText(result_img, score_text, (x + 5, y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(result_img, score_text, (x + 5, y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 保存所有调试图像
    cv2.imwrite(f'{debug_dir}/01_original.jpg', rgb_image)
    cv2.imwrite(f'{debug_dir}/02_hsv.jpg', hsv)
    cv2.imwrite(f'{debug_dir}/03_blue_mask1.jpg', blue_mask1)
    cv2.imwrite(f'{debug_dir}/04_blue_mask2.jpg', blue_mask2)
    cv2.imwrite(f'{debug_dir}/05_dark_mask.jpg', dark_mask)
    cv2.imwrite(f'{debug_dir}/06_gray_mask.jpg', gray_mask)
    cv2.imwrite(f'{debug_dir}/07_combined_mask.jpg', combined_mask)
    cv2.imwrite(f'{debug_dir}/08_cleaned.jpg', cleaned)
    cv2.imwrite(f'{debug_dir}/09_closed_small.jpg', closed_small)
    cv2.imwrite(f'{debug_dir}/10_closed_horizontal.jpg', closed_horizontal)
    cv2.imwrite(f'{debug_dir}/11_closed_vertical.jpg', closed_vertical)
    cv2.imwrite(f'{debug_dir}/12_closed_rect.jpg', closed_rect)
    cv2.imwrite(f'{debug_dir}/13_final_closed.jpg', final_closed)
    cv2.imwrite(f'{debug_dir}/14_final_result.jpg', result_img)
    
    # 统计和保存结果
    if panels:
        quality_scores = [p['metrics']['quality_score'] for p in panels]
        excellent = [p for p in panels if p['metrics']['quality_score'] > 0.8]
        good = [p for p in panels if 0.7 < p['metrics']['quality_score'] <= 0.8]
        fair = [p for p in panels if p['metrics']['quality_score'] <= 0.7]
        
        result_data = {
            'method': 'final_morphology_detection',
            'total_panels': len(panels),
            'detection_time': datetime.now().isoformat(),
            'image_size': {'width': width, 'height': height},
            'quality_distribution': {
                'excellent': len(excellent),
                'good': len(good),
                'fair': len(fair)
            },
            'statistics': {
                'avg_quality_score': float(np.mean(quality_scores)),
                'min_quality_score': float(np.min(quality_scores)),
                'max_quality_score': float(np.max(quality_scores)),
                'std_quality_score': float(np.std(quality_scores))
            },
            'panels': panels
        }
        
        with open(f'{debug_dir}/results.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f'\n=== 最终形态学检测结果 ===')
        print(f'检测到 {len(panels)} 个光伏板')
        print(f'优秀质量 (>0.8): {len(excellent)} 个')
        print(f'良好质量 (0.7-0.8): {len(good)} 个')
        print(f'一般质量 (≤0.7): {len(fair)} 个')
        print(f'平均质量分数: {np.mean(quality_scores):.3f}')
        print(f'质量分数范围: [{np.min(quality_scores):.3f}, {np.max(quality_scores):.3f}]')
        print(f'结果保存在: {debug_dir}')
    else:
        print('未检测到符合条件的光伏板')
    
    return panels, debug_dir

if __name__ == "__main__":
    # 运行最终检测
    panels, debug_dir = final_morphology_detection('DJI_0518_W.JPG') 