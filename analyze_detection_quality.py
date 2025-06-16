#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def analyze_detection_quality(json_file: str, result_image_path: str = None):
    """分析光伏板检测质量"""
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    panels = data['panels']
    total_panels = len(panels)
    
    print("=== 智能光伏板检测结果分析 ===")
    print(f"总检测面板数: {total_panels}")
    print(f"图像尺寸: {data['image_size']['width']} x {data['image_size']['height']}")
    print(f"检测方法: {data['method']}")
    print(f"检测时间: {data['detection_time']}")
    
    # 按检测方法分类
    contour_panels = [p for p in panels if p.get('detection_method') == 'contour']
    grid_panels = [p for p in panels if p.get('detection_method') == 'grid']
    
    print(f"\n=== 检测方法分布 ===")
    print(f"轮廓检测: {len(contour_panels)} 个面板")
    print(f"网格检测: {len(grid_panels)} 个面板")
    
    # 统计面板尺寸
    widths = [panel['position']['width'] for panel in panels]
    heights = [panel['position']['height'] for panel in panels]
    areas = [panel['area'] for panel in panels]
    aspect_ratios = [panel['aspect_ratio'] for panel in panels]
    
    print(f"\n=== 面板尺寸统计 ===")
    print(f"宽度范围: {min(widths)} - {max(widths)} 像素 (平均: {np.mean(widths):.1f})")
    print(f"高度范围: {min(heights)} - {max(heights)} 像素 (平均: {np.mean(heights):.1f})")
    print(f"面积范围: {min(areas)} - {max(areas)} 像素² (平均: {np.mean(areas):.1f})")
    print(f"长宽比范围: {min(aspect_ratios):.2f} - {max(aspect_ratios):.2f} (平均: {np.mean(aspect_ratios):.2f})")
    
    # 质量评估
    print(f"\n=== 质量评估 ===")
    
    # 合理尺寸的面板 (基于实际光伏板尺寸)
    reasonable_size_panels = [p for p in panels if 30 <= p['position']['width'] <= 200 and 30 <= p['position']['height'] <= 200]
    print(f"合理尺寸面板 (30-200像素): {len(reasonable_size_panels)} / {total_panels} ({len(reasonable_size_panels)/total_panels*100:.1f}%)")
    
    # 合理长宽比的面板
    reasonable_aspect_panels = [p for p in panels if 0.7 <= p['aspect_ratio'] <= 1.4]
    print(f"合理长宽比面板 (0.7-1.4): {len(reasonable_aspect_panels)} / {total_panels} ({len(reasonable_aspect_panels)/total_panels*100:.1f}%)")
    
    # 面积一致性分析
    median_area = np.median(areas)
    consistent_area_panels = [p for p in panels if 0.5 * median_area <= p['area'] <= 2.0 * median_area]
    print(f"面积一致性面板 (±50%中位数): {len(consistent_area_panels)} / {total_panels} ({len(consistent_area_panels)/total_panels*100:.1f}%)")
    
    # 综合质量评估
    quality_panels = [p for p in panels if 
                     30 <= p['position']['width'] <= 200 and 
                     30 <= p['position']['height'] <= 200 and
                     0.7 <= p['aspect_ratio'] <= 1.4 and
                     0.5 * median_area <= p['area'] <= 2.0 * median_area]
    
    print(f"高质量面板 (综合评估): {len(quality_panels)} / {total_panels} ({len(quality_panels)/total_panels*100:.1f}%)")
    
    # 空间分布分析
    print(f"\n=== 空间分布分析 ===")
    
    # 计算面板密度
    image_area = data['image_size']['width'] * data['image_size']['height']
    panel_density = total_panels / (image_area / 1000000)  # 每平方百万像素的面板数
    print(f"面板密度: {panel_density:.2f} 个/百万像素²")
    
    # 分析面板间距
    x_positions = [p['position']['x'] for p in panels]
    y_positions = [p['position']['y'] for p in panels]
    
    # 计算相邻面板的平均间距
    x_positions.sort()
    y_positions.sort()
    
    x_gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1) if x_positions[i+1] - x_positions[i] < 100]
    y_gaps = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1) if y_positions[i+1] - y_positions[i] < 100]
    
    if x_gaps:
        print(f"水平间距: 平均 {np.mean(x_gaps):.1f} 像素")
    if y_gaps:
        print(f"垂直间距: 平均 {np.mean(y_gaps):.1f} 像素")
    
    # 估算实际光伏板数量
    print(f"\n=== 实际光伏板数量估算 ===")
    
    # 基于质量过滤的估算
    print(f"基于质量过滤: {len(quality_panels)} 个")
    
    # 基于面积聚类的估算
    print(f"基于面积一致性: {len(consistent_area_panels)} 个")
    
    # 基于尺寸一致性的估算
    print(f"基于尺寸合理性: {len(reasonable_size_panels)} 个")
    
    # 创建可视化图表
    create_analysis_plots(panels, quality_panels)
    
    return {
        'total_panels': total_panels,
        'contour_panels': len(contour_panels),
        'grid_panels': len(grid_panels),
        'quality_panels': len(quality_panels),
        'reasonable_size': len(reasonable_size_panels),
        'reasonable_aspect': len(reasonable_aspect_panels),
        'consistent_area': len(consistent_area_panels),
        'panel_density': panel_density,
        'statistics': {
            'width_stats': (min(widths), max(widths), np.mean(widths), np.std(widths)),
            'height_stats': (min(heights), max(heights), np.mean(heights), np.std(heights)),
            'area_stats': (min(areas), max(areas), np.mean(areas), np.std(areas)),
            'aspect_ratio_stats': (min(aspect_ratios), max(aspect_ratios), np.mean(aspect_ratios), np.std(aspect_ratios))
        }
    }


def create_analysis_plots(all_panels: List[Dict[str, Any]], quality_panels: List[Dict[str, Any]]):
    """创建分析图表"""
    
    # 提取数据
    all_widths = [p['position']['width'] for p in all_panels]
    all_heights = [p['position']['height'] for p in all_panels]
    all_areas = [p['area'] for p in all_panels]
    all_aspects = [p['aspect_ratio'] for p in all_panels]
    
    quality_widths = [p['position']['width'] for p in quality_panels]
    quality_heights = [p['position']['height'] for p in quality_panels]
    quality_areas = [p['area'] for p in quality_panels]
    quality_aspects = [p['aspect_ratio'] for p in quality_panels]
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('光伏板检测质量分析', fontsize=16, fontweight='bold')
    
    # 1. 宽度分布
    axes[0, 0].hist(all_widths, bins=30, alpha=0.7, label='所有面板', color='lightblue')
    axes[0, 0].hist(quality_widths, bins=30, alpha=0.7, label='高质量面板', color='darkblue')
    axes[0, 0].set_xlabel('宽度 (像素)')
    axes[0, 0].set_ylabel('数量')
    axes[0, 0].set_title('面板宽度分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 高度分布
    axes[0, 1].hist(all_heights, bins=30, alpha=0.7, label='所有面板', color='lightgreen')
    axes[0, 1].hist(quality_heights, bins=30, alpha=0.7, label='高质量面板', color='darkgreen')
    axes[0, 1].set_xlabel('高度 (像素)')
    axes[0, 1].set_ylabel('数量')
    axes[0, 1].set_title('面板高度分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 面积分布
    axes[0, 2].hist(all_areas, bins=30, alpha=0.7, label='所有面板', color='lightcoral')
    axes[0, 2].hist(quality_areas, bins=30, alpha=0.7, label='高质量面板', color='darkred')
    axes[0, 2].set_xlabel('面积 (像素²)')
    axes[0, 2].set_ylabel('数量')
    axes[0, 2].set_title('面板面积分布')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 长宽比分布
    axes[1, 0].hist(all_aspects, bins=30, alpha=0.7, label='所有面板', color='lightyellow')
    axes[1, 0].hist(quality_aspects, bins=30, alpha=0.7, label='高质量面板', color='orange')
    axes[1, 0].set_xlabel('长宽比')
    axes[1, 0].set_ylabel('数量')
    axes[1, 0].set_title('面板长宽比分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 宽度vs高度散点图
    all_x = [p['position']['x'] for p in all_panels]
    all_y = [p['position']['y'] for p in all_panels]
    quality_x = [p['position']['x'] for p in quality_panels]
    quality_y = [p['position']['y'] for p in quality_panels]
    
    axes[1, 1].scatter(all_x, all_y, alpha=0.5, s=10, label='所有面板', color='lightgray')
    axes[1, 1].scatter(quality_x, quality_y, alpha=0.7, s=15, label='高质量面板', color='red')
    axes[1, 1].set_xlabel('X坐标 (像素)')
    axes[1, 1].set_ylabel('Y坐标 (像素)')
    axes[1, 1].set_title('面板空间分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].invert_yaxis()  # 反转Y轴以匹配图像坐标
    
    # 6. 检测方法统计
    contour_count = len([p for p in all_panels if p.get('detection_method') == 'contour'])
    grid_count = len([p for p in all_panels if p.get('detection_method') == 'grid'])
    
    methods = ['轮廓检测', '网格检测']
    counts = [contour_count, grid_count]
    colors = ['skyblue', 'lightgreen']
    
    axes[1, 2].pie(counts, labels=methods, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('检测方法分布')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'detection_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"分析图表已保存到: {os.path.join(output_dir, 'detection_analysis.png')}")
    
    # 显示图表
    plt.show()


def compare_detection_results(json_files: List[str]):
    """比较不同检测方法的结果"""
    
    results = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        method_name = data.get('method', os.path.basename(json_file))
        total_panels = data.get('total_panels', len(data.get('panels', [])))
        
        results.append({
            'method': method_name,
            'total_panels': total_panels,
            'file': json_file
        })
    
    print("\n=== 检测方法比较 ===")
    for result in results:
        print(f"{result['method']}: {result['total_panels']} 个面板")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='分析光伏板检测质量')
    parser.add_argument('--json', '-j', type=str, required=True, help='JSON结果文件路径')
    parser.add_argument('--image', '-i', type=str, help='结果图像路径（可选）')
    parser.add_argument('--compare', '-c', nargs='+', help='比较多个JSON文件')
    args = parser.parse_args()
    
    # 分析主要结果
    analysis_result = analyze_detection_quality(args.json, args.image)
    
    # 如果提供了比较文件，进行比较
    if args.compare:
        compare_files = [args.json] + args.compare
        compare_detection_results(compare_files)
    
    return analysis_result


if __name__ == "__main__":
    main() 