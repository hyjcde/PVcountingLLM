#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def analyze_panel_results(json_file: str):
    """分析光伏板检测结果"""
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    panels = data['panels']
    total_panels = len(panels)
    
    print("=== RGB光伏板分割结果分析 ===")
    print(f"总检测面板数: {total_panels}")
    print(f"图像尺寸: {data['image_size']['width']} x {data['image_size']['height']}")
    print(f"检测方法: {data['method']}")
    print(f"检测时间: {data['detection_time']}")
    
    # 统计面板尺寸
    widths = [panel['position']['width'] for panel in panels]
    heights = [panel['position']['height'] for panel in panels]
    areas = [panel['position']['width'] * panel['position']['height'] for panel in panels]
    aspect_ratios = [panel['aspect_ratio'] for panel in panels]
    pixel_ratios = [panel['panel_pixel_ratio'] for panel in panels]
    
    print(f"\n=== 面板尺寸统计 ===")
    print(f"宽度范围: {min(widths)} - {max(widths)} 像素 (平均: {np.mean(widths):.1f})")
    print(f"高度范围: {min(heights)} - {max(heights)} 像素 (平均: {np.mean(heights):.1f})")
    print(f"面积范围: {min(areas)} - {max(areas)} 像素² (平均: {np.mean(areas):.1f})")
    print(f"长宽比范围: {min(aspect_ratios):.2f} - {max(aspect_ratios):.2f} (平均: {np.mean(aspect_ratios):.2f})")
    print(f"像素比例范围: {min(pixel_ratios):.2f} - {max(pixel_ratios):.2f} (平均: {np.mean(pixel_ratios):.2f})")
    
    # 统计网格分布
    rows = [panel['row'] for panel in panels]
    columns = [panel['column'] for panel in panels]
    
    print(f"\n=== 网格分布统计 ===")
    print(f"行范围: {min(rows)} - {max(rows)} (共 {max(rows)} 行)")
    print(f"列范围: {min(columns)} - {max(columns)} (共 {max(columns)} 列)")
    
    # 按行统计面板数量
    row_counts = {}
    for panel in panels:
        row = panel['row']
        if row not in row_counts:
            row_counts[row] = 0
        row_counts[row] += 1
    
    print(f"\n=== 各行面板数量 ===")
    for row in sorted(row_counts.keys()):
        print(f"第 {row} 行: {row_counts[row]} 个面板")
    
    # 质量评估
    print(f"\n=== 质量评估 ===")
    
    # 合理尺寸的面板
    reasonable_size_panels = [p for p in panels if 50 <= p['position']['width'] <= 500 and 50 <= p['position']['height'] <= 500]
    print(f"合理尺寸面板 (50-500像素): {len(reasonable_size_panels)} / {total_panels} ({len(reasonable_size_panels)/total_panels*100:.1f}%)")
    
    # 合理长宽比的面板
    reasonable_aspect_panels = [p for p in panels if 0.5 <= p['aspect_ratio'] <= 2.0]
    print(f"合理长宽比面板 (0.5-2.0): {len(reasonable_aspect_panels)} / {total_panels} ({len(reasonable_aspect_panels)/total_panels*100:.1f}%)")
    
    # 高像素比例的面板
    high_pixel_ratio_panels = [p for p in panels if p['panel_pixel_ratio'] >= 0.7]
    print(f"高像素比例面板 (≥70%): {len(high_pixel_ratio_panels)} / {total_panels} ({len(high_pixel_ratio_panels)/total_panels*100:.1f}%)")
    
    # 综合质量评估
    quality_panels = [p for p in panels if 
                     50 <= p['position']['width'] <= 500 and 
                     50 <= p['position']['height'] <= 500 and
                     0.5 <= p['aspect_ratio'] <= 2.0 and
                     p['panel_pixel_ratio'] >= 0.5]
    
    print(f"高质量面板 (综合评估): {len(quality_panels)} / {total_panels} ({len(quality_panels)/total_panels*100:.1f}%)")
    
    # 估算实际光伏板数量
    print(f"\n=== 实际光伏板数量估算 ===")
    
    # 基于网格分布估算
    max_row = max(rows)
    max_col = max(columns)
    theoretical_max = max_row * max_col
    print(f"理论最大数量 ({max_row} x {max_col}): {theoretical_max}")
    
    # 基于质量过滤估算
    print(f"基于质量过滤的估算: {len(quality_panels)}")
    
    # 基于面积聚类估算
    median_area = np.median(areas)
    similar_area_panels = [p for p in panels if 0.5 * median_area <= (p['position']['width'] * p['position']['height']) <= 2.0 * median_area]
    print(f"基于面积相似性的估算: {len(similar_area_panels)}")
    
    return {
        'total_panels': total_panels,
        'quality_panels': len(quality_panels),
        'reasonable_size': len(reasonable_size_panels),
        'reasonable_aspect': len(reasonable_aspect_panels),
        'high_pixel_ratio': len(high_pixel_ratio_panels),
        'similar_area': len(similar_area_panels),
        'grid_estimate': theoretical_max,
        'statistics': {
            'width_range': (min(widths), max(widths)),
            'height_range': (min(heights), max(heights)),
            'area_range': (min(areas), max(areas)),
            'aspect_ratio_range': (min(aspect_ratios), max(aspect_ratios)),
            'pixel_ratio_range': (min(pixel_ratios), max(pixel_ratios))
        }
    }


def main():
    parser = argparse.ArgumentParser(description='分析RGB光伏板检测结果')
    parser.add_argument('--json', '-j', type=str, required=True, help='JSON结果文件路径')
    args = parser.parse_args()
    
    results = analyze_panel_results(args.json)
    
    return results


if __name__ == "__main__":
    main() 