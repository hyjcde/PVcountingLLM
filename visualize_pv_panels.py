#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class PVPanelVisualizer:
    """
    光伏板可视化和编号工具
    用于清晰地显示检测到的光伏板并进行编号
    """
    
    def __init__(self):
        self.rgb_image = None
        self.panels = []
        self.image_height = 0
        self.image_width = 0
    
    def load_image_and_results(self, image_path: str, results_path: str):
        """加载图像和检测结果"""
        # 加载图像
        self.rgb_image = cv2.imread(image_path)
        if self.rgb_image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        # 转换BGR到RGB (OpenCV使用BGR，matplotlib使用RGB)
        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width = self.rgb_image.shape[:2]
        print(f"图像尺寸: {self.image_width} x {self.image_height}")
        
        # 加载检测结果
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        self.panels = results_data.get('panels', [])
        print(f"加载了 {len(self.panels)} 个光伏板的检测结果")
        
        return self.rgb_image, self.panels
    
    def create_opencv_visualization(self, output_path: str = None) -> np.ndarray:
        """使用OpenCV创建可视化图像"""
        print("创建OpenCV可视化图像...")
        
        # 创建图像副本
        vis_img = self.rgb_image.copy()
        # 转换回BGR用于OpenCV操作
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # 定义颜色 (BGR格式)
        colors = {
            'high_quality': (0, 255, 0),    # 绿色 - 高质量
            'medium_quality': (0, 255, 255), # 黄色 - 中等质量
            'low_quality': (0, 0, 255),     # 红色 - 低质量
            'default': (255, 0, 0)          # 蓝色 - 默认
        }
        
        # 绘制每个面板
        for i, panel in enumerate(self.panels):
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 根据质量得分选择颜色
            size_score = panel.get("size_match_score", 0)
            if size_score > 0.8:
                color = colors['high_quality']
                quality_text = "High"
            elif size_score > 0.6:
                color = colors['medium_quality']
                quality_text = "Med"
            else:
                color = colors['low_quality']
                quality_text = "Low"
            
            # 绘制矩形边框
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 4)
            
            # 绘制半透明填充
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(vis_img, 0.8, overlay, 0.2, 0, vis_img)
            
            # 计算文字位置
            text_x = x + 5
            text_y = y + 25
            
            # 绘制面板ID (大号字体)
            cv2.putText(vis_img, panel["id"], 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(vis_img, panel["id"], 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # 绘制网格位置
            grid_pos = panel.get("grid_position", f"#{i+1}")
            cv2.putText(vis_img, grid_pos, 
                       (text_x, text_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis_img, grid_pos, 
                       (text_x, text_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 绘制尺寸信息
            size_text = f"{w}x{h}"
            cv2.putText(vis_img, size_text, 
                       (text_x, text_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(vis_img, size_text, 
                       (text_x, text_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # 绘制质量信息
            if "blue_pixel_ratio" in panel:
                ratio_text = f"Blue:{panel['blue_pixel_ratio']:.2f}"
                cv2.putText(vis_img, ratio_text, 
                           (text_x, text_y + 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
                cv2.putText(vis_img, ratio_text, 
                           (text_x, text_y + 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # 添加图例
        self._add_legend_opencv(vis_img, colors)
        
        # 添加统计信息
        self._add_statistics_opencv(vis_img)
        
        # 保存图像
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"OpenCV可视化图像已保存到: {output_path}")
        
        # 转换回RGB用于返回
        return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    def create_matplotlib_visualization(self, output_path: str = None, figsize: tuple = (20, 15)):
        """使用Matplotlib创建高质量可视化图像"""
        print("创建Matplotlib可视化图像...")
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 显示原始图像
        ax.imshow(self.rgb_image)
        
        # 定义颜色映射
        color_map = {
            'high_quality': 'green',
            'medium_quality': 'yellow', 
            'low_quality': 'red',
            'default': 'blue'
        }
        
        # 绘制每个面板
        for i, panel in enumerate(self.panels):
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 根据质量得分选择颜色
            size_score = panel.get("size_match_score", 0)
            if size_score > 0.8:
                color = color_map['high_quality']
                alpha = 0.7
            elif size_score > 0.6:
                color = color_map['medium_quality']
                alpha = 0.6
            else:
                color = color_map['low_quality']
                alpha = 0.5
            
            # 创建矩形
            rect = patches.Rectangle((x, y), w, h, 
                                   linewidth=3, 
                                   edgecolor=color, 
                                   facecolor=color, 
                                   alpha=alpha)
            ax.add_patch(rect)
            
            # 添加文字标注
            # 面板ID
            ax.text(x + 5, y + 20, panel["id"], 
                   fontsize=12, fontweight='bold', 
                   color='white', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            # 网格位置
            grid_pos = panel.get("grid_position", f"#{i+1}")
            ax.text(x + 5, y + 40, grid_pos, 
                   fontsize=10, 
                   color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
            
            # 尺寸信息
            size_text = f"{w}×{h}"
            ax.text(x + 5, y + 60, size_text, 
                   fontsize=8, 
                   color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
        
        # 设置标题
        title = f"光伏板检测结果 - 共检测到 {len(self.panels)} 个面板"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加图例
        self._add_legend_matplotlib(ax, color_map)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Matplotlib可视化图像已保存到: {output_path}")
        
        # 显示图像
        plt.show()
        
        return fig, ax
    
    def create_numbered_grid_visualization(self, output_path: str = None):
        """创建带编号的网格可视化"""
        print("创建带编号的网格可视化...")
        
        # 创建图像副本
        vis_img = self.rgb_image.copy()
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # 按行列排序面板
        sorted_panels = sorted(self.panels, key=lambda p: (p.get('row', 0), p.get('column', 0)))
        
        # 绘制每个面板
        for i, panel in enumerate(sorted_panels):
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 使用统一的绿色
            color = (0, 255, 0)  # 绿色
            
            # 绘制矩形边框
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 3)
            
            # 计算中心位置用于编号
            center_x = x + w // 2
            center_y = y + h // 2
            
            # 绘制大号编号
            panel_number = str(i + 1)
            font_scale = 1.2
            thickness = 3
            
            # 获取文字尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                panel_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # 计算文字位置（居中）
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            
            # 绘制背景圆形
            cv2.circle(vis_img, (center_x, center_y), 
                      max(text_width, text_height) // 2 + 10, 
                      (255, 255, 255), -1)
            cv2.circle(vis_img, (center_x, center_y), 
                      max(text_width, text_height) // 2 + 10, 
                      (0, 0, 0), 2)
            
            # 绘制编号
            cv2.putText(vis_img, panel_number, 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            # 在角落显示详细信息
            info_text = f"PV{i+1:03d}"
            cv2.putText(vis_img, info_text, 
                       (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis_img, info_text, 
                       (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 添加标题
        title_text = f"光伏板编号图 - 共 {len(self.panels)} 个面板"
        cv2.putText(vis_img, title_text, 
                   (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 6)
        cv2.putText(vis_img, title_text, 
                   (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        
        # 保存图像
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"编号网格可视化图像已保存到: {output_path}")
        
        return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    def _add_legend_opencv(self, img: np.ndarray, colors: dict):
        """添加图例到OpenCV图像"""
        legend_x = 50
        legend_y = self.image_height - 200
        
        # 背景
        cv2.rectangle(img, (legend_x - 10, legend_y - 30), 
                     (legend_x + 200, legend_y + 100), (255, 255, 255), -1)
        cv2.rectangle(img, (legend_x - 10, legend_y - 30), 
                     (legend_x + 200, legend_y + 100), (0, 0, 0), 2)
        
        # 标题
        cv2.putText(img, "Quality Legend:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 图例项
        items = [
            ("High Quality", colors['high_quality']),
            ("Medium Quality", colors['medium_quality']),
            ("Low Quality", colors['low_quality'])
        ]
        
        for i, (label, color) in enumerate(items):
            y_pos = legend_y + 20 + i * 25
            cv2.rectangle(img, (legend_x, y_pos), (legend_x + 15, y_pos + 15), color, -1)
            cv2.putText(img, label, (legend_x + 25, y_pos + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def _add_statistics_opencv(self, img: np.ndarray):
        """添加统计信息到OpenCV图像"""
        stats_x = self.image_width - 300
        stats_y = 50
        
        # 计算统计信息
        total_panels = len(self.panels)
        high_quality = len([p for p in self.panels if p.get("size_match_score", 0) > 0.8])
        medium_quality = len([p for p in self.panels if 0.6 < p.get("size_match_score", 0) <= 0.8])
        low_quality = len([p for p in self.panels if p.get("size_match_score", 0) <= 0.6])
        
        # 背景
        cv2.rectangle(img, (stats_x - 10, stats_y - 10), 
                     (stats_x + 280, stats_y + 120), (255, 255, 255), -1)
        cv2.rectangle(img, (stats_x - 10, stats_y - 10), 
                     (stats_x + 280, stats_y + 120), (0, 0, 0), 2)
        
        # 统计文本
        stats_text = [
            f"Total Panels: {total_panels}",
            f"High Quality: {high_quality} ({high_quality/total_panels*100:.1f}%)",
            f"Medium Quality: {medium_quality} ({medium_quality/total_panels*100:.1f}%)",
            f"Low Quality: {low_quality} ({low_quality/total_panels*100:.1f}%)"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(img, text, (stats_x, stats_y + 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def _add_legend_matplotlib(self, ax, color_map: dict):
        """添加图例到Matplotlib图像"""
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor=color_map['high_quality'], label='高质量 (>0.8)'),
            Patch(facecolor=color_map['medium_quality'], label='中等质量 (0.6-0.8)'),
            Patch(facecolor=color_map['low_quality'], label='低质量 (<0.6)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fontsize=12)
    
    def generate_panel_report(self, output_path: str = None):
        """生成面板检测报告"""
        print("生成面板检测报告...")
        
        # 计算统计信息
        total_panels = len(self.panels)
        if total_panels == 0:
            print("没有检测到面板")
            return
        
        high_quality = [p for p in self.panels if p.get("size_match_score", 0) > 0.8]
        medium_quality = [p for p in self.panels if 0.6 < p.get("size_match_score", 0) <= 0.8]
        low_quality = [p for p in self.panels if p.get("size_match_score", 0) <= 0.6]
        
        # 尺寸统计
        widths = [p["position"]["width"] for p in self.panels]
        heights = [p["position"]["height"] for p in self.panels]
        
        report = f"""
=== 光伏板检测报告 ===
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

检测概况:
- 总计检测到: {total_panels} 个光伏板
- 高质量面板: {len(high_quality)} 个 ({len(high_quality)/total_panels*100:.1f}%)
- 中等质量面板: {len(medium_quality)} 个 ({len(medium_quality)/total_panels*100:.1f}%)
- 低质量面板: {len(low_quality)} 个 ({len(low_quality)/total_panels*100:.1f}%)

尺寸统计:
- 平均宽度: {np.mean(widths):.1f} 像素
- 平均高度: {np.mean(heights):.1f} 像素
- 宽度范围: {min(widths)} - {max(widths)} 像素
- 高度范围: {min(heights)} - {max(heights)} 像素

面板详细信息:
"""
        
        # 添加每个面板的详细信息
        for i, panel in enumerate(self.panels, 1):
            pos = panel["position"]
            score = panel.get("size_match_score", 0)
            blue_ratio = panel.get("blue_pixel_ratio", 0)
            
            report += f"""
面板 {i:2d}: {panel['id']}
  - 位置: ({pos['x']}, {pos['y']})
  - 尺寸: {pos['width']} × {pos['height']} 像素
  - 网格位置: {panel.get('grid_position', 'N/A')}
  - 质量得分: {score:.3f}
  - 蓝色像素比例: {blue_ratio:.3f}
"""
        
        print(report)
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存到: {output_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='光伏板可视化和编号工具')
    parser.add_argument('--image', '-i', type=str, required=True, help='RGB图像路径')
    parser.add_argument('--results', '-r', type=str, required=True, help='检测结果JSON文件路径')
    parser.add_argument('--output_dir', '-o', type=str, default='visualization_output', help='输出目录')
    parser.add_argument('--all', action='store_true', help='生成所有类型的可视化')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建可视化器
    visualizer = PVPanelVisualizer()
    
    # 加载图像和结果
    try:
        visualizer.load_image_and_results(args.image, args.results)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    if args.all or True:  # 默认生成所有可视化
        # 1. OpenCV可视化
        opencv_output = os.path.join(args.output_dir, "opencv_visualization.jpg")
        visualizer.create_opencv_visualization(opencv_output)
        
        # 2. Matplotlib可视化
        matplotlib_output = os.path.join(args.output_dir, "matplotlib_visualization.png")
        visualizer.create_matplotlib_visualization(matplotlib_output)
        
        # 3. 编号网格可视化
        numbered_output = os.path.join(args.output_dir, "numbered_panels.jpg")
        visualizer.create_numbered_grid_visualization(numbered_output)
        
        # 4. 生成报告
        report_output = os.path.join(args.output_dir, "panel_detection_report.txt")
        visualizer.generate_panel_report(report_output)
        
        print(f"\n所有可视化文件已保存到目录: {args.output_dir}")


if __name__ == "__main__":
    main() 