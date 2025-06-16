#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class FixedSizePanelDetector:
    """
    基于固定尺寸特征的光伏板检测算法
    利用光伏板尺寸相对固定的特点进行精确检测
    """
    
    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        self.rgb_image = None
        self.debug_dir = None
        self.image_height = 0
        self.image_width = 0
        
        # 光伏板的预期尺寸范围（像素）
        self.expected_panel_width = (80, 150)   # 预期宽度范围
        self.expected_panel_height = (80, 150)  # 预期高度范围
        self.expected_aspect_ratio = (0.8, 1.3) # 预期长宽比范围
    
    def load_image(self, image_path: str) -> np.ndarray:
        """加载RGB图像"""
        self.rgb_image = cv2.imread(image_path)
        if self.rgb_image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        self.image_height, self.image_width = self.rgb_image.shape[:2]
        print(f"图像尺寸: {self.image_width} x {self.image_height}")
        
        # 创建调试目录
        if self.debug_mode:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            self.debug_dir = f"fixed_size_debug_{base_name}_{timestamp}"
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # 保存原始图像
            cv2.imwrite(os.path.join(self.debug_dir, "01_original.jpg"), self.rgb_image)
        
        return self.rgb_image
    
    def extract_blue_panels(self) -> np.ndarray:
        """提取蓝色光伏板区域 - 优化HSV范围"""
        print("提取蓝色光伏板区域...")
        
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        
        # 针对光伏板优化的蓝色范围
        lower_blue = np.array([100, 80, 50])   # 蓝色下限
        upper_blue = np.array([130, 255, 180]) # 蓝色上限
        
        # 创建蓝色掩码
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "02_hsv.jpg"), hsv)
            cv2.imwrite(os.path.join(self.debug_dir, "03_blue_mask.jpg"), blue_mask)
        
        return blue_mask
    
    def detect_precise_edges(self) -> np.ndarray:
        """精确的边缘检测"""
        print("精确边缘检测...")
        
        # 转换为灰度图
        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测 - 调整参数以获得更好的边缘
        canny_edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 形态学操作连接断开的边缘
        kernel = np.ones((2, 2), np.uint8)
        canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)
        
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "04_canny_edges.jpg"), canny_edges)
        
        return canny_edges
    
    def estimate_panel_size(self, blue_mask: np.ndarray) -> Tuple[int, int]:
        """估算单个光伏板的尺寸"""
        print("估算光伏板尺寸...")
        
        # 查找轮廓
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        widths = []
        heights = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 30000:  # 过滤合理大小的轮廓
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 只考虑长宽比合理的轮廓
                if 0.7 < aspect_ratio < 1.4:
                    widths.append(w)
                    heights.append(h)
        
        if widths and heights:
            avg_width = int(np.median(widths))
            avg_height = int(np.median(heights))
            print(f"估算的平均面板尺寸: {avg_width} x {avg_height}")
            
            # 更新预期尺寸范围
            self.expected_panel_width = (int(avg_width * 0.8), int(avg_width * 1.2))
            self.expected_panel_height = (int(avg_height * 0.8), int(avg_height * 1.2))
            
            return avg_width, avg_height
        else:
            print("无法估算面板尺寸，使用默认值")
            return 110, 110  # 默认尺寸
    
    def detect_grid_with_fixed_size(self, edges: np.ndarray, panel_width: int, panel_height: int) -> Tuple[List[int], List[int]]:
        """基于固定尺寸检测网格线"""
        print("基于固定尺寸检测网格线...")
        
        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                               minLineLength=max(panel_width, panel_height) * 2, 
                               maxLineGap=50)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算线的角度和长度
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # 只考虑足够长的线
                if length > max(panel_width, panel_height) * 1.5:
                    # 分类水平线和垂直线
                    if abs(angle) < 10 or abs(angle) > 170:  # 水平线
                        y_pos = (y1 + y2) // 2
                        horizontal_lines.append(y_pos)
                    elif abs(abs(angle) - 90) < 10:  # 垂直线
                        x_pos = (x1 + x2) // 2
                        vertical_lines.append(x_pos)
        
        # 聚类相近的线 - 使用面板尺寸的一半作为容差
        tolerance = min(panel_width, panel_height) // 3
        horizontal_lines = self._cluster_lines(horizontal_lines, tolerance=tolerance)
        vertical_lines = self._cluster_lines(vertical_lines, tolerance=tolerance)
        
        # 如果检测到的网格线不足，尝试基于预期间距生成网格
        if len(horizontal_lines) < 3:
            horizontal_lines = self._generate_grid_lines(self.image_height, panel_height, horizontal_lines)
        if len(vertical_lines) < 3:
            vertical_lines = self._generate_grid_lines(self.image_width, panel_width, vertical_lines)
        
        print(f"检测到 {len(horizontal_lines)} 条水平线, {len(vertical_lines)} 条垂直线")
        
        if self.debug_dir:
            # 绘制检测到的线
            debug_img = self.rgb_image.copy()
            for y in horizontal_lines:
                cv2.line(debug_img, (0, y), (self.image_width, y), (0, 255, 0), 3)
            for x in vertical_lines:
                cv2.line(debug_img, (x, 0), (x, self.image_height), (255, 0, 0), 3)
            cv2.imwrite(os.path.join(self.debug_dir, "05_detected_grid_lines.jpg"), debug_img)
        
        return horizontal_lines, vertical_lines
    
    def _cluster_lines(self, lines: List[int], tolerance: int = 25) -> List[int]:
        """聚类相近的线条"""
        if not lines:
            return []
        
        lines_sorted = sorted(lines)
        clustered = []
        current_cluster = [lines_sorted[0]]
        
        for i in range(1, len(lines_sorted)):
            if lines_sorted[i] - current_cluster[-1] <= tolerance:
                current_cluster.append(lines_sorted[i])
            else:
                center = sum(current_cluster) // len(current_cluster)
                clustered.append(center)
                current_cluster = [lines_sorted[i]]
        
        if current_cluster:
            center = sum(current_cluster) // len(current_cluster)
            clustered.append(center)
        
        return clustered
    
    def _generate_grid_lines(self, image_dimension: int, panel_size: int, existing_lines: List[int]) -> List[int]:
        """基于面板尺寸生成网格线"""
        if not existing_lines:
            # 如果没有检测到线，基于面板尺寸生成均匀网格
            num_panels = image_dimension // panel_size
            lines = [i * panel_size for i in range(num_panels + 1)]
        else:
            # 基于现有线条扩展网格
            lines = existing_lines.copy()
            
            # 向前扩展
            first_line = min(existing_lines)
            pos = first_line - panel_size
            while pos > 0:
                lines.append(pos)
                pos -= panel_size
            
            # 向后扩展
            last_line = max(existing_lines)
            pos = last_line + panel_size
            while pos < image_dimension:
                lines.append(pos)
                pos += panel_size
            
            lines = sorted(list(set(lines)))
        
        return lines
    
    def segment_panels_with_fixed_size(self, blue_mask: np.ndarray, 
                                     horizontal_lines: List[int], 
                                     vertical_lines: List[int]) -> List[Dict[str, Any]]:
        """基于固定尺寸分割光伏板"""
        print("基于固定尺寸分割光伏板...")
        
        panels = []
        
        # 确保有边界线
        if 0 not in horizontal_lines:
            horizontal_lines.insert(0, 0)
        if self.image_height-1 not in horizontal_lines:
            horizontal_lines.append(self.image_height-1)
        if 0 not in vertical_lines:
            vertical_lines.insert(0, 0)
        if self.image_width-1 not in vertical_lines:
            vertical_lines.append(self.image_width-1)
        
        horizontal_lines.sort()
        vertical_lines.sort()
        
        print(f"网格分割: {len(horizontal_lines)-1} 行 x {len(vertical_lines)-1} 列")
        
        # 根据网格线创建面板
        panel_id = 1
        for r in range(len(horizontal_lines) - 1):
            for c in range(len(vertical_lines) - 1):
                y1, y2 = horizontal_lines[r], horizontal_lines[r + 1]
                x1, x2 = vertical_lines[c], vertical_lines[c + 1]
                
                # 检查面板区域的有效性
                width, height = x2 - x1, y2 - y1
                
                # 基于预期尺寸的严格检查
                if (self.expected_panel_width[0] <= width <= self.expected_panel_width[1] and
                    self.expected_panel_height[0] <= height <= self.expected_panel_height[1]):
                    
                    # 长宽比检查
                    aspect_ratio = width / height if height > 0 else 0
                    if self.expected_aspect_ratio[0] <= aspect_ratio <= self.expected_aspect_ratio[1]:
                        
                        # 检查该区域是否包含蓝色像素
                        roi_blue = blue_mask[y1:y2, x1:x2]
                        if roi_blue.size > 0:
                            blue_pixel_ratio = np.sum(roi_blue > 0) / roi_blue.size
                            
                            # 降低蓝色像素比例要求，因为可能有阴影或反射
                            if blue_pixel_ratio > 0.3:  # 30%的像素是蓝色
                                # 计算面板质量指标
                                blue_area = np.sum(roi_blue > 0)
                                total_area = width * height
                                
                                # 计算面板在图像中的位置
                                center_x = x1 + width // 2
                                center_y = y1 + height // 2
                                
                                panel = {
                                    "id": f"PV{panel_id:03d}",
                                    "grid_position": f"R{r+1}-C{c+1}",
                                    "row": r + 1,
                                    "column": c + 1,
                                    "position": {
                                        "x": int(x1),
                                        "y": int(y1),
                                        "width": int(width),
                                        "height": int(height),
                                        "center_x": int(center_x),
                                        "center_y": int(center_y)
                                    },
                                    "blue_pixel_ratio": float(blue_pixel_ratio),
                                    "blue_area": int(blue_area),
                                    "total_area": int(total_area),
                                    "aspect_ratio": float(aspect_ratio),
                                    "size_match_score": float(self._calculate_size_match_score(width, height)),
                                    "detection_method": "fixed_size_grid"
                                }
                                
                                panels.append(panel)
                                panel_id += 1
        
        return panels
    
    def _calculate_size_match_score(self, width: int, height: int) -> float:
        """计算尺寸匹配得分"""
        # 计算与预期尺寸的匹配程度
        width_center = (self.expected_panel_width[0] + self.expected_panel_width[1]) / 2
        height_center = (self.expected_panel_height[0] + self.expected_panel_height[1]) / 2
        
        width_score = 1.0 - abs(width - width_center) / width_center
        height_score = 1.0 - abs(height - height_center) / height_center
        
        return (width_score + height_score) / 2
    
    def visualize_results(self, panels: List[Dict[str, Any]]) -> np.ndarray:
        """可视化检测结果"""
        print(f"可视化 {len(panels)} 个光伏板的检测结果...")
        
        result_img = self.rgb_image.copy()
        
        # 绘制每个面板
        for panel in panels:
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 根据尺寸匹配得分使用不同颜色
            size_score = panel.get("size_match_score", 0)
            if size_score > 0.8:
                color = (0, 255, 0)  # 绿色 - 高匹配度
            elif size_score > 0.6:
                color = (0, 255, 255)  # 黄色 - 中等匹配度
            else:
                color = (0, 0, 255)  # 红色 - 低匹配度
            
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            
            # 显示面板ID
            cv2.putText(result_img, panel["id"], 
                       (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示尺寸信息
            size_text = f"{w}x{h}"
            cv2.putText(result_img, size_text, 
                       (x + 5, y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 显示匹配得分
            score_text = f"{size_score:.2f}"
            cv2.putText(result_img, score_text, 
                       (x + 5, y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 保存结果
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "06_final_result.jpg"), result_img)
        
        return result_img
    
    def save_results(self, panels: List[Dict[str, Any]]):
        """保存结果到JSON文件"""
        if self.debug_dir:
            # 计算统计信息
            if panels:
                sizes = [(p["position"]["width"], p["position"]["height"]) for p in panels]
                avg_width = np.mean([s[0] for s in sizes])
                avg_height = np.mean([s[1] for s in sizes])
                size_std_width = np.std([s[0] for s in sizes])
                size_std_height = np.std([s[1] for s in sizes])
                
                size_stats = {
                    "average_size": {"width": float(avg_width), "height": float(avg_height)},
                    "size_std": {"width": float(size_std_width), "height": float(size_std_height)},
                    "size_range": {
                        "width": {"min": min(s[0] for s in sizes), "max": max(s[0] for s in sizes)},
                        "height": {"min": min(s[1] for s in sizes), "max": max(s[1] for s in sizes)}
                    }
                }
            else:
                size_stats = {}
            
            result_data = {
                "method": "fixed_size_panel_detection",
                "total_panels": len(panels),
                "detection_time": datetime.now().isoformat(),
                "image_size": {
                    "width": self.image_width,
                    "height": self.image_height
                },
                "expected_panel_size": {
                    "width_range": self.expected_panel_width,
                    "height_range": self.expected_panel_height,
                    "aspect_ratio_range": self.expected_aspect_ratio
                },
                "size_statistics": size_stats,
                "panels": panels
            }
            
            json_path = os.path.join(self.debug_dir, "fixed_size_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"结果已保存到: {json_path}")
    
    def process_image(self, image_path: str):
        """处理图像的主函数"""
        print(f"开始基于固定尺寸的光伏板检测: {image_path}")
        
        # 加载图像
        self.load_image(image_path)
        
        # 提取蓝色光伏板区域
        blue_mask = self.extract_blue_panels()
        
        # 精确边缘检测
        edges = self.detect_precise_edges()
        
        # 估算面板尺寸
        panel_width, panel_height = self.estimate_panel_size(blue_mask)
        
        # 基于固定尺寸检测网格线
        horizontal_lines, vertical_lines = self.detect_grid_with_fixed_size(edges, panel_width, panel_height)
        
        # 基于固定尺寸分割面板
        panels = self.segment_panels_with_fixed_size(blue_mask, horizontal_lines, vertical_lines)
        
        # 可视化结果
        result_img = self.visualize_results(panels)
        
        # 保存结果
        self.save_results(panels)
        
        print(f"\n=== 固定尺寸检测结果 ===")
        print(f"检测到 {len(panels)} 个光伏板")
        print(f"预期面板尺寸: {self.expected_panel_width[0]}-{self.expected_panel_width[1]} x {self.expected_panel_height[0]}-{self.expected_panel_height[1]}")
        
        if panels:
            # 统计尺寸匹配情况
            high_quality = [p for p in panels if p.get("size_match_score", 0) > 0.8]
            medium_quality = [p for p in panels if 0.6 < p.get("size_match_score", 0) <= 0.8]
            low_quality = [p for p in panels if p.get("size_match_score", 0) <= 0.6]
            
            print(f"高质量匹配: {len(high_quality)} 个 ({len(high_quality)/len(panels)*100:.1f}%)")
            print(f"中等质量匹配: {len(medium_quality)} 个 ({len(medium_quality)/len(panels)*100:.1f}%)")
            print(f"低质量匹配: {len(low_quality)} 个 ({len(low_quality)/len(panels)*100:.1f}%)")
        
        if self.debug_dir:
            print(f"调试文件保存在: {self.debug_dir}")
        
        return panels, result_img


def main():
    parser = argparse.ArgumentParser(description='基于固定尺寸特征的光伏板检测')
    parser.add_argument('--image', '-i', type=str, required=True, help='RGB图像路径')
    args = parser.parse_args()
    
    # 创建检测器
    detector = FixedSizePanelDetector(debug_mode=True)
    
    # 处理图像
    panels, result_img = detector.process_image(args.image)
    
    return panels, result_img


if __name__ == "__main__":
    main() 