#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class RGBPanelSegmenter:
    """
    专门用于RGB图像光伏板分割的类
    """
    
    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        self.rgb_image = None
        self.debug_dir = None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """加载RGB图像"""
        self.rgb_image = cv2.imread(image_path)
        if self.rgb_image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        # 创建调试目录
        if self.debug_mode:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            self.debug_dir = f"rgb_debug_{base_name}_{timestamp}"
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # 保存原始图像
            cv2.imwrite(os.path.join(self.debug_dir, "01_original.jpg"), self.rgb_image)
        
        return self.rgb_image
    
    def extract_pv_panels_by_color(self) -> np.ndarray:
        """基于颜色特征提取光伏板区域"""
        print("提取光伏板颜色区域...")
        
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        
        # 定义光伏板的颜色范围
        # 深蓝色范围（光伏板的主要颜色）
        lower_blue = np.array([100, 50, 20])
        upper_blue = np.array([130, 255, 120])
        
        # 深色/黑色范围
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 60])
        
        # 创建颜色掩码
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # 合并掩码
        color_mask = cv2.bitwise_or(mask_blue, mask_dark)
        
        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "02_hsv.jpg"), hsv)
            cv2.imwrite(os.path.join(self.debug_dir, "03_mask_blue.jpg"), mask_blue)
            cv2.imwrite(os.path.join(self.debug_dir, "04_mask_dark.jpg"), mask_dark)
            cv2.imwrite(os.path.join(self.debug_dir, "05_color_mask.jpg"), color_mask)
        
        return color_mask
    
    def detect_grid_lines(self) -> Tuple[List[int], List[int]]:
        """检测光伏板阵列的网格线"""
        print("检测网格线...")
        
        # 转换为灰度图
        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 边缘检测
        edges = cv2.Canny(enhanced, 30, 100)
        
        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, 
                               minLineLength=300, maxLineGap=30)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算线的角度
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # 分类水平线和垂直线
                if abs(angle) < 10 or abs(angle) > 170:  # 水平线
                    y_pos = (y1 + y2) // 2
                    horizontal_lines.append(y_pos)
                elif abs(abs(angle) - 90) < 10:  # 垂直线
                    x_pos = (x1 + x2) // 2
                    vertical_lines.append(x_pos)
        
        # 聚类相似的线
        horizontal_lines = self._cluster_coordinates(horizontal_lines, tolerance=30)
        vertical_lines = self._cluster_coordinates(vertical_lines, tolerance=30)
        
        print(f"检测到 {len(horizontal_lines)} 条水平线, {len(vertical_lines)} 条垂直线")
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "06_gray.jpg"), gray)
            cv2.imwrite(os.path.join(self.debug_dir, "07_enhanced.jpg"), enhanced)
            cv2.imwrite(os.path.join(self.debug_dir, "08_edges.jpg"), edges)
            
            # 绘制检测到的线
            debug_img = self.rgb_image.copy()
            for y in horizontal_lines:
                cv2.line(debug_img, (0, y), (debug_img.shape[1], y), (0, 255, 0), 2)
            for x in vertical_lines:
                cv2.line(debug_img, (x, 0), (x, debug_img.shape[0]), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(self.debug_dir, "09_detected_lines.jpg"), debug_img)
        
        return horizontal_lines, vertical_lines
    
    def _cluster_coordinates(self, coords: List[int], tolerance: int = 30) -> List[int]:
        """聚类坐标"""
        if not coords:
            return []
        
        coords_sorted = sorted(coords)
        clusters = []
        current_cluster = [coords_sorted[0]]
        
        for coord in coords_sorted[1:]:
            if coord - current_cluster[-1] <= tolerance:
                current_cluster.append(coord)
            else:
                cluster_center = sum(current_cluster) // len(current_cluster)
                clusters.append(cluster_center)
                current_cluster = [coord]
        
        if current_cluster:
            cluster_center = sum(current_cluster) // len(current_cluster)
            clusters.append(cluster_center)
        
        return clusters
    
    def segment_by_grid_lines(self, color_mask: np.ndarray, 
                             horizontal_lines: List[int], vertical_lines: List[int]) -> List[Dict[str, Any]]:
        """基于网格线分割光伏板"""
        print("基于网格线分割光伏板...")
        
        panels = []
        h, w = color_mask.shape
        
        # 添加边界线
        if 0 not in horizontal_lines:
            horizontal_lines.insert(0, 0)
        if h-1 not in horizontal_lines:
            horizontal_lines.append(h-1)
        if 0 not in vertical_lines:
            vertical_lines.insert(0, 0)
        if w-1 not in vertical_lines:
            vertical_lines.append(w-1)
        
        horizontal_lines.sort()
        vertical_lines.sort()
        
        print(f"网格分割: {len(horizontal_lines)-1} 行 x {len(vertical_lines)-1} 列")
        
        # 根据网格线创建面板
        for r in range(len(horizontal_lines) - 1):
            for c in range(len(vertical_lines) - 1):
                y1, y2 = horizontal_lines[r], horizontal_lines[r + 1]
                x1, x2 = vertical_lines[c], vertical_lines[c + 1]
                
                # 检查这个区域是否包含足够的光伏板像素
                roi_mask = color_mask[y1:y2, x1:x2]
                if roi_mask.size == 0:
                    continue
                    
                panel_pixel_ratio = np.sum(roi_mask > 0) / roi_mask.size
                
                # 只有当区域包含足够多的光伏板像素时才认为是有效面板
                if panel_pixel_ratio > 0.15:  # 15%的像素是光伏板
                    panel = {
                        "id": f"R{r+1}-C{c+1}",
                        "row": r + 1,
                        "column": c + 1,
                        "position": {
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1
                        },
                        "panel_pixel_ratio": panel_pixel_ratio
                    }
                    
                    panels.append(panel)
        
        return panels
    
    def segment_by_contours(self, color_mask: np.ndarray) -> List[Dict[str, Any]]:
        """基于轮廓分割光伏板"""
        print("基于轮廓分割光伏板...")
        
        # 查找轮廓
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        panels = []
        valid_contours = []
        
        # 过滤轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 根据RGB图像的高分辨率调整面积阈值
            min_area = 3000   # 最小面积
            max_area = 100000 # 最大面积
            
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 检查长宽比是否合理
                if 0.3 < aspect_ratio < 3.0:
                    valid_contours.append((contour, x, y, w, h, area))
        
        # 按面积排序
        valid_contours.sort(key=lambda x: x[5], reverse=True)
        
        # 限制轮廓数量
        max_panels = 300
        valid_contours = valid_contours[:max_panels]
        
        # 按位置排序（从上到下，从左到右）
        valid_contours.sort(key=lambda x: (x[2], x[1]))
        
        # 创建面板信息
        for i, (contour, x, y, w, h, area) in enumerate(valid_contours):
            panel = {
                "id": f"P{i+1:03d}",
                "row": i // 25 + 1,  # 假设每行最多25个面板
                "column": i % 25 + 1,
                "position": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "area": area,
                "aspect_ratio": w / h if h > 0 else 0
            }
            
            panels.append(panel)
        
        return panels
    
    def visualize_results(self, panels: List[Dict[str, Any]], method_name: str):
        """可视化分割结果"""
        debug_img = self.rgb_image.copy()
        
        for panel in panels:
            pos = panel["position"]
            cv2.rectangle(debug_img, (pos["x"], pos["y"]), 
                         (pos["x"] + pos["width"], pos["y"] + pos["height"]), 
                         (0, 255, 0), 2)
            
            # 显示面板ID
            cv2.putText(debug_img, panel["id"], 
                       (pos["x"] + 5, pos["y"] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # 显示额外信息
            if "panel_pixel_ratio" in panel:
                cv2.putText(debug_img, f"{panel['panel_pixel_ratio']:.2f}", 
                           (pos["x"] + 5, pos["y"] + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            elif "area" in panel:
                cv2.putText(debug_img, f"A:{int(panel['area'])}", 
                           (pos["x"] + 5, pos["y"] + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, f"10_{method_name}_result.jpg"), debug_img)
        
        print(f"{method_name}方法检测到 {len(panels)} 个光伏板")
        return debug_img
    
    def process_image(self, image_path: str):
        """处理图像的主函数"""
        print(f"处理图像: {image_path}")
        
        # 加载图像
        self.load_image(image_path)
        
        # 提取光伏板颜色区域
        color_mask = self.extract_pv_panels_by_color()
        
        # 检测网格线
        horizontal_lines, vertical_lines = self.detect_grid_lines()
        
        # 尝试两种分割方法
        results = {}
        
        # 方法1：基于网格线分割
        if len(horizontal_lines) >= 3 and len(vertical_lines) >= 5:
            grid_panels = self.segment_by_grid_lines(color_mask, horizontal_lines, vertical_lines)
            grid_img = self.visualize_results(grid_panels, "grid")
            results["grid"] = {"panels": grid_panels, "image": grid_img}
        else:
            print("网格线检测不足，跳过网格分割方法")
        
        # 方法2：基于轮廓分割
        contour_panels = self.segment_by_contours(color_mask)
        contour_img = self.visualize_results(contour_panels, "contour")
        results["contour"] = {"panels": contour_panels, "image": contour_img}
        
        # 输出总结
        print("\n=== 分割结果总结 ===")
        for method, result in results.items():
            print(f"{method}方法: {len(result['panels'])} 个面板")
        
        if self.debug_dir:
            print(f"调试文件保存在: {self.debug_dir}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='RGB图像光伏板分割测试')
    parser.add_argument('--image', '-i', type=str, required=True, help='RGB图像路径')
    args = parser.parse_args()
    
    # 创建分割器
    segmenter = RGBPanelSegmenter(debug_mode=True)
    
    # 处理图像
    results = segmenter.process_image(args.image)
    
    return results


if __name__ == "__main__":
    main() 