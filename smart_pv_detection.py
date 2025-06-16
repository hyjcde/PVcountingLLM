#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class SmartPVDetection:
    """
    智能光伏板检测算法
    基于实际光伏板的视觉特征进行精确检测
    """
    
    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        self.rgb_image = None
        self.debug_dir = None
        self.image_height = 0
        self.image_width = 0
    
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
            self.debug_dir = f"smart_pv_debug_{base_name}_{timestamp}"
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # 保存原始图像
            cv2.imwrite(os.path.join(self.debug_dir, "01_original.jpg"), self.rgb_image)
        
        return self.rgb_image
    
    def detect_pv_array_region(self) -> Tuple[int, int, int, int]:
        """检测主要的光伏板阵列区域"""
        print("检测光伏板阵列区域...")
        
        # 转换到不同色彩空间进行分析
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        
        # 方法1: 基于HSV的光伏板检测（深蓝色）
        # 扩大HSV范围以捕获更多光伏板像素
        lower_blue1 = np.array([100, 50, 20])
        upper_blue1 = np.array([130, 255, 120])
        mask_blue = cv2.inRange(hsv, lower_blue1, upper_blue1)
        
        # 方法2: 基于亮度的检测（光伏板通常较暗）
        _, mask_dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # 方法3: 基于LAB色彩空间的检测
        l_channel, a_channel, b_channel = cv2.split(lab)
        _, mask_lab = cv2.threshold(l_channel, 100, 255, cv2.THRESH_BINARY_INV)
        
        # 结合多种掩码
        combined_mask = cv2.bitwise_or(mask_blue, mask_dark)
        combined_mask = cv2.bitwise_or(combined_mask, mask_lab)
        
        # 形态学操作
        kernel = np.ones((10, 10), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找最大的连通区域
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("未找到光伏板区域，使用整个图像")
            return 0, 0, self.image_width, self.image_height
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 扩展边界
        margin = 50
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(self.image_width - x, w + 2 * margin)
        h = min(self.image_height - y, h + 2 * margin)
        
        print(f"光伏板阵列区域: ({x}, {y}) - ({x+w}, {y+h})")
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "02_hsv.jpg"), hsv)
            cv2.imwrite(os.path.join(self.debug_dir, "03_mask_blue.jpg"), mask_blue)
            cv2.imwrite(os.path.join(self.debug_dir, "04_mask_dark.jpg"), mask_dark)
            cv2.imwrite(os.path.join(self.debug_dir, "05_mask_lab.jpg"), mask_lab)
            cv2.imwrite(os.path.join(self.debug_dir, "06_combined_mask.jpg"), combined_mask)
            
            debug_img = self.rgb_image.copy()
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.imwrite(os.path.join(self.debug_dir, "07_array_region.jpg"), debug_img)
        
        return x, y, w, h
    
    def detect_individual_panels_by_template(self, roi_region: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """使用模板匹配和轮廓检测来识别单个光伏板"""
        print("使用轮廓检测识别单个光伏板...")
        
        x, y, w, h = roi_region
        roi_img = self.rgb_image[y:y+h, x:x+w]
        
        # 转换为灰度图
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # 使用多种阈值方法
        # 方法1: OTSU阈值
        _, thresh_otsu = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 方法2: 自适应阈值
        thresh_adaptive = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 15, 5)
        
        # 结合两种阈值
        combined_thresh = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        panels = []
        panel_id = 1
        
        # 分析每个轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面积过滤（基于图像尺寸调整）
            min_area = 1000   # 最小面积
            max_area = 50000  # 最大面积
            
            if min_area < area < max_area:
                # 获取边界框
                cx, cy, cw, ch = cv2.boundingRect(contour)
                
                # 长宽比检查
                aspect_ratio = cw / ch if ch > 0 else 0
                if 0.5 < aspect_ratio < 2.0:  # 光伏板通常接近正方形
                    
                    # 转换回全图坐标
                    global_x = x + cx
                    global_y = y + cy
                    
                    # 检查轮廓的矩形度（轮廓面积与边界框面积的比值）
                    rect_area = cw * ch
                    rectangularity = area / rect_area if rect_area > 0 else 0
                    
                    if rectangularity > 0.6:  # 轮廓应该相对矩形
                        panel = {
                            "id": f"PV{panel_id:03d}",
                            "position": {
                                "x": global_x,
                                "y": global_y,
                                "width": cw,
                                "height": ch
                            },
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                            "rectangularity": rectangularity,
                            "contour_area": area
                        }
                        
                        panels.append(panel)
                        panel_id += 1
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "08_roi_gray.jpg"), gray_roi)
            cv2.imwrite(os.path.join(self.debug_dir, "09_thresh_otsu.jpg"), thresh_otsu)
            cv2.imwrite(os.path.join(self.debug_dir, "10_thresh_adaptive.jpg"), thresh_adaptive)
            cv2.imwrite(os.path.join(self.debug_dir, "11_combined_thresh.jpg"), combined_thresh)
            
            # 绘制检测到的轮廓
            contour_img = roi_img.copy()
            for i, panel in enumerate(panels):
                pos = panel["position"]
                # 转换回ROI坐标
                roi_x = pos["x"] - x
                roi_y = pos["y"] - y
                cv2.rectangle(contour_img, (roi_x, roi_y), 
                             (roi_x + pos["width"], roi_y + pos["height"]), 
                             (0, 255, 0), 2)
                cv2.putText(contour_img, panel["id"], 
                           (roi_x + 5, roi_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imwrite(os.path.join(self.debug_dir, "12_detected_contours.jpg"), contour_img)
        
        return panels
    
    def detect_panels_by_grid_analysis(self, roi_region: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """通过网格分析检测光伏板"""
        print("通过网格分析检测光伏板...")
        
        x, y, w, h = roi_region
        roi_img = self.rgb_image[y:y+h, x:x+w]
        
        # 转换为灰度图
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray_roi, 50, 150, apertureSize=3)
        
        # 使用霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for rho, theta in lines[:, 0]:
                # 分类水平线和垂直线
                angle = theta * 180 / np.pi
                
                if abs(angle) < 10 or abs(angle - 180) < 10:  # 水平线
                    y_pos = int(rho / np.sin(theta)) if np.sin(theta) != 0 else 0
                    if 0 <= y_pos < h:
                        horizontal_lines.append(y_pos + y)
                elif abs(angle - 90) < 10:  # 垂直线
                    x_pos = int(rho / np.cos(theta)) if np.cos(theta) != 0 else 0
                    if 0 <= x_pos < w:
                        vertical_lines.append(x_pos + x)
        
        # 去重和排序
        horizontal_lines = sorted(list(set(horizontal_lines)))
        vertical_lines = sorted(list(set(vertical_lines)))
        
        # 聚类相近的线
        horizontal_lines = self._cluster_lines(horizontal_lines, tolerance=30)
        vertical_lines = self._cluster_lines(vertical_lines, tolerance=30)
        
        print(f"检测到 {len(horizontal_lines)} 条水平线, {len(vertical_lines)} 条垂直线")
        
        panels = []
        
        # 如果检测到足够的网格线，创建网格面板
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            panel_id = 1
            for r in range(len(horizontal_lines) - 1):
                for c in range(len(vertical_lines) - 1):
                    y1, y2 = horizontal_lines[r], horizontal_lines[r + 1]
                    x1, x2 = vertical_lines[c], vertical_lines[c + 1]
                    
                    width, height = x2 - x1, y2 - y1
                    
                    # 尺寸检查
                    if 50 < width < 300 and 50 < height < 300:
                        aspect_ratio = width / height if height > 0 else 0
                        if 0.5 < aspect_ratio < 2.0:
                            panel = {
                                "id": f"GRID{panel_id:03d}",
                                "grid_position": f"R{r+1}-C{c+1}",
                                "position": {
                                    "x": x1,
                                    "y": y1,
                                    "width": width,
                                    "height": height
                                },
                                "area": width * height,
                                "aspect_ratio": aspect_ratio,
                                "detection_method": "grid"
                            }
                            
                            panels.append(panel)
                            panel_id += 1
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "13_edges.jpg"), edges)
            
            # 绘制检测到的线
            line_img = roi_img.copy()
            for y_pos in horizontal_lines:
                cv2.line(line_img, (0, y_pos - y), (w, y_pos - y), (0, 255, 0), 2)
            for x_pos in vertical_lines:
                cv2.line(line_img, (x_pos - x, 0), (x_pos - x, h), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(self.debug_dir, "14_grid_lines.jpg"), line_img)
        
        return panels
    
    def _cluster_lines(self, lines: List[int], tolerance: int = 30) -> List[int]:
        """聚类相近的线条"""
        if not lines:
            return []
        
        clustered = []
        current_cluster = [lines[0]]
        
        for i in range(1, len(lines)):
            if lines[i] - current_cluster[-1] <= tolerance:
                current_cluster.append(lines[i])
            else:
                center = sum(current_cluster) // len(current_cluster)
                clustered.append(center)
                current_cluster = [lines[i]]
        
        if current_cluster:
            center = sum(current_cluster) // len(current_cluster)
            clustered.append(center)
        
        return clustered
    
    def merge_detection_results(self, contour_panels: List[Dict[str, Any]], 
                               grid_panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并不同检测方法的结果"""
        print("合并检测结果...")
        
        all_panels = []
        
        # 添加轮廓检测的结果
        for panel in contour_panels:
            panel["detection_method"] = "contour"
            all_panels.append(panel)
        
        # 添加网格检测的结果（避免重复）
        for grid_panel in grid_panels:
            # 检查是否与现有面板重叠
            is_duplicate = False
            for existing_panel in all_panels:
                if self._panels_overlap(grid_panel, existing_panel, threshold=0.5):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_panels.append(grid_panel)
        
        # 按位置排序
        all_panels.sort(key=lambda p: (p["position"]["y"], p["position"]["x"]))
        
        # 重新编号
        for i, panel in enumerate(all_panels):
            panel["id"] = f"PV{i+1:03d}"
        
        return all_panels
    
    def _panels_overlap(self, panel1: Dict[str, Any], panel2: Dict[str, Any], threshold: float = 0.5) -> bool:
        """检查两个面板是否重叠"""
        pos1 = panel1["position"]
        pos2 = panel2["position"]
        
        # 计算重叠区域
        x1 = max(pos1["x"], pos2["x"])
        y1 = max(pos1["y"], pos2["y"])
        x2 = min(pos1["x"] + pos1["width"], pos2["x"] + pos2["width"])
        y2 = min(pos1["y"] + pos1["height"], pos2["y"] + pos2["height"])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        overlap_area = (x2 - x1) * (y2 - y1)
        area1 = pos1["width"] * pos1["height"]
        area2 = pos2["width"] * pos2["height"]
        
        overlap_ratio = overlap_area / min(area1, area2)
        return overlap_ratio > threshold
    
    def visualize_results(self, panels: List[Dict[str, Any]]) -> np.ndarray:
        """可视化检测结果"""
        print(f"可视化 {len(panels)} 个光伏板的检测结果...")
        
        result_img = self.rgb_image.copy()
        
        # 绘制每个面板
        for panel in panels:
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 根据检测方法使用不同颜色
            if panel.get("detection_method") == "contour":
                color = (0, 255, 0)  # 绿色 - 轮廓检测
            else:
                color = (255, 0, 0)  # 蓝色 - 网格检测
            
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
            
            # 显示面板ID
            cv2.putText(result_img, panel["id"], 
                       (x + 5, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 显示检测方法
            method_text = panel.get("detection_method", "unknown")
            cv2.putText(result_img, method_text, 
                       (x + 5, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 保存结果
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "15_final_result.jpg"), result_img)
        
        return result_img
    
    def save_results(self, panels: List[Dict[str, Any]]):
        """保存结果到JSON文件"""
        if self.debug_dir:
            result_data = {
                "method": "smart_pv_detection",
                "total_panels": len(panels),
                "detection_time": datetime.now().isoformat(),
                "image_size": {
                    "width": self.image_width,
                    "height": self.image_height
                },
                "panels": panels
            }
            
            json_path = os.path.join(self.debug_dir, "smart_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"结果已保存到: {json_path}")
    
    def process_image(self, image_path: str):
        """处理图像的主函数"""
        print(f"开始智能光伏板检测: {image_path}")
        
        # 加载图像
        self.load_image(image_path)
        
        # 检测光伏板阵列区域
        roi_region = self.detect_pv_array_region()
        
        # 使用轮廓检测识别单个面板
        contour_panels = self.detect_individual_panels_by_template(roi_region)
        
        # 使用网格分析检测面板
        grid_panels = self.detect_panels_by_grid_analysis(roi_region)
        
        # 合并检测结果
        all_panels = self.merge_detection_results(contour_panels, grid_panels)
        
        # 可视化结果
        result_img = self.visualize_results(all_panels)
        
        # 保存结果
        self.save_results(all_panels)
        
        print(f"\n=== 智能检测结果 ===")
        print(f"轮廓检测: {len(contour_panels)} 个面板")
        print(f"网格检测: {len(grid_panels)} 个面板")
        print(f"合并后总计: {len(all_panels)} 个面板")
        
        if self.debug_dir:
            print(f"调试文件保存在: {self.debug_dir}")
        
        return all_panels, result_img


def main():
    parser = argparse.ArgumentParser(description='智能光伏板检测算法')
    parser.add_argument('--image', '-i', type=str, required=True, help='RGB图像路径')
    args = parser.parse_args()
    
    # 创建检测器
    detector = SmartPVDetection(debug_mode=True)
    
    # 处理图像
    panels, result_img = detector.process_image(args.image)
    
    return panels, result_img


if __name__ == "__main__":
    main() 