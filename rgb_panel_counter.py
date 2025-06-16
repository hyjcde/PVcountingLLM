#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


class ImprovedRGBPanelCounter:
    """
    改进的RGB图像光伏板计数器
    专门针对高分辨率RGB图像进行精确分割
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
            self.debug_dir = f"improved_rgb_debug_{base_name}_{timestamp}"
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # 保存原始图像
            cv2.imwrite(os.path.join(self.debug_dir, "01_original.jpg"), self.rgb_image)
        
        return self.rgb_image
    
    def extract_solar_panel_regions(self) -> np.ndarray:
        """提取光伏板区域 - 多种方法结合"""
        print("提取光伏板区域...")
        
        # 转换到不同色彩空间
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2LAB)
        
        # 方法1: HSV颜色分割
        # 深蓝色范围（光伏板的主要颜色）
        lower_blue = np.array([100, 30, 20])
        upper_blue = np.array([140, 255, 150])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 深色范围
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # 方法2: LAB色彩空间分割
        # 在LAB空间中，光伏板通常有特定的a和b值
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 基于亮度的分割
        _, mask_brightness = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_brightness = 255 - mask_brightness  # 反转，因为光伏板较暗
        
        # 方法3: 基于纹理的分割
        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        
        # 使用Sobel算子检测边缘
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
        
        # 结合所有掩码
        combined_mask = cv2.bitwise_or(mask_blue, mask_dark)
        combined_mask = cv2.bitwise_or(combined_mask, mask_brightness)
        
        # 形态学操作
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        
        # 去除噪声
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        # 填充空洞
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large)
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "02_hsv.jpg"), hsv)
            cv2.imwrite(os.path.join(self.debug_dir, "03_mask_blue.jpg"), mask_blue)
            cv2.imwrite(os.path.join(self.debug_dir, "04_mask_dark.jpg"), mask_dark)
            cv2.imwrite(os.path.join(self.debug_dir, "05_mask_brightness.jpg"), mask_brightness)
            cv2.imwrite(os.path.join(self.debug_dir, "06_sobel.jpg"), sobel_magnitude)
            cv2.imwrite(os.path.join(self.debug_dir, "07_combined_mask.jpg"), combined_mask)
        
        return combined_mask
    
    def detect_panel_array_region(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """检测主要的光伏板阵列区域"""
        print("检测主要光伏板阵列区域...")
        
        # 查找最大的连通区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0, self.image_width, self.image_height
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 扩展边界框以确保包含完整的面板
        margin = 50
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(self.image_width - x, w + 2 * margin)
        h = min(self.image_height - y, h + 2 * margin)
        
        print(f"主要阵列区域: ({x}, {y}) - ({x+w}, {y+h})")
        
        if self.debug_dir:
            debug_img = self.rgb_image.copy()
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.imwrite(os.path.join(self.debug_dir, "08_array_region.jpg"), debug_img)
        
        return x, y, w, h
    
    def detect_grid_structure(self, roi_mask: np.ndarray, roi_x: int, roi_y: int) -> Tuple[List[int], List[int]]:
        """在ROI区域内检测网格结构"""
        print("检测网格结构...")
        
        # 在ROI区域内工作
        roi_rgb = self.rgb_image[roi_y:roi_y+roi_mask.shape[0], roi_x:roi_x+roi_mask.shape[1]]
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi_gray)
        
        # 多尺度边缘检测
        edges1 = cv2.Canny(enhanced, 30, 90)
        edges2 = cv2.Canny(enhanced, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # 使用概率霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=200, maxLineGap=50)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算线的长度和角度
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # 只考虑足够长的线
                if length > 150:
                    # 分类水平线和垂直线
                    if abs(angle) < 15 or abs(angle) > 165:  # 水平线
                        y_pos = (y1 + y2) // 2
                        horizontal_lines.append(y_pos)
                    elif abs(abs(angle) - 90) < 15:  # 垂直线
                        x_pos = (x1 + x2) // 2
                        vertical_lines.append(x_pos)
        
        # 使用DBSCAN聚类相似的线
        horizontal_lines = self._cluster_lines_dbscan(horizontal_lines, eps=30)
        vertical_lines = self._cluster_lines_dbscan(vertical_lines, eps=30)
        
        # 转换回全图坐标
        horizontal_lines = [y + roi_y for y in horizontal_lines]
        vertical_lines = [x + roi_x for x in vertical_lines]
        
        print(f"检测到 {len(horizontal_lines)} 条水平线, {len(vertical_lines)} 条垂直线")
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "09_roi_enhanced.jpg"), enhanced)
            cv2.imwrite(os.path.join(self.debug_dir, "10_roi_edges.jpg"), edges)
            
            # 绘制检测到的线
            debug_img = self.rgb_image.copy()
            for y in horizontal_lines:
                cv2.line(debug_img, (0, y), (self.image_width, y), (0, 255, 0), 3)
            for x in vertical_lines:
                cv2.line(debug_img, (x, 0), (x, self.image_height), (255, 0, 0), 3)
            cv2.imwrite(os.path.join(self.debug_dir, "11_detected_grid.jpg"), debug_img)
        
        return horizontal_lines, vertical_lines
    
    def _cluster_lines_dbscan(self, lines: List[int], eps: int = 30) -> List[int]:
        """使用DBSCAN聚类线条"""
        if len(lines) < 2:
            return lines
        
        # 转换为numpy数组
        lines_array = np.array(lines).reshape(-1, 1)
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=1).fit(lines_array)
        labels = clustering.labels_
        
        # 计算每个聚类的中心
        clustered_lines = []
        for label in set(labels):
            if label != -1:  # 忽略噪声点
                cluster_points = lines_array[labels == label]
                center = int(np.mean(cluster_points))
                clustered_lines.append(center)
        
        return sorted(clustered_lines)
    
    def segment_panels_by_grid(self, mask: np.ndarray, horizontal_lines: List[int], 
                              vertical_lines: List[int]) -> List[Dict[str, Any]]:
        """基于网格线分割光伏板"""
        print("基于网格线分割光伏板...")
        
        panels = []
        
        # 添加边界线
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
                
                # 检查这个区域是否包含足够的光伏板像素
                roi_mask = mask[y1:y2, x1:x2]
                if roi_mask.size == 0:
                    continue
                
                panel_pixel_ratio = np.sum(roi_mask > 0) / roi_mask.size
                
                # 检查面板尺寸是否合理
                width, height = x2 - x1, y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                # 只有当区域满足条件时才认为是有效面板
                if (panel_pixel_ratio > 0.1 and  # 10%的像素是光伏板
                    width > 50 and height > 50 and  # 最小尺寸
                    0.3 < aspect_ratio < 3.0):  # 合理的长宽比
                    
                    panel = {
                        "id": f"P{panel_id:03d}",
                        "grid_position": f"R{r+1}-C{c+1}",
                        "row": r + 1,
                        "column": c + 1,
                        "position": {
                            "x": x1,
                            "y": y1,
                            "width": width,
                            "height": height
                        },
                        "panel_pixel_ratio": panel_pixel_ratio,
                        "aspect_ratio": aspect_ratio
                    }
                    
                    panels.append(panel)
                    panel_id += 1
        
        return panels
    
    def refine_panel_detection(self, panels: List[Dict[str, Any]], mask: np.ndarray) -> List[Dict[str, Any]]:
        """精细化面板检测"""
        print("精细化面板检测...")
        
        refined_panels = []
        
        for panel in panels:
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 提取面板区域
            panel_roi = mask[y:y+h, x:x+w]
            panel_rgb = self.rgb_image[y:y+h, x:x+w]
            
            # 在面板区域内查找轮廓
            contours, _ = cv2.findContours(panel_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                
                # 如果轮廓面积足够大，更新面板位置
                if contour_area > (w * h * 0.1):  # 至少占10%的面积
                    cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                    
                    # 更新面板位置（转换回全图坐标）
                    panel["position"]["x"] = x + cx
                    panel["position"]["y"] = y + cy
                    panel["position"]["width"] = cw
                    panel["position"]["height"] = ch
                    panel["refined_area"] = contour_area
            
            refined_panels.append(panel)
        
        return refined_panels
    
    def visualize_results(self, panels: List[Dict[str, Any]], method_name: str):
        """可视化分割结果"""
        debug_img = self.rgb_image.copy()
        
        # 绘制面板边界框
        for panel in panels:
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 绘制矩形
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 显示面板ID
            cv2.putText(debug_img, panel["id"], 
                       (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 显示网格位置
            if "grid_position" in panel:
                cv2.putText(debug_img, panel["grid_position"], 
                           (x + 5, y + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, f"12_{method_name}_result.jpg"), debug_img)
        
        print(f"{method_name}方法检测到 {len(panels)} 个光伏板")
        return debug_img
    
    def save_results(self, panels: List[Dict[str, Any]], method_name: str):
        """保存结果到JSON文件"""
        if self.debug_dir:
            result_data = {
                "method": method_name,
                "total_panels": len(panels),
                "detection_time": datetime.now().isoformat(),
                "image_size": {
                    "width": self.image_width,
                    "height": self.image_height
                },
                "panels": panels
            }
            
            json_path = os.path.join(self.debug_dir, f"{method_name}_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"结果已保存到: {json_path}")
    
    def process_image(self, image_path: str):
        """处理图像的主函数"""
        print(f"处理图像: {image_path}")
        
        # 加载图像
        self.load_image(image_path)
        
        # 提取光伏板区域
        panel_mask = self.extract_solar_panel_regions()
        
        # 检测主要阵列区域
        roi_x, roi_y, roi_w, roi_h = self.detect_panel_array_region(panel_mask)
        roi_mask = panel_mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # 检测网格结构
        horizontal_lines, vertical_lines = self.detect_grid_structure(roi_mask, roi_x, roi_y)
        
        # 基于网格分割面板
        if len(horizontal_lines) >= 3 and len(vertical_lines) >= 5:
            panels = self.segment_panels_by_grid(panel_mask, horizontal_lines, vertical_lines)
            
            # 精细化检测
            refined_panels = self.refine_panel_detection(panels, panel_mask)
            
            # 可视化结果
            self.visualize_results(refined_panels, "improved_grid")
            
            # 保存结果
            self.save_results(refined_panels, "improved_grid")
            
            print(f"\n=== 最终结果 ===")
            print(f"检测到 {len(refined_panels)} 个光伏板")
            
            if self.debug_dir:
                print(f"调试文件保存在: {self.debug_dir}")
            
            return refined_panels
        else:
            print("网格线检测不足，无法进行精确分割")
            return []


def main():
    parser = argparse.ArgumentParser(description='改进的RGB图像光伏板计数器')
    parser.add_argument('--image', '-i', type=str, required=True, help='RGB图像路径')
    args = parser.parse_args()
    
    # 创建计数器
    counter = ImprovedRGBPanelCounter(debug_mode=True)
    
    # 处理图像
    panels = counter.process_image(args.image)
    
    return panels


if __name__ == "__main__":
    main() 