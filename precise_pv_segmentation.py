#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class PrecisePVSegmentation:
    """
    精确的光伏板分割算法
    专门针对光伏板的视觉特征进行优化
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
            self.debug_dir = f"precise_pv_debug_{base_name}_{timestamp}"
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # 保存原始图像
            cv2.imwrite(os.path.join(self.debug_dir, "01_original.jpg"), self.rgb_image)
        
        return self.rgb_image
    
    def extract_pv_array_region(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """精确提取光伏板阵列区域"""
        print("提取光伏板阵列区域...")
        
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        
        # 光伏板的深蓝色/黑色特征
        # 调整HSV范围以更精确地捕获光伏板
        lower_pv = np.array([90, 30, 10])   # 深蓝色下限
        upper_pv = np.array([130, 255, 100])  # 深蓝色上限
        
        # 创建光伏板掩码
        pv_mask = cv2.inRange(hsv, lower_pv, upper_pv)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        pv_mask = cv2.morphologyEx(pv_mask, cv2.MORPH_OPEN, kernel)
        pv_mask = cv2.morphologyEx(pv_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找最大的连通区域（主要的光伏板阵列）
        contours, _ = cv2.findContours(pv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("未找到光伏板区域")
            return pv_mask, (0, 0, self.image_width, self.image_height)
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 扩展边界框以确保包含完整的阵列
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(self.image_width - x, w + 2 * margin)
        h = min(self.image_height - y, h + 2 * margin)
        
        print(f"光伏板阵列区域: ({x}, {y}) - ({x+w}, {y+h})")
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "02_hsv.jpg"), hsv)
            cv2.imwrite(os.path.join(self.debug_dir, "03_pv_mask.jpg"), pv_mask)
            
            debug_img = self.rgb_image.copy()
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.imwrite(os.path.join(self.debug_dir, "04_array_region.jpg"), debug_img)
        
        return pv_mask, (x, y, w, h)
    
    def detect_panel_grid_lines(self, roi_region: Tuple[int, int, int, int]) -> Tuple[List[int], List[int]]:
        """在ROI区域内检测光伏板的网格线"""
        print("检测光伏板网格线...")
        
        x, y, w, h = roi_region
        roi_img = self.rgb_image[y:y+h, x:x+w]
        
        # 转换为灰度图
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 使用自适应阈值突出光伏板边界
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        
        # 反转图像，使光伏板区域为白色
        adaptive_thresh = 255 - adaptive_thresh
        
        # 形态学操作连接断开的线条
        kernel_h = np.ones((1, 15), np.uint8)  # 水平核
        kernel_v = np.ones((15, 1), np.uint8)  # 垂直核
        
        # 检测水平线
        horizontal_lines_img = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel_h)
        
        # 检测垂直线
        vertical_lines_img = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel_v)
        
        # 使用霍夫变换检测直线
        horizontal_lines = []
        vertical_lines = []
        
        # 检测水平线
        h_lines = cv2.HoughLinesP(horizontal_lines_img, 1, np.pi/180, threshold=50, 
                                 minLineLength=w//4, maxLineGap=20)
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                # 确保是水平线
                if abs(y2 - y1) < 10:
                    y_pos = (y1 + y2) // 2 + y  # 转换回全图坐标
                    horizontal_lines.append(y_pos)
        
        # 检测垂直线
        v_lines = cv2.HoughLinesP(vertical_lines_img, 1, np.pi/180, threshold=50, 
                                 minLineLength=h//4, maxLineGap=20)
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                # 确保是垂直线
                if abs(x2 - x1) < 10:
                    x_pos = (x1 + x2) // 2 + x  # 转换回全图坐标
                    vertical_lines.append(x_pos)
        
        # 去重和排序
        horizontal_lines = sorted(list(set(horizontal_lines)))
        vertical_lines = sorted(list(set(vertical_lines)))
        
        # 聚类相近的线条
        horizontal_lines = self._cluster_lines(horizontal_lines, tolerance=20)
        vertical_lines = self._cluster_lines(vertical_lines, tolerance=20)
        
        print(f"检测到 {len(horizontal_lines)} 条水平线, {len(vertical_lines)} 条垂直线")
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "05_roi_gray.jpg"), gray)
            cv2.imwrite(os.path.join(self.debug_dir, "06_adaptive_thresh.jpg"), adaptive_thresh)
            cv2.imwrite(os.path.join(self.debug_dir, "07_horizontal_lines.jpg"), horizontal_lines_img)
            cv2.imwrite(os.path.join(self.debug_dir, "08_vertical_lines.jpg"), vertical_lines_img)
            
            # 绘制检测到的线
            debug_img = self.rgb_image.copy()
            for y_pos in horizontal_lines:
                cv2.line(debug_img, (0, y_pos), (self.image_width, y_pos), (0, 255, 0), 2)
            for x_pos in vertical_lines:
                cv2.line(debug_img, (x_pos, 0), (x_pos, self.image_height), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(self.debug_dir, "09_detected_grid.jpg"), debug_img)
        
        return horizontal_lines, vertical_lines
    
    def _cluster_lines(self, lines: List[int], tolerance: int = 20) -> List[int]:
        """聚类相近的线条"""
        if not lines:
            return []
        
        clustered = []
        current_cluster = [lines[0]]
        
        for i in range(1, len(lines)):
            if lines[i] - current_cluster[-1] <= tolerance:
                current_cluster.append(lines[i])
            else:
                # 计算当前聚类的中心
                center = sum(current_cluster) // len(current_cluster)
                clustered.append(center)
                current_cluster = [lines[i]]
        
        # 添加最后一个聚类
        if current_cluster:
            center = sum(current_cluster) // len(current_cluster)
            clustered.append(center)
        
        return clustered
    
    def segment_individual_panels(self, pv_mask: np.ndarray, horizontal_lines: List[int], 
                                 vertical_lines: List[int]) -> List[Dict[str, Any]]:
        """基于网格线分割单个光伏板"""
        print("分割单个光伏板...")
        
        panels = []
        
        # 确保有足够的网格线
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            print("网格线不足，无法进行分割")
            return panels
        
        # 根据网格线创建面板
        panel_id = 1
        for r in range(len(horizontal_lines) - 1):
            for c in range(len(vertical_lines) - 1):
                y1, y2 = horizontal_lines[r], horizontal_lines[r + 1]
                x1, x2 = vertical_lines[c], vertical_lines[c + 1]
                
                # 检查面板区域的有效性
                width, height = x2 - x1, y2 - y1
                
                # 面板尺寸合理性检查
                if width < 30 or height < 30 or width > 500 or height > 500:
                    continue
                
                # 长宽比检查（光伏板通常接近正方形或矩形）
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    continue
                
                # 检查该区域是否包含足够的光伏板像素
                roi_mask = pv_mask[y1:y2, x1:x2]
                if roi_mask.size == 0:
                    continue
                
                panel_pixel_ratio = np.sum(roi_mask > 0) / roi_mask.size
                
                # 只有当区域包含足够多的光伏板像素时才认为是有效面板
                if panel_pixel_ratio > 0.3:  # 30%的像素是光伏板
                    panel = {
                        "id": f"PV{panel_id:03d}",
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
                        "aspect_ratio": aspect_ratio,
                        "area": width * height
                    }
                    
                    panels.append(panel)
                    panel_id += 1
        
        return panels
    
    def refine_panel_boundaries(self, panels: List[Dict[str, Any]], pv_mask: np.ndarray) -> List[Dict[str, Any]]:
        """精细化面板边界"""
        print("精细化面板边界...")
        
        refined_panels = []
        
        for panel in panels:
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 提取面板区域
            panel_roi = pv_mask[y:y+h, x:x+w]
            
            # 在面板区域内查找轮廓
            contours, _ = cv2.findContours(panel_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                
                # 如果轮廓面积足够大，更新面板位置
                if contour_area > (w * h * 0.2):  # 至少占20%的面积
                    cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                    
                    # 更新面板位置（转换回全图坐标）
                    panel["position"]["x"] = x + cx
                    panel["position"]["y"] = y + cy
                    panel["position"]["width"] = cw
                    panel["position"]["height"] = ch
                    panel["refined_area"] = contour_area
                    panel["aspect_ratio"] = cw / ch if ch > 0 else 0
            
            refined_panels.append(panel)
        
        return refined_panels
    
    def visualize_results(self, panels: List[Dict[str, Any]]) -> np.ndarray:
        """可视化分割结果"""
        print(f"可视化 {len(panels)} 个光伏板的分割结果...")
        
        # 创建结果图像
        result_img = self.rgb_image.copy()
        
        # 绘制每个面板
        for i, panel in enumerate(panels):
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 绘制边界框
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 显示面板ID
            cv2.putText(result_img, panel["id"], 
                       (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 显示像素比例
            ratio_text = f"{panel['panel_pixel_ratio']:.2f}"
            cv2.putText(result_img, ratio_text, 
                       (x + 5, y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 保存结果
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "10_final_result.jpg"), result_img)
        
        return result_img
    
    def save_results(self, panels: List[Dict[str, Any]]):
        """保存结果到JSON文件"""
        if self.debug_dir:
            result_data = {
                "method": "precise_pv_segmentation",
                "total_panels": len(panels),
                "detection_time": datetime.now().isoformat(),
                "image_size": {
                    "width": self.image_width,
                    "height": self.image_height
                },
                "panels": panels
            }
            
            json_path = os.path.join(self.debug_dir, "precise_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"结果已保存到: {json_path}")
    
    def process_image(self, image_path: str):
        """处理图像的主函数"""
        print(f"开始精确分割光伏板: {image_path}")
        
        # 加载图像
        self.load_image(image_path)
        
        # 提取光伏板阵列区域
        pv_mask, roi_region = self.extract_pv_array_region()
        
        # 检测网格线
        horizontal_lines, vertical_lines = self.detect_panel_grid_lines(roi_region)
        
        # 分割单个面板
        panels = self.segment_individual_panels(pv_mask, horizontal_lines, vertical_lines)
        
        # 精细化边界
        refined_panels = self.refine_panel_boundaries(panels, pv_mask)
        
        # 可视化结果
        result_img = self.visualize_results(refined_panels)
        
        # 保存结果
        self.save_results(refined_panels)
        
        print(f"\n=== 精确分割结果 ===")
        print(f"检测到 {len(refined_panels)} 个光伏板")
        
        if self.debug_dir:
            print(f"调试文件保存在: {self.debug_dir}")
        
        return refined_panels, result_img


def main():
    parser = argparse.ArgumentParser(description='精确的光伏板分割算法')
    parser.add_argument('--image', '-i', type=str, required=True, help='RGB图像路径')
    args = parser.parse_args()
    
    # 创建分割器
    segmenter = PrecisePVSegmentation(debug_mode=True)
    
    # 处理图像
    panels, result_img = segmenter.process_image(args.image)
    
    return panels, result_img


if __name__ == "__main__":
    main() 