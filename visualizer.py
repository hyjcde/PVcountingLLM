import json
import os
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np


class Visualizer:
    """
    结果可视化类
    负责生成带标注的输出图像和JSON结果
    """
    
    def __init__(self, debug_dir: str = None):
        """
        初始化可视化器
        
        Args:
            debug_dir: 调试输出目录，如果为None则不输出中间结果
        """
        # 状态对应的颜色 (BGR格式)
        self.status_colors = {
            "Normal": (0, 255, 0),    # 绿色
            "Warning": (0, 165, 255),  # 橙色
            "Critical": (0, 0, 255)    # 红色
        }
        self.debug_dir = debug_dir
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
    
    def create_annotated_image(self, original_image: np.ndarray, panels: List[Dict[str, Any]],
                              array_corners: np.ndarray = None) -> np.ndarray:
        """
        在原始图像上标注光伏板信息
        
        Args:
            original_image: 原始图像
            panels: 包含光伏板信息的字典列表
            array_corners: 光伏阵列的四个角点坐标
            
        Returns:
            带标注的图像
        """
        # 创建图像副本
        annotated_image = original_image.copy()
        
        # 如果提供了阵列角点，绘制阵列轮廓
        if array_corners is not None:
            # 确保角点是整数类型的numpy数组
            corners_int = np.array(array_corners, dtype=np.int32)
            cv2.polylines(annotated_image, [corners_int], True, (255, 0, 0), 3)
            
            # 输出调试信息
            if self.debug_dir:
                corners_img = original_image.copy()
                cv2.polylines(corners_img, [corners_int], True, (255, 0, 0), 3)
                for i, point in enumerate(corners_int):
                    cv2.circle(corners_img, tuple(point), 10, (0, 0, 255), -1)
                    cv2.putText(corners_img, str(i), (point[0] + 10, point[1] + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(self.debug_dir, "detected_corners.jpg"), corners_img)
        
        # 创建半透明叠加层
        overlay = annotated_image.copy()
        
        # 遍历所有面板
        for panel in panels:
            # 只有在面板有状态信息时才进行标注
            if "status" not in panel:
                continue
            
            status = panel["status"]
            panel_id = panel["id"]
            
            # 获取面板在原始图像中的位置
            if "original_position" in panel:
                pos = panel["original_position"]
                x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
                
                # 根据状态选择颜色
                color = self.status_colors.get(status, (0, 255, 0))
                
                # 绘制矩形和ID
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                cv2.putText(annotated_image, f"{panel_id}: {status}", (x + 5, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 应用半透明效果
        alpha = 0.3  # 透明度
        cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
        
        # 保存中间结果
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "annotated_panels.jpg"), annotated_image)
        
        return annotated_image
    
    def map_panels_to_original(self, panels: List[Dict[str, Any]], perspective_matrix: np.ndarray,
                             original_shape: tuple) -> List[Dict[str, Any]]:
        """
        将校正图像中的面板位置映射回原始图像
        
        Args:
            panels: 包含光伏板信息的字典列表
            perspective_matrix: 透视变换矩阵
            original_shape: 原始图像的形状
            
        Returns:
            更新了original_position的面板列表
        """
        # 透视变换的逆矩阵
        inv_matrix = np.linalg.inv(perspective_matrix)
        
        # 创建可视化图像（如果启用了调试）
        if self.debug_dir:
            debug_img = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)
        
        for panel in panels:
            pos = panel["position"]
            
            # 定义校正图像中的四个角点
            warped_points = np.array([
                [pos["x"], pos["y"]],
                [pos["x"] + pos["width"], pos["y"]],
                [pos["x"] + pos["width"], pos["y"] + pos["height"]],
                [pos["x"], pos["y"] + pos["height"]]
            ], dtype=np.float32)
            
            # 将点映射回原始图像
            original_points = []
            for point in warped_points:
                # 应用逆透视变换
                px, py, pw = np.dot(inv_matrix, [point[0], point[1], 1])
                original_points.append([int(px / pw), int(py / pw)])
            
            # 计算原始图像中的边界框
            x_coords = [p[0] for p in original_points]
            y_coords = [p[1] for p in original_points]
            
            x = max(0, min(x_coords))
            y = max(0, min(y_coords))
            w = min(original_shape[1] - x, max(x_coords) - x)
            h = min(original_shape[0] - y, max(y_coords) - y)
            
            # 添加到面板信息中
            panel["original_position"] = {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            }
            
            # 添加多边形点
            panel["original_polygon"] = original_points
            
            # 在调试图像上绘制映射后的面板
            if self.debug_dir:
                # 绘制多边形
                pts = np.array(original_points, dtype=np.int32)
                cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
                # 绘制ID
                center_x = sum(x_coords) // len(x_coords)
                center_y = sum(y_coords) // len(y_coords)
                cv2.putText(debug_img, panel["id"], (center_x, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 保存调试图像
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "mapped_panels.jpg"), debug_img)
        
        return panels
    
    def generate_json_output(self, panels: List[Dict[str, Any]], array_properties: Dict[str, Any],
                           image_path: str) -> Dict[str, Any]:
        """
        生成JSON格式的输出结果
        
        Args:
            panels: 包含光伏板信息的字典列表
            array_properties: 阵列属性字典
            image_path: 原始图像路径
            
        Returns:
            JSON格式的结果字典
        """
        # 获取图像文件名
        image_name = os.path.basename(image_path)
        
        # 创建输出字典
        output = {
            "image_source": image_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "array_properties": array_properties,
            "panels": []
        }
        
        # 添加面板信息
        for panel in panels:
            # 创建不包含图像数据的面板信息副本
            panel_info = {
                "id": panel["id"],
                "row": panel["row"],
                "column": panel["column"],
                "status": panel["status"],
                "anomaly_type": panel["anomaly_type"],
                "thermal_data": panel["thermal_data"],
                "description": panel["description"]
            }
            
            # 如果有原始位置信息，也添加
            if "original_position" in panel:
                panel_info["position"] = {
                    "x": panel["original_position"]["x"],
                    "y": panel["original_position"]["y"],
                    "width": panel["original_position"]["width"],
                    "height": panel["original_position"]["height"]
                }
            
            output["panels"].append(panel_info)
        
        # 保存中间JSON结果
        if self.debug_dir:
            debug_json_path = os.path.join(self.debug_dir, "analysis_debug.json")
            with open(debug_json_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        
        return output
    
    def save_results(self, annotated_image: np.ndarray, json_data: Dict[str, Any],
                   output_dir: str, base_filename: str = "output") -> tuple:
        """
        保存结果到文件
        
        Args:
            annotated_image: 带标注的图像
            json_data: JSON格式的结果数据
            output_dir: 输出目录
            base_filename: 基础文件名
            
        Returns:
            Tuple[图像文件路径, JSON文件路径]
        """
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存标注图像
        image_path = os.path.join(output_dir, f"{base_filename}_annotated.jpg")
        cv2.imwrite(image_path, annotated_image)
        
        # 保存JSON数据
        json_path = os.path.join(output_dir, f"{base_filename}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return image_path, json_path 