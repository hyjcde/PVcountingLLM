import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class ImageRegistration:
    """
    图像配准类
    用于将可见光图像的边缘信息映射到热成像图像上
    """
    
    def __init__(self, debug_mode: bool = False, debug_dir: str = None):
        """
        初始化图像配准器
        
        Args:
            debug_mode: 是否启用调试模式
            debug_dir: 调试输出目录
        """
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        self.rgb_image = None
        self.thermal_image = None
        self.registration_matrix = None
        
    def load_images(self, rgb_path: str, thermal_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载RGB图像和热成像图像
        
        Args:
            rgb_path: RGB图像路径
            thermal_path: 热成像图像路径
            
        Returns:
            Tuple[RGB图像, 热成像图像]
        """
        self.rgb_image = cv2.imread(rgb_path)
        self.thermal_image = cv2.imread(thermal_path)
        
        if self.rgb_image is None:
            raise FileNotFoundError(f"无法加载RGB图像: {rgb_path}")
        if self.thermal_image is None:
            raise FileNotFoundError(f"无法加载热成像图像: {thermal_path}")
        
        # 保存原始图像到调试目录
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "rgb_original.jpg"), self.rgb_image)
            cv2.imwrite(os.path.join(self.debug_dir, "thermal_original.jpg"), self.thermal_image)
        
        return self.rgb_image, self.thermal_image
    
    def register_images(self, method: str = "orb") -> np.ndarray:
        """
        配准两张图像
        
        Args:
            method: 配准方法，可选 "orb", "sift", "surf"
            
        Returns:
            配准变换矩阵
        """
        # 转换为灰度图
        rgb_gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        thermal_gray = cv2.cvtColor(self.thermal_image, cv2.COLOR_BGR2GRAY)
        
        # 调整热成像图像大小以匹配RGB图像
        thermal_resized = cv2.resize(thermal_gray, (rgb_gray.shape[1], rgb_gray.shape[0]))
        
        if method.lower() == "orb":
            detector = cv2.ORB_create(nfeatures=5000)
        elif method.lower() == "sift":
            detector = cv2.SIFT_create()
        elif method.lower() == "surf":
            detector = cv2.xfeatures2d.SURF_create()
        else:
            detector = cv2.ORB_create(nfeatures=5000)
        
        # 检测关键点和描述符
        kp1, des1 = detector.detectAndCompute(rgb_gray, None)
        kp2, des2 = detector.detectAndCompute(thermal_resized, None)
        
        if des1 is None or des2 is None:
            print("警告：无法检测到足够的特征点，使用简单的尺度变换")
            # 如果特征检测失败，使用简单的尺度变换
            h_ratio = self.thermal_image.shape[0] / self.rgb_image.shape[0]
            w_ratio = self.thermal_image.shape[1] / self.rgb_image.shape[1]
            self.registration_matrix = np.array([
                [w_ratio, 0, 0],
                [0, h_ratio, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            return self.registration_matrix
        
        # 匹配特征点
        if method.lower() == "orb":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 保留最好的匹配
        good_matches = matches[:min(len(matches), 100)]
        
        if len(good_matches) < 10:
            print("警告：匹配点数量不足，使用简单的尺度变换")
            h_ratio = self.thermal_image.shape[0] / self.rgb_image.shape[0]
            w_ratio = self.thermal_image.shape[1] / self.rgb_image.shape[1]
            self.registration_matrix = np.array([
                [w_ratio, 0, 0],
                [0, h_ratio, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            return self.registration_matrix
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 调整目标点坐标到原始热成像图像尺寸
        scale_x = self.thermal_image.shape[1] / thermal_resized.shape[1]
        scale_y = self.thermal_image.shape[0] / thermal_resized.shape[0]
        dst_pts[:, :, 0] *= scale_x
        dst_pts[:, :, 1] *= scale_y
        
        # 计算单应性矩阵
        self.registration_matrix, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 5.0
        )
        
        # 保存调试信息
        if self.debug_dir and self.debug_mode:
            # 绘制匹配点
            matched_img = cv2.drawMatches(
                rgb_gray, kp1, thermal_resized, kp2, 
                good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(os.path.join(self.debug_dir, "feature_matches.jpg"), matched_img)
        
        return self.registration_matrix
    
    def extract_rgb_edges(self) -> np.ndarray:
        """
        从RGB图像中提取清晰的边缘信息
        
        Returns:
            边缘图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 形态学操作增强边缘
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "rgb_edges.jpg"), edges)
        
        return edges
    
    def map_edges_to_thermal(self, edges: np.ndarray) -> np.ndarray:
        """
        将RGB图像的边缘映射到热成像图像坐标系
        
        Args:
            edges: RGB图像的边缘
            
        Returns:
            映射到热成像坐标系的边缘
        """
        if self.registration_matrix is None:
            raise ValueError("请先调用register_images方法进行图像配准")
        
        # 将边缘图像变换到热成像坐标系
        mapped_edges = cv2.warpPerspective(
            edges, 
            self.registration_matrix, 
            (self.thermal_image.shape[1], self.thermal_image.shape[0])
        )
        
        # 保存调试信息
        if self.debug_dir:
            cv2.imwrite(os.path.join(self.debug_dir, "mapped_edges.jpg"), mapped_edges)
            
            # 创建叠加图像显示配准效果
            thermal_gray = cv2.cvtColor(self.thermal_image, cv2.COLOR_BGR2GRAY)
            overlay = cv2.addWeighted(thermal_gray, 0.7, mapped_edges, 0.3, 0)
            cv2.imwrite(os.path.join(self.debug_dir, "edge_overlay.jpg"), overlay)
        
        return mapped_edges
    
    def find_panel_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        """
        从边缘图像中查找光伏板轮廓
        
        Args:
            edges: 边缘图像
            
        Returns:
            光伏板轮廓列表
        """
        # 形态学操作增强边缘
        kernel = np.ones((3, 3), np.uint8)
        edges_enhanced = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges_enhanced = cv2.morphologyEx(edges_enhanced, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges_enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓：保留面积适中的矩形轮廓
        panel_contours = []
        min_area = 100  # 降低最小面积阈值
        max_area = 20000  # 降低最大面积阈值
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # 计算轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # 检查长宽比是否合理（光伏板通常接近正方形或矩形）
                if 0.3 < aspect_ratio < 3.0:
                    # 计算轮廓面积与边界框面积的比值
                    bbox_area = w * h
                    fill_ratio = area / bbox_area if bbox_area > 0 else 0
                    
                    # 如果填充比例合理，认为是有效轮廓
                    if fill_ratio > 0.3:
                        panel_contours.append(contour)
        
        # 如果检测到的轮廓太少，尝试使用更宽松的参数
        if len(panel_contours) < 10:
            print(f"检测到的轮廓数量较少({len(panel_contours)})，尝试使用更宽松的参数...")
            panel_contours = []
            min_area = 50  # 进一步降低最小面积
            max_area = 50000  # 增加最大面积
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.2 < aspect_ratio < 5.0:  # 更宽松的长宽比
                        bbox_area = w * h
                        fill_ratio = area / bbox_area if bbox_area > 0 else 0
                        
                        if fill_ratio > 0.2:  # 更宽松的填充比例
                            panel_contours.append(contour)
        
        # 保存调试信息
        if self.debug_dir:
            debug_img = self.thermal_image.copy()
            cv2.drawContours(debug_img, panel_contours, -1, (0, 255, 0), 2)
            
            # 在每个轮廓上标注面积信息
            for i, contour in enumerate(panel_contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(debug_img, f"{i}:{int(area)}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imwrite(os.path.join(self.debug_dir, "detected_panels.jpg"), debug_img)
            
            # 保存增强的边缘图像
            cv2.imwrite(os.path.join(self.debug_dir, "edges_enhanced.jpg"), edges_enhanced)
        
        print(f"检测到 {len(panel_contours)} 个候选轮廓")
        return panel_contours
    
    def create_grid_from_contours(self, contours: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        从轮廓创建网格化的面板信息
        
        Args:
            contours: 光伏板轮廓列表
            
        Returns:
            面板信息列表
        """
        panels = []
        
        if not contours:
            print("警告：未检测到任何轮廓，尝试使用网格分割...")
            return self._create_grid_fallback()
        
        # 计算每个轮廓的边界框和中心点
        panel_info = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            area = cv2.contourArea(contour)
            panel_info.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'area': area,
                'index': i
            })
        
        # 按位置排序：先按y坐标（行），再按x坐标（列）
        panel_info.sort(key=lambda p: (p['center'][1], p['center'][0]))
        
        # 确定网格结构 - 改进的聚类算法
        y_coords = [p['center'][1] for p in panel_info]
        x_coords = [p['center'][0] for p in panel_info]
        
        # 聚类y坐标确定行
        y_clusters = self._cluster_coordinates(y_coords, tolerance=30)
        # 聚类x坐标确定列
        x_clusters = self._cluster_coordinates(x_coords, tolerance=30)
        
        print(f"检测到 {len(y_clusters)} 行, {len(x_clusters)} 列")
        
        # 为每个面板分配行列号
        for i, info in enumerate(panel_info):
            center_y = info['center'][1]
            center_x = info['center'][0]
            
            # 确定行号
            row = 1
            for j, y_cluster in enumerate(y_clusters):
                if abs(center_y - y_cluster) <= 30:
                    row = j + 1
                    break
            
            # 确定列号
            col = 1
            for j, x_cluster in enumerate(x_clusters):
                if abs(center_x - x_cluster) <= 30:
                    col = j + 1
                    break
            
            # 提取ROI
            x, y, w, h = info['bbox']
            # 确保ROI在图像范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, self.thermal_image.shape[1] - x)
            h = min(h, self.thermal_image.shape[0] - y)
            
            if w > 0 and h > 0:
                roi = self.thermal_image[y:y+h, x:x+w]
                
                panel = {
                    "id": f"R{row}-C{col}",
                    "row": row,
                    "column": col,
                    "roi": roi,
                    "position": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    },
                    "contour": info['contour'],
                    "area": info['area']
                }
                
                panels.append(panel)
        
        # 保存调试信息
        if self.debug_dir:
            debug_img = self.thermal_image.copy()
            for panel in panels:
                pos = panel["position"]
                cv2.rectangle(debug_img, (pos["x"], pos["y"]), 
                             (pos["x"] + pos["width"], pos["y"] + pos["height"]), 
                             (0, 255, 0), 2)
                cv2.putText(debug_img, panel["id"], 
                           (pos["x"] + 5, pos["y"] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                # 显示面积信息
                cv2.putText(debug_img, f"A:{int(panel['area'])}", 
                           (pos["x"] + 5, pos["y"] + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            cv2.imwrite(os.path.join(self.debug_dir, "grid_panels.jpg"), debug_img)
        
        return panels
    
    def _cluster_coordinates(self, coords: List[int], tolerance: int = 30) -> List[int]:
        """
        聚类坐标
        
        Args:
            coords: 坐标列表
            tolerance: 聚类容差
            
        Returns:
            聚类中心列表
        """
        if not coords:
            return []
        
        coords_sorted = sorted(coords)
        clusters = []
        current_cluster = [coords_sorted[0]]
        
        for coord in coords_sorted[1:]:
            if coord - current_cluster[-1] <= tolerance:
                current_cluster.append(coord)
            else:
                # 计算当前聚类的中心
                cluster_center = sum(current_cluster) // len(current_cluster)
                clusters.append(cluster_center)
                current_cluster = [coord]
        
        # 添加最后一个聚类
        if current_cluster:
            cluster_center = sum(current_cluster) // len(current_cluster)
            clusters.append(cluster_center)
        
        return clusters
    
    def _create_grid_fallback(self) -> List[Dict[str, Any]]:
        """
        当轮廓检测失败时的备用网格分割方法
        
        Returns:
            面板信息列表
        """
        print("使用备用网格分割方法...")
        panels = []
        
        # 使用固定的网格大小
        h, w = self.thermal_image.shape[:2]
        
        # 估算网格大小（基于图像大小）
        estimated_rows = max(8, h // 100)
        estimated_cols = max(15, w // 80)
        
        cell_height = h // estimated_rows
        cell_width = w // estimated_cols
        
        for r in range(estimated_rows):
            for c in range(estimated_cols):
                x = c * cell_width
                y = r * cell_height
                
                # 确保不超出图像边界
                if x + cell_width <= w and y + cell_height <= h:
                    roi = self.thermal_image[y:y+cell_height, x:x+cell_width]
                    
                    panel = {
                        "id": f"R{r+1}-C{c+1}",
                        "row": r + 1,
                        "column": c + 1,
                        "roi": roi,
                        "position": {
                            "x": x,
                            "y": y,
                            "width": cell_width,
                            "height": cell_height
                        }
                    }
                    
                    panels.append(panel)
        
        return panels
    
    def detect_array_region(self, edges: np.ndarray) -> Tuple[int, int, int, int]:
        """
        检测光伏阵列的主要区域
        
        Args:
            edges: 边缘图像
            
        Returns:
            Tuple[x, y, width, height] 阵列区域的边界框
        """
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果没有找到轮廓，使用图像中心区域
            h, w = edges.shape
            margin_x = w // 8
            margin_y = h // 8
            return margin_x, margin_y, w - 2*margin_x, h - 2*margin_y
        
        # 找到最大的轮廓作为主阵列区域
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 保存调试信息
        if self.debug_dir:
            debug_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.imwrite(os.path.join(self.debug_dir, "array_region.jpg"), debug_img)
        
        return x, y, w, h
    
    def create_fixed_grid(self, array_region: Tuple[int, int, int, int], 
                         rows: int = None, cols: int = None) -> List[Dict[str, Any]]:
        """
        在检测到的阵列区域内创建固定网格
        
        Args:
            array_region: 阵列区域 (x, y, width, height)
            rows: 行数，如果为None则自动估算
            cols: 列数，如果为None则自动估算
            
        Returns:
            面板信息列表
        """
        x, y, w, h = array_region
        
        # 如果没有指定行列数，根据图像大小和光伏板典型尺寸估算
        if rows is None:
            # 假设每个光伏板高度约为50-100像素
            estimated_panel_height = 80
            rows = max(6, h // estimated_panel_height)
        
        if cols is None:
            # 假设每个光伏板宽度约为50-100像素
            estimated_panel_width = 80
            cols = max(10, w // estimated_panel_width)
        
        print(f"使用固定网格: {rows} 行 x {cols} 列")
        
        # 计算每个网格的大小
        cell_width = w // cols
        cell_height = h // rows
        
        panels = []
        
        for r in range(rows):
            for c in range(cols):
                # 计算当前网格的位置
                grid_x = x + c * cell_width
                grid_y = y + r * cell_height
                
                # 确保不超出阵列区域
                actual_width = min(cell_width, x + w - grid_x)
                actual_height = min(cell_height, y + h - grid_y)
                
                if actual_width > 10 and actual_height > 10:  # 确保网格足够大
                    # 提取ROI
                    roi = self.thermal_image[grid_y:grid_y+actual_height, 
                                           grid_x:grid_x+actual_width]
                    
                    panel = {
                        "id": f"R{r+1}-C{c+1}",
                        "row": r + 1,
                        "column": c + 1,
                        "roi": roi,
                        "position": {
                            "x": grid_x,
                            "y": grid_y,
                            "width": actual_width,
                            "height": actual_height
                        }
                    }
                    
                    panels.append(panel)
        
        # 保存调试信息
        if self.debug_dir:
            debug_img = self.thermal_image.copy()
            for panel in panels:
                pos = panel["position"]
                cv2.rectangle(debug_img, (pos["x"], pos["y"]), 
                             (pos["x"] + pos["width"], pos["y"] + pos["height"]), 
                             (0, 255, 0), 1)
                cv2.putText(debug_img, panel["id"], 
                           (pos["x"] + 5, pos["y"] + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(self.debug_dir, "fixed_grid.jpg"), debug_img)
        
        return panels
    
    def refine_grid_with_edges(self, panels: List[Dict[str, Any]], 
                              edges: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用边缘信息优化网格分割
        
        Args:
            panels: 初始网格面板列表
            edges: 边缘图像
            
        Returns:
            优化后的面板列表
        """
        refined_panels = []
        
        for panel in panels:
            pos = panel["position"]
            x, y, w, h = pos["x"], pos["y"], pos["width"], pos["height"]
            
            # 提取当前面板区域的边缘
            panel_edges = edges[y:y+h, x:x+w]
            
            # 在面板区域内查找轮廓
            contours, _ = cv2.findContours(panel_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # 如果轮廓面积足够大，使用轮廓边界框优化位置
                if area > (w * h * 0.1):  # 轮廓面积至少占网格的10%
                    cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                    
                    # 调整到全局坐标系
                    refined_x = x + cx
                    refined_y = y + cy
                    refined_w = cw
                    refined_h = ch
                    
                    # 确保优化后的区域在图像范围内
                    refined_x = max(0, refined_x)
                    refined_y = max(0, refined_y)
                    refined_w = min(refined_w, self.thermal_image.shape[1] - refined_x)
                    refined_h = min(refined_h, self.thermal_image.shape[0] - refined_y)
                    
                    if refined_w > 10 and refined_h > 10:
                        # 更新ROI
                        roi = self.thermal_image[refined_y:refined_y+refined_h, 
                                               refined_x:refined_x+refined_w]
                        
                        panel["position"] = {
                            "x": refined_x,
                            "y": refined_y,
                            "width": refined_w,
                            "height": refined_h
                        }
                        panel["roi"] = roi
            
            refined_panels.append(panel)
        
        # 保存调试信息
        if self.debug_dir:
            debug_img = self.thermal_image.copy()
            for panel in refined_panels:
                pos = panel["position"]
                cv2.rectangle(debug_img, (pos["x"], pos["y"]), 
                             (pos["x"] + pos["width"], pos["y"] + pos["height"]), 
                             (255, 0, 0), 1)
                cv2.putText(debug_img, panel["id"], 
                           (pos["x"] + 5, pos["y"] + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(self.debug_dir, "refined_grid.jpg"), debug_img)
        
        return refined_panels
    
    def process_dual_images(self, rgb_path: str, thermal_path: str, 
                           rows: int = None, cols: int = None) -> List[Dict[str, Any]]:
        """
        处理双图像：从RGB图像提取边缘，映射到热成像图像，并分割面板
        
        Args:
            rgb_path: RGB图像路径
            thermal_path: 热成像图像路径
            rows: 指定的行数
            cols: 指定的列数
            
        Returns:
            面板信息列表
        """
        # 加载图像
        self.load_images(rgb_path, thermal_path)
        
        # 配准图像
        print("正在进行图像配准...")
        self.register_images()
        
        # 从RGB图像提取边缘
        print("从RGB图像提取边缘...")
        rgb_edges = self.extract_rgb_edges()
        
        # 将边缘映射到热成像坐标系
        print("将边缘映射到热成像坐标系...")
        mapped_edges = self.map_edges_to_thermal(rgb_edges)
        
        # 检测阵列主要区域
        print("检测光伏阵列区域...")
        array_region = self.detect_array_region(mapped_edges)
        
        # 创建固定网格
        print("创建固定网格分割...")
        panels = self.create_fixed_grid(array_region, rows, cols)
        
        # 使用边缘信息优化网格
        print("使用边缘信息优化网格...")
        refined_panels = self.refine_grid_with_edges(panels, mapped_edges)
        
        print(f"最终检测到 {len(refined_panels)} 个光伏板")
        
        return refined_panels 