from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class ImageProcessor:
    """
    光伏板热成像图像处理类
    负责图像预处理、透视校正和面板分割
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        初始化图像处理器
        
        Args:
            debug_mode: 是否启用调试模式，显示中间处理结果
        """
        self.debug_mode = debug_mode
        self.original_image = None
        self.warped_image = None
        self.perspective_matrix = None
        self.array_corners = None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像并返回
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            加载的图像
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        return self.original_image
    
    def preprocess_image(self, image: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        图像预处理：增强对比度、降噪、边缘检测等
        
        Args:
            image: 输入图像，如果为None则使用已加载的原始图像
            
        Returns:
            Tuple[灰度图, 二值化图像, 边缘检测结果]
        """
        if image is None:
            if self.original_image is None:
                raise ValueError("未加载图像，请先调用load_image方法")
            image = self.original_image
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用CLAHE（对比度受限的自适应直方图均衡化）增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 使用自适应阈值分割
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # 形态学操作清理噪声
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 边缘检测（使用更适合热成像的参数）
        edges = cv2.Canny(blurred, 30, 100)
        
        if self.debug_mode:
            cv2.imshow("Gray Image", gray)
            cv2.imshow("Enhanced Image", enhanced)
            cv2.imshow("Binary Image", binary)
            cv2.imshow("Edges", edges)
            cv2.waitKey(0)
        
        return gray, binary, edges
    
    def find_array_corners(self, binary: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        在二值图像和边缘图上查找光伏阵列的四个角点
        
        Args:
            binary: 二值化图像
            edges: 边缘检测结果图像
            
        Returns:
            光伏阵列的四个角点坐标，按顺时针排序
        """
        # 查找轮廓（优先使用二值图像）
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果二值图像没有找到轮廓，尝试使用边缘图像
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("未能检测到任何轮廓")
        
        # 按面积排序并保留最大的几个轮廓
        large_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        # 尝试找到最接近矩形的轮廓
        best_contour = None
        best_rect_score = float('inf')
        
        for contour in large_contours:
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 计算轮廓面积与其最小外接矩形面积的比值
            contour_area = cv2.contourArea(contour)
            rect_area = cv2.contourArea(box)
            if rect_area > 0:
                # 比值越接近1，说明轮廓越接近矩形
                rect_score = abs(1 - contour_area / rect_area)
                if rect_score < best_rect_score:
                    best_rect_score = rect_score
                    best_contour = contour
        
        if best_contour is None:
            # 如果没有找到合适的轮廓，使用最大的轮廓
            best_contour = large_contours[0]
        
        # 获取近似多边形
        epsilon = 0.02 * cv2.arcLength(best_contour, True)
        approx = cv2.approxPolyDP(best_contour, epsilon, True)
        
        # 如果近似后的多边形不是四边形，则使用最小外接矩形的四个角点
        if len(approx) != 4:
            rect = cv2.minAreaRect(best_contour)
            box = cv2.boxPoints(rect)
            approx = np.int0(box)
        
        # 确保角点按顺时针排序
        approx = self._order_points(approx.reshape(len(approx), 2))
        
        self.array_corners = approx
        
        if self.debug_mode:
            debug_img = self.original_image.copy()
            # 绘制所有大轮廓
            cv2.drawContours(debug_img, large_contours, -1, (0, 255, 0), 2)
            # 绘制选中的轮廓
            cv2.drawContours(debug_img, [best_contour], 0, (0, 0, 255), 3)
            # 绘制角点
            for point in approx:
                point_tuple = (int(point[0]), int(point[1]))
                cv2.circle(debug_img, point_tuple, 10, (255, 0, 0), -1)
            cv2.imshow("Detected Contours and Corners", debug_img)
            cv2.waitKey(0)
        
        return approx
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        对四个角点进行排序，使其按照左上、右上、右下、左下的顺时针顺序排列
        
        Args:
            pts: 输入的四个角点坐标
            
        Returns:
            排序后的角点坐标
        """
        # 初始化排序后的点
        rect = np.zeros((4, 2), dtype="float32")
        
        # 左上角点的坐标和最小，右下角点的坐标和最大
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        
        # 计算点的差值，右上角点的差值最小，左下角点的差值最大
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        
        return rect
    
    def perspective_transform(self, corners: np.ndarray = None, target_width: int = 1000, target_height: int = 1000) -> np.ndarray:
        """
        执行透视变换，将倾斜的光伏阵列转换为正视图
        
        Args:
            corners: 光伏阵列的四个角点，如果为None则使用已检测的角点
            target_width: 输出图像的宽度
            target_height: 输出图像的高度
            
        Returns:
            透视校正后的图像
        """
        if corners is None:
            if self.array_corners is None:
                raise ValueError("未检测到光伏阵列角点，请先调用find_array_corners方法")
            corners = self.array_corners
        
        # 定义目标四边形的四个角点（正视图）
        dst = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype="float32")
        
        # 计算透视变换矩阵
        self.perspective_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        
        # 执行透视变换
        self.warped_image = cv2.warpPerspective(self.original_image, self.perspective_matrix, (target_width, target_height))
        
        if self.debug_mode:
            cv2.imshow("Warped Image", self.warped_image)
            cv2.waitKey(0)
        
        return self.warped_image
    
    def segment_panels_by_grid(self, warped_image: np.ndarray = None, rows: int = None, cols: int = None) -> List[Dict[str, Any]]:
        """
        通过网格法分割光伏板
        
        Args:
            warped_image: 透视校正后的图像，如果为None则使用已校正的图像
            rows: 光伏阵列的行数，如果为None则自动检测
            cols: 光伏阵列的列数，如果为None则自动检测
            
        Returns:
            包含每个光伏板信息的字典列表，每个字典包含id, row, col, roi等信息
        """
        if warped_image is None:
            if self.warped_image is None:
                raise ValueError("未进行透视校正，请先调用perspective_transform方法")
            warped_image = self.warped_image
        
        # 如果未提供行列数，尝试自动检测
        if rows is None or cols is None:
            rows, cols = self._detect_grid_size(warped_image)
        
        height, width = warped_image.shape[:2]
        cell_width = width // cols
        cell_height = height // rows
        
        panels = []
        
        # 遍历网格，分割每个光伏板
        for r in range(rows):
            for c in range(cols):
                x = c * cell_width
                y = r * cell_height
                
                # 提取ROI
                roi = warped_image[y:y+cell_height, x:x+cell_width]
                
                # 创建面板信息字典
                panel_info = {
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
                
                panels.append(panel_info)
        
        if self.debug_mode:
            debug_img = warped_image.copy()
            for panel in panels:
                pos = panel["position"]
                cv2.rectangle(debug_img, 
                             (pos["x"], pos["y"]), 
                             (pos["x"] + pos["width"], pos["y"] + pos["height"]), 
                             (0, 255, 0), 2)
                cv2.putText(debug_img, panel["id"], 
                           (pos["x"] + 10, pos["y"] + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Segmented Panels", debug_img)
            cv2.waitKey(0)
        
        return panels
    
    def segment_panels(self, warped_image: np.ndarray = None, rows: int = None, cols: int = None) -> List[Dict[str, Any]]:
        """
        分割光伏板（结合网格法和轮廓检测）
        
        Args:
            warped_image: 透视校正后的图像，如果为None则使用已校正的图像
            rows: 光伏阵列的行数，如果为None则自动检测
            cols: 光伏阵列的列数，如果为None则自动检测
            
        Returns:
            包含每个光伏板信息的字典列表
        """
        # 首先尝试使用轮廓检测方法
        try:
            # 使用网格法作为备选方案
            return self.segment_panels_by_grid(warped_image, rows, cols)
        except Exception as e:
            print(f"网格分割失败: {str(e)}，尝试使用网格法")
            return self.segment_panels_by_grid(warped_image, rows, cols)
    
    def _detect_grid_size(self, warped_image: np.ndarray) -> Tuple[int, int]:
        """
        尝试自动检测光伏阵列的行数和列数
        
        Args:
            warped_image: 透视校正后的图像
            
        Returns:
            Tuple[行数, 列数]
        """
        # 转换为灰度图
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 使用Sobel算子检测水平和垂直边缘
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # 转换为绝对值并归一化
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        
        # 二值化
        _, binary_x = cv2.threshold(abs_sobel_x, 50, 255, cv2.THRESH_BINARY)
        _, binary_y = cv2.threshold(abs_sobel_y, 50, 255, cv2.THRESH_BINARY)
        
        if self.debug_mode:
            cv2.imshow("Horizontal Edges", binary_x)
            cv2.imshow("Vertical Edges", binary_y)
            cv2.waitKey(0)
        
        # 投影到x轴和y轴
        projection_x = np.sum(binary_x, axis=0) / 255
        projection_y = np.sum(binary_y, axis=1) / 255
        
        # 平滑投影曲线
        projection_x = cv2.GaussianBlur(projection_x, (15, 1), 0)
        projection_y = cv2.GaussianBlur(projection_y, (15, 1), 0)
        
        # 查找峰值（表示边界）
        peaks_x = self._find_peaks(projection_x)
        peaks_y = self._find_peaks(projection_y)
        
        # 计算行列数
        cols = len(peaks_x) + 1
        rows = len(peaks_y) + 1
        
        # 如果检测失败，使用默认值
        if rows < 2:
            rows = 8  # 默认行数
        if cols < 2:
            cols = 15  # 默认列数
        
        print(f"检测到光伏阵列大小: {rows} 行 x {cols} 列")
        
        return rows, cols
    
    def _find_peaks(self, signal: np.ndarray, min_distance: int = 20) -> List[int]:
        """
        在信号中查找峰值
        
        Args:
            signal: 一维信号
            min_distance: 峰值之间的最小距离
            
        Returns:
            峰值位置列表
        """
        # 计算信号的一阶导数
        diff = np.diff(signal)
        
        # 查找导数符号变化的位置（峰值）
        peaks = []
        for i in range(1, len(diff)):
            if diff[i-1] > 0 and diff[i] <= 0:
                peaks.append(i)
        
        # 过滤掉太近的峰值
        filtered_peaks = []
        if peaks:
            filtered_peaks.append(peaks[0])
            for peak in peaks[1:]:
                if peak - filtered_peaks[-1] >= min_distance:
                    filtered_peaks.append(peak)
        
        return filtered_peaks
    
    def _cluster_lines(self, lines: List[int], threshold: int = 20) -> List[int]:
        """
        聚类相似的线
        
        Args:
            lines: 线的位置列表
            threshold: 聚类阈值
            
        Returns:
            聚类后的线位置列表
        """
        if not lines:
            return []
        
        lines.sort()
        clusters = []
        current_cluster = [lines[0]]
        
        for i in range(1, len(lines)):
            if lines[i] - current_cluster[-1] < threshold:
                current_cluster.append(lines[i])
            else:
                clusters.append(sum(current_cluster) // len(current_cluster))
                current_cluster = [lines[i]]
        
        if current_cluster:
            clusters.append(sum(current_cluster) // len(current_cluster))
        
        return clusters 