from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class ThermalAnalyzer:
    """
    光伏板热成像分析类
    负责分析每个光伏板的热状态
    """
    
    def __init__(self):
        """
        初始化热分析器
        """
        # 状态阈值，可根据实际情况调整
        self.warning_threshold = 0.1  # 高于平均温度10%为警告
        self.critical_threshold = 0.2  # 高于平均温度20%为严重
        
        # 热斑检测参数
        self.hotspot_area_threshold = 0.05  # 热斑面积占比阈值
        self.hotspot_intensity_threshold = 0.15  # 热斑强度阈值
        
        # 旁路二极管故障检测参数
        self.diode_area_ratio = 0.33  # 旁路二极管故障通常影响约1/3的面板
        self.diode_area_tolerance = 0.1  # 面积比例容差
    
    def analyze_panel(self, panel: Dict[str, Any], array_avg_value: float = None) -> Dict[str, Any]:
        """
        分析单个光伏板的热状态
        
        Args:
            panel: 包含光伏板信息的字典，必须包含"roi"键
            array_avg_value: 整个阵列的平均温度值，如果为None则不进行相对比较
            
        Returns:
            添加了热分析结果的面板字典
        """
        roi = panel["roi"]
        
        # 转换为灰度图（如果不是灰度图）
        if len(roi.shape) > 2:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
        
        # 计算ROI的统计数据
        avg_value = np.mean(gray_roi)
        max_value = np.max(gray_roi)
        min_value = np.min(gray_roi)
        std_dev = np.std(gray_roi)
        
        # 创建热数据字典
        thermal_data = {
            "average_value": float(avg_value),
            "max_value": float(max_value),
            "min_value": float(min_value),
            "std_deviation": float(std_dev)
        }
        
        # 初始化状态为正常
        status = "Normal"
        anomaly_type = "None"
        description = "Panel operating normally."
        
        # 如果提供了阵列平均值，进行相对比较
        if array_avg_value is not None:
            relative_temp = avg_value / array_avg_value
            
            # 检测异常
            if relative_temp > (1 + self.critical_threshold):
                status = "Critical"
                # 进一步分析异常类型
                anomaly_info = self._analyze_anomaly_type(gray_roi)
                anomaly_type = anomaly_info["type"]
                description = anomaly_info["description"]
            elif relative_temp > (1 + self.warning_threshold):
                status = "Warning"
                description = "Panel temperature is higher than average. Monitoring recommended."
        else:
            # 如果没有阵列平均值，使用标准差判断
            if std_dev > 15:  # 阈值可调整
                status = "Warning"
                description = "Panel shows significant temperature variation. Inspection recommended."
        
        # 更新面板信息
        panel.update({
            "status": status,
            "anomaly_type": anomaly_type,
            "thermal_data": thermal_data,
            "description": description
        })
        
        return panel
    
    def analyze_array(self, panels: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        分析整个光伏阵列
        
        Args:
            panels: 包含所有光伏板信息的字典列表
            
        Returns:
            Tuple[更新后的面板列表, 阵列属性字典]
        """
        # 计算整个阵列的平均温度
        avg_values = [np.mean(panel["roi"]) if len(panel["roi"].shape) == 2 
                     else np.mean(cv2.cvtColor(panel["roi"], cv2.COLOR_BGR2GRAY)) 
                     for panel in panels]
        array_avg_value = np.mean(avg_values)
        
        # 分析每个面板
        analyzed_panels = [self.analyze_panel(panel, array_avg_value) for panel in panels]
        
        # 计算阵列属性
        rows = max(panel["row"] for panel in panels)
        cols = max(panel["column"] for panel in panels)
        total_panels = len(panels)
        
        # 统计异常面板
        warning_count = sum(1 for panel in analyzed_panels if panel["status"] == "Warning")
        critical_count = sum(1 for panel in analyzed_panels if panel["status"] == "Critical")
        
        # 创建阵列属性字典
        array_properties = {
            "rows": rows,
            "columns": cols,
            "total_panels": total_panels,
            "array_avg_temperature": float(array_avg_value),
            "warning_panels": warning_count,
            "critical_panels": critical_count,
            "health_percentage": float((total_panels - warning_count - critical_count) / total_panels * 100)
        }
        
        return analyzed_panels, array_properties
    
    def _analyze_anomaly_type(self, gray_roi: np.ndarray) -> Dict[str, str]:
        """
        分析异常类型
        
        Args:
            gray_roi: 光伏板的灰度ROI图像
            
        Returns:
            包含异常类型和描述的字典
        """
        height, width = gray_roi.shape
        avg_value = np.mean(gray_roi)
        
        # 二值化图像，分离热区
        threshold = avg_value * (1 + self.hotspot_intensity_threshold)
        _, binary = cv2.threshold(gray_roi, threshold, 255, cv2.THRESH_BINARY)
        
        # 查找热区轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果没有热区，可能是整体温度均匀升高
        if not contours:
            return {
                "type": "Uniform Heating",
                "description": "Panel shows uniformly elevated temperature. Possible shading or dirt coverage."
            }
        
        # 计算最大热区的面积占比
        max_contour = max(contours, key=cv2.contourArea)
        max_area_ratio = cv2.contourArea(max_contour) / (height * width)
        
        # 分析热区的形状和位置
        x, y, w, h = cv2.boundingRect(max_contour)
        position_ratio_x = x / width
        position_ratio_y = y / height
        aspect_ratio = w / h if h > 0 else 0
        
        # 判断是否是旁路二极管故障（约1/3面板区域）
        diode_lower = self.diode_area_ratio - self.diode_area_tolerance
        diode_upper = self.diode_area_ratio + self.diode_area_tolerance
        
        if diode_lower < max_area_ratio < diode_upper:
            # 检查是否是垂直条状（左/中/右三分之一）
            if aspect_ratio < 0.5 and (position_ratio_x < 0.1 or 
                                      (position_ratio_x > 0.3 and position_ratio_x < 0.4) or 
                                      position_ratio_x > 0.6):
                return {
                    "type": "Diode Activated",
                    "description": "Approximately 1/3 of the panel shows elevated temperature. Bypass diode likely activated."
                }
        
        # 判断是否是热斑
        if max_area_ratio < self.hotspot_area_threshold:
            return {
                "type": "Hotspot",
                "description": "Panel shows localized hotspot. Possible cell damage or interconnection issue."
            }
        
        # 判断是否是串故障
        if aspect_ratio > 5 or aspect_ratio < 0.2:  # 非常细长的形状
            return {
                "type": "String Failure",
                "description": "Linear pattern of elevated temperature detected. Possible string interconnection failure."
            }
        
        # 其他未分类的异常
        return {
            "type": "Unknown Anomaly",
            "description": "Panel shows abnormal temperature pattern. Detailed inspection recommended."
        } 