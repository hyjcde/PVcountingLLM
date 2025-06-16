#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
from datetime import datetime

import cv2
import numpy as np

from image_processor import ImageProcessor
from image_registration import ImageRegistration
from thermal_analyzer import ThermalAnalyzer
from visualizer import Visualizer


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='光伏板热成像自动识别与分析')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入热成像图片路径')
    parser.add_argument('--rgb', type=str, help='对应的RGB图像路径（用于更好的边缘检测）')
    parser.add_argument('--output', '-o', type=str, default='output', help='输出目录路径')
    parser.add_argument('--rows', '-r', type=int, help='光伏阵列行数，如果不提供则自动检测')
    parser.add_argument('--cols', '-c', type=int, help='光伏阵列列数，如果不提供则自动检测')
    parser.add_argument('--debug', '-d', action='store_true', help='启用调试模式，显示中间处理结果')
    parser.add_argument('--save-debug', '-s', action='store_true', help='保存中间处理结果到文件')
    parser.add_argument('--dual-mode', action='store_true', help='使用双图像模式（RGB+热成像）')
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"错误：输入文件 {args.input} 不存在")
        return
    
    # 如果启用双图像模式，检查RGB图像是否存在
    if args.dual_mode or args.rgb:
        if not args.rgb:
            print("错误：双图像模式需要提供RGB图像路径")
            return
        if not os.path.isfile(args.rgb):
            print(f"错误：RGB图像文件 {args.rgb} 不存在")
            return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 创建调试输出目录
    debug_dir = None
    if args.save_debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(args.input))[0]
        debug_dir = os.path.join(args.output, f"debug_{base_filename}_{timestamp}")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 复制原始图像到调试目录
        shutil.copy(args.input, os.path.join(debug_dir, os.path.basename(args.input)))
        if args.rgb:
            shutil.copy(args.rgb, os.path.join(debug_dir, os.path.basename(args.rgb)))
    
    # 获取输入文件的基础名称（不含扩展名）
    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    
    # 初始化处理器
    thermal_analyzer = ThermalAnalyzer()
    visualizer = Visualizer(debug_dir=debug_dir)
    
    print(f"处理图像: {args.input}")
    if args.rgb:
        print(f"RGB参考图像: {args.rgb}")
    
    try:
        panels = None
        array_corners = None
        original_image = cv2.imread(args.input)
        
        if args.dual_mode or args.rgb:
            # 使用双图像模式
            print("使用双图像模式进行处理...")
            image_registration = ImageRegistration(debug_mode=args.debug, debug_dir=debug_dir)
            panels = image_registration.process_dual_images(args.rgb, args.input, args.rows, args.cols)
            
            # 对于双图像模式，我们不需要透视校正，因为已经在热成像坐标系中
            # 但我们需要为可视化器提供一些虚拟的角点
            h, w = original_image.shape[:2]
            array_corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.int32)
            
        else:
            # 使用单图像模式
            print("使用单图像模式进行处理...")
            image_processor = ImageProcessor(debug_mode=args.debug)
            
            # 第一阶段：图像预处理与阵列定位
            print("阶段1: 图像预处理与阵列定位")
            original_image = image_processor.load_image(args.input)
            
            # 保存原始图像到调试目录
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "original.jpg"), original_image)
            
            gray, binary, edges = image_processor.preprocess_image()
            
            # 保存预处理结果到调试目录
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "gray.jpg"), gray)
                cv2.imwrite(os.path.join(debug_dir, "binary.jpg"), binary)
                cv2.imwrite(os.path.join(debug_dir, "edges.jpg"), edges)
            
            array_corners = image_processor.find_array_corners(binary, edges)
            
            # 第二阶段：透视校正与单板分割
            print("阶段2: 透视校正与单板分割")
            warped_image = image_processor.perspective_transform(array_corners)
            
            # 保存透视校正结果到调试目录
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "warped.jpg"), warped_image)
            
            panels = image_processor.segment_panels(warped_image, args.rows, args.cols)
            
            # 保存分割结果到调试目录
            if debug_dir:
                segmented_img = warped_image.copy()
                for panel in panels:
                    pos = panel["position"]
                    cv2.rectangle(segmented_img, 
                                 (pos["x"], pos["y"]), 
                                 (pos["x"] + pos["width"], pos["y"] + pos["height"]), 
                                 (0, 255, 0), 2)
                    cv2.putText(segmented_img, panel["id"], 
                               (pos["x"] + 10, pos["y"] + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(debug_dir, "segmented.jpg"), segmented_img)
        
        # 第三阶段：单板分析与数据结构化
        print("阶段3: 单板分析与数据结构化")
        analyzed_panels, array_properties = thermal_analyzer.analyze_array(panels)
        
        # 处理面板位置映射
        if args.dual_mode or args.rgb:
            # 双图像模式下，面板已经在原始坐标系中，直接添加original_position
            for panel in analyzed_panels:
                panel["original_position"] = panel["position"].copy()
            mapped_panels = analyzed_panels
        else:
            # 单图像模式下，需要将面板位置映射回原始图像
            mapped_panels = visualizer.map_panels_to_original(
                analyzed_panels, 
                image_processor.perspective_matrix, 
                original_image.shape
            )
        
        # 第四阶段：结果可视化
        print("阶段4: 结果可视化")
        annotated_image = visualizer.create_annotated_image(
            original_image, 
            mapped_panels, 
            array_corners
        )
        
        # 生成JSON输出
        json_data = visualizer.generate_json_output(
            mapped_panels, 
            array_properties, 
            args.input
        )
        
        # 保存结果
        image_path, json_path = visualizer.save_results(
            annotated_image, 
            json_data, 
            args.output, 
            base_filename
        )
        
        print(f"处理完成！")
        print(f"标注图像已保存至: {image_path}")
        print(f"分析结果已保存至: {json_path}")
        
        if debug_dir:
            print(f"调试文件已保存至: {debug_dir}")
        
        # 如果在调试模式下，显示结果图像
        if args.debug:
            cv2.imshow("Annotated Image", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 