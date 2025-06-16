#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys

from main import main as process_image


def test_with_sample_image():
    """
    使用示例图像测试系统
    """
    # 创建示例图像目录
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    # 检查是否提供了示例图像路径
    parser = argparse.ArgumentParser(description='测试光伏板热成像分析系统')
    parser.add_argument('--image', '-i', type=str, help='示例热成像图片路径')
    parser.add_argument('--rgb', type=str, help='对应的RGB图像路径')
    parser.add_argument('--rows', '-r', type=int, help='光伏阵列行数，如果不提供则自动检测')
    parser.add_argument('--cols', '-c', type=int, help='光伏阵列列数，如果不提供则自动检测')
    parser.add_argument('--dual-mode', action='store_true', help='使用双图像模式（RGB+热成像）')
    args = parser.parse_args()
    
    if args.image and os.path.isfile(args.image):
        # 使用提供的图像
        image_path = args.image
        print(f"使用提供的热成像图像: {image_path}")
    else:
        print("未提供有效的热成像图像路径。请使用 --image 参数指定图像路径。")
        return
    
    # 设置系统参数并运行主程序
    sys_args = [
        'main.py',
        '--input', image_path,
        '--output', 'test_output',
        '--debug',
        '--save-debug'
    ]
    
    # 如果提供了RGB图像，添加到参数中
    if args.rgb and os.path.isfile(args.rgb):
        sys_args.extend(['--rgb', args.rgb])
        print(f"使用RGB参考图像: {args.rgb}")
    
    # 如果启用双图像模式
    if args.dual_mode:
        sys_args.append('--dual-mode')
        print("启用双图像模式")
    
    # 如果提供了行列数，添加到参数中
    if args.rows:
        sys_args.extend(['--rows', str(args.rows)])
    if args.cols:
        sys_args.extend(['--cols', str(args.cols)])
    
    # 设置命令行参数
    sys.argv = sys_args
    
    # 运行主程序
    process_image()
    
    # 检查输出结果
    output_json = os.path.join('test_output', os.path.splitext(os.path.basename(image_path))[0] + '_analysis.json')
    if os.path.isfile(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 打印分析摘要
        print("\n分析结果摘要:")
        print(f"总面板数: {result['array_properties']['total_panels']}")
        print(f"正常面板数: {result['array_properties']['total_panels'] - result['array_properties']['warning_panels'] - result['array_properties']['critical_panels']}")
        print(f"警告面板数: {result['array_properties']['warning_panels']}")
        print(f"严重异常面板数: {result['array_properties']['critical_panels']}")
        print(f"阵列健康度: {result['array_properties']['health_percentage']:.2f}%")
        
        # 打印异常面板详情
        if result['array_properties']['warning_panels'] + result['array_properties']['critical_panels'] > 0:
            print("\n异常面板详情:")
            for panel in result['panels']:
                if panel['status'] != 'Normal':
                    print(f"面板 {panel['id']}: {panel['status']} - {panel['anomaly_type']} - {panel['description']}")
    else:
        print(f"错误：未找到输出文件 {output_json}")


if __name__ == "__main__":
    test_with_sample_image() 