"""
示例脚本 - PaintsUndo-Vector 基本用法

演示如何使用 PaintsUndo-Vector 从一张图片生成矢量笔画。
"""

import os
import sys

import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.optimizer import MultiStageOptimizer, StageConfig, DEFAULT_STAGE_CONFIGS
from core.renderer import DifferentiableRenderer
from export.svg_export import SVGExporter
from export.json_export import JSONExporter
from replay.player import StrokePlayer
from utils.image import load_image, save_image


def main():
    # 配置
    input_path = "examples/input.jpg"  # 替换为你的图片路径
    output_dir = "examples/output"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    canvas_size = (640, 480)
    
    # 加载目标图像
    if not os.path.exists(input_path):
        print(f"请将测试图片放在: {input_path}")
        print("或修改 input_path 变量指向你的图片")
        return
    
    target_image = load_image(input_path, canvas_size, device)
    
    # 创建优化器
    optimizer = MultiStageOptimizer(
        canvas_size=canvas_size,
        device=device,
        output_dir=output_dir,
    )
    
    # 使用默认阶段配置运行优化
    result = optimizer.optimize(
        target_image=target_image,
        stage_configs=DEFAULT_STAGE_CONFIGS,
    )
    
    # 导出 SVG
    svg_exporter = SVGExporter(canvas_size=canvas_size)
    svg_exporter.export(
        result["strokes"],
        result["brush_names"],
        os.path.join(output_dir, "result.svg"),
        include_animation=True,
    )
    
    # 导出 JSON
    json_exporter = JSONExporter()
    json_exporter.export(
        result["strokes"],
        result["brush_names"],
        os.path.join(output_dir, "strokes.json"),
        canvas_size=canvas_size,
    )
    
    # 导出 HTML 回放器
    player = StrokePlayer(canvas_size=canvas_size)
    player.export_html_player(
        result["stages_history"],
        os.path.join(output_dir, "replay.html"),
    )
    
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
