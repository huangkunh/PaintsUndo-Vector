"""
命令行入口

使用方式：
    paints-undo-vector --input target.png --output output/
    paints-undo-vector --input target.png --output output/ --brush marker,pencil
    paints-undo-vector --input target.png --output output/ --config custom.yaml
"""

import argparse
import os
import sys
import yaml

import torch

from core.optimizer import MultiStageOptimizer, StageConfig
from core.renderer import DifferentiableRenderer
from core.scheduler import StageScheduler
from export.svg_export import SVGExporter
from export.json_export import JSONExporter
from replay.player import StrokePlayer
from utils.image import load_image, save_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaintsUndo-Vector: 基于可微渲染优化的矢量笔画生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  paints-undo-vector --input photo.png --output result/
  
  # 指定笔刷
  paints-undo-vector --input photo.png --output result/ --brush marker,pencil,watercolor
  
  # 自定义配置
  paints-undo-vector --input photo.png --output result/ --config my_config.yaml
  
  # 仅运行特定阶段
  paints-undo-vector --input photo.png --output result/ --stage 2
  
  # 指定 GPU
  paints-undo-vector --input photo.png --output result/ --device cuda:0
        """,
    )
    
    parser.add_argument("--input", "-i", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出目录")
    parser.add_argument("--config", "-c", type=str, default=None, help="配置文件路径（YAML）")
    parser.add_argument("--brush", "-b", type=str, default=None, help="笔刷类型，逗号分隔")
    parser.add_argument("--stage", "-s", type=int, default=None, help="仅运行指定阶段（1-3）")
    parser.add_argument("--device", "-d", type=str, default=None, help="设备（cuda/cpu/cuda:0）")
    parser.add_argument("--num-strokes", "-n", type=int, default=None, help="笔画数量")
    parser.add_argument("--iterations", type=int, default=None, help="每阶段迭代次数")
    parser.add_argument("--resolution", type=int, default=None, help="渲染分辨率")
    parser.add_argument("--export-svg", action="store_true", default=True, help="导出 SVG")
    parser.add_argument("--export-json", action="store_true", default=True, help="导出 JSON")
    parser.add_argument("--export-html", action="store_true", default=True, help="导出 HTML 回放器")
    parser.add_argument("--no-svg", action="store_true", help="不导出 SVG")
    parser.add_argument("--no-json", action="store_true", help="不导出 JSON")
    parser.add_argument("--no-html", action="store_true", help="不导出 HTML 回放器")
    parser.add_argument("--preview", action="store_true", help="预览模式（少量迭代）")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    if config_path is None:
        # 使用默认配置
        default_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 确定设备
    if args.device:
        device = args.device
    elif config.get("device"):
        device = config["device"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"PaintsUndo-Vector v0.1.0")
    print(f"设备: {device}")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 画布设置
    canvas_config = config.get("canvas", {})
    canvas_width = canvas_config.get("width", 640)
    canvas_height = canvas_config.get("height", 480)
    canvas_size = (canvas_width, canvas_height)
    
    # 加载目标图像
    target_image = load_image(args.input, canvas_size, device)
    print(f"目标图像加载完成: {canvas_size}")
    
    # 构建阶段配置
    stage_configs = []
    stages_config = config.get("stages", [])
    
    for i, sc in enumerate(stages_config):
        stage = StageConfig(
            name=sc.get("name", f"Stage {i+1}"),
            num_strokes=args.num_strokes or sc.get("num_strokes", 20),
            max_width=sc.get("max_width", 50.0),
            min_width=sc.get("min_width", 20.0),
            num_control_points=sc.get("num_control_points", 5),
            resolution=tuple(sc.get("resolution", [256, 256])),
            num_iterations=args.iterations or sc.get("num_iterations", 500),
            lr=sc.get("lr", 1.0),
            lr_width=sc.get("lr_width", 0.1),
            lr_color=sc.get("lr_color", 0.01),
            brush_names=args.brush.split(",") if args.brush else sc.get("brush_names", ["marker"]),
            lambda_pixel=sc.get("lambda_pixel", 1.0),
            lambda_perceptual=sc.get("lambda_perceptual", 10.0),
            lambda_length=sc.get("lambda_length", 0.01),
            lambda_smooth=sc.get("lambda_smooth", 0.005),
        )
        stage_configs.append(stage)
    
    # 预览模式
    if args.preview:
        for sc in stage_configs:
            sc.num_iterations = 50
            sc.num_strokes = max(5, sc.num_strokes // 4)
    
    # 仅运行指定阶段
    if args.stage is not None:
        stage_idx = args.stage - 1
        if 0 <= stage_idx < len(stage_configs):
            stage_configs = [stage_configs[stage_idx]]
        else:
            print(f"错误: 阶段 {args.stage} 不存在（可用: 1-{len(stage_configs)}）")
            sys.exit(1)
    
    # 运行优化
    optimizer = MultiStageOptimizer(
        canvas_size=canvas_size,
        device=device,
        output_dir=args.output,
    )
    
    result = optimizer.optimize(
        target_image=target_image,
        stage_configs=stage_configs,
    )
    
    # 导出结果
    all_strokes = result["strokes"]
    all_brush_names = result["brush_names"]
    
    # SVG 导出
    if args.export_svg and not args.no_svg:
        svg_path = os.path.join(args.output, "result.svg")
        exporter = SVGExporter(canvas_size=canvas_size)
        exporter.export(all_strokes, all_brush_names, svg_path, include_animation=True)
        print(f"SVG 导出完成: {svg_path}")
    
    # JSON 导出
    if args.export_json and not args.no_json:
        json_path = os.path.join(args.output, "strokes.json")
        json_exporter = JSONExporter()
        json_exporter.export(all_strokes, all_brush_names, json_path, canvas_size=canvas_size)
        print(f"JSON 导出完成: {json_path}")
    
    # HTML 回放器导出
    if args.export_html and not args.no_html:
        html_path = os.path.join(args.output, "replay.html")
        player = StrokePlayer(canvas_size=canvas_size)
        player.export_html_player(result["stages_history"], html_path)
        print(f"HTML 回放器导出完成: {html_path}")
    
    print("\n完成！所有结果已保存到:", args.output)


if __name__ == "__main__":
    main()
