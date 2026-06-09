#!/usr/bin/env python3
"""
PaintsUndo-Vector Gradio Web UI

功能：
- 上传图片并生成矢量笔画
- 实时预览优化过程
- 阶段可视化
- 参数预设（快速/标准/精细）
- 导出 SVG/JSON/HTML 回放
- 绘画过程叙述
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.optimizer import MultiStageOptimizer, StageConfig, DEFAULT_STAGE_CONFIGS
from core.renderer import DifferentiableRenderer
from core.painting_sim import sort_strokes_human_like, generate_painting_narrative
from export.svg_export import SVGExporter
from export.json_export import JSONExporter
from replay.player import StrokePlayer
from utils.image import load_image, save_image


# 参数预设
PRESETS = {
    "快速预览": {
        "num_strokes_s1": 16, "num_strokes_s2": 64, "num_strokes_s3": 256,
        "max_width_s1": 50.0, "max_width_s2": 15.0, "max_width_s3": 3.0,
        "num_iter": 200, "lr": 1.0,
    },
    "标准质量": {
        "num_strokes_s1": 32, "num_strokes_s2": 128, "num_strokes_s3": 512,
        "max_width_s1": 50.0, "max_width_s2": 15.0, "max_width_s3": 3.0,
        "num_iter": 500, "lr": 1.0,
    },
    "精细还原": {
        "num_strokes_s1": 64, "num_strokes_s2": 256, "num_strokes_s3": 1024,
        "max_width_s1": 50.0, "max_width_s2": 15.0, "max_width_s3": 3.0,
        "num_iter": 1000, "lr": 0.5,
    },
}


def run_optimization(
    input_image,
    preset,
    num_strokes_s1, num_strokes_s2, num_strokes_s3,
    max_width_s1, max_width_s2, max_width_s3,
    num_iter, lr,
    device_choice,
    enable_human_sort,
    progress=gr.Progress(),
):
    """运行优化流程"""
    if input_image is None:
        return None, None, None, None, "请先上传一张图片"
    
    progress(0, desc="初始化...")
    
    # 处理输入图像
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    input_image = input_image.convert("RGB")
    
    # 设备选择
    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice
    
    # 画布大小
    canvas_size = (640, 480)
    
    # 转换为张量
    img = input_image.resize((canvas_size[0], canvas_size[1]), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    target_image = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    
    # 创建阶段配置
    stage_configs = [
        StageConfig(
            name="铺底色",
            num_strokes=num_strokes_s1,
            max_width=max_width_s1,
            min_width=max_width_s1 * 0.4,
            num_control_points=5,
            resolution=(256, 256),
            num_iterations=num_iter,
            lr=lr,
            lr_width=lr * 0.1,
            lr_color=lr * 0.01,
            brush_names=["marker", "watercolor", "airbrush"],
        ),
        StageConfig(
            name="形体刻画",
            num_strokes=num_strokes_s2,
            max_width=max_width_s2,
            min_width=max_width_s2 * 0.3,
            num_control_points=7,
            resolution=(512, 512),
            num_iterations=int(num_iter * 1.6),
            lr=lr,
            lr_width=lr * 0.1,
            lr_color=lr * 0.01,
            brush_names=["pressure", "pressure_sharp", "marker"],
        ),
        StageConfig(
            name="细节线稿",
            num_strokes=num_strokes_s3,
            max_width=max_width_s3,
            min_width=max_width_s3 * 0.3,
            num_control_points=9,
            resolution=(1024, 1024),
            num_iterations=int(num_iter * 2),
            lr=lr * 0.5,
            lr_width=lr * 0.05,
            lr_color=lr * 0.005,
            brush_names=["pencil", "pressure_sharp", "hatching"],
        ),
    ]
    
    # 创建临时输出目录
    output_dir = tempfile.mkdtemp(prefix="paints_undo_")
    
    # 进度回调
    stage_previews = {}
    
    def progress_callback(stage_idx, iteration, total, loss_dict):
        pct = (stage_idx * total + iteration) / (len(stage_configs) * total)
        stage_name = stage_configs[stage_idx].name
        progress(
            min(pct, 0.99),
            desc=f"阶段 {stage_idx + 1}/3: {stage_name} - 迭代 {iteration}/{total}"
        )
    
    # 运行优化
    start_time = time.time()
    
    try:
        optimizer = MultiStageOptimizer(
            canvas_size=canvas_size,
            device=device,
            output_dir=output_dir,
        )
        
        result = optimizer.optimize(
            target_image=target_image,
            stage_configs=stage_configs,
            progress_callback=progress_callback,
        )
        
        # 人类绘画排序
        all_strokes = result["strokes"]
        all_brush_names = result["brush_names"]
        
        if enable_human_sort:
            all_strokes, all_brush_names = sort_strokes_human_like(
                all_strokes, all_brush_names, target_image
            )
        
        # 生成最终渲染
        progress(0.95, desc="生成最终结果...")
        
        final_renderer = DifferentiableRenderer(canvas_size=canvas_size, device=device)
        stroke_groups = final_renderer._group_strokes_by_brush(all_strokes, all_brush_names)
        final_rendered = final_renderer.render_multi_brush(stroke_groups)
        
        # 转换为 PIL 图像
        rendered_np = final_rendered.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        rendered_np = (rendered_np * 255).astype(np.uint8)
        output_pil = Image.fromarray(rendered_np)
        
        # 导出 SVG
        svg_path = os.path.join(output_dir, "result.svg")
        svg_exporter = SVGExporter(canvas_size=canvas_size)
        svg_exporter.export(
            all_strokes, all_brush_names, svg_path,
            include_animation=True,
            stages_history=result.get("stages_history"),
        )
        
        # 导出 JSON
        json_path = os.path.join(output_dir, "strokes.json")
        json_exporter = JSONExporter()
        json_exporter.export(
            all_strokes, all_brush_names, json_path,
            canvas_size=canvas_size,
        )
        
        # 导出 HTML 回放
        html_path = os.path.join(output_dir, "replay.html")
        player = StrokePlayer(canvas_size=canvas_size)
        player.export_html_player(result["stages_history"], html_path)
        
        elapsed = time.time() - start_time
        
        # 统计信息
        stats = (
            f"完成！耗时: {elapsed:.1f}秒\n"
            f"设备: {device}\n"
            f"总笔画数: {len(all_strokes)}\n"
            f"  - 底色: {num_strokes_s1} 条\n"
            f"  - 刻画: {num_strokes_s2} 条\n"
            f"  - 细节: {num_strokes_s3} 条\n"
            f"笔刷类型: {', '.join(set(all_brush_names))}\n"
            f"画布大小: {canvas_size[0]}×{canvas_size[1]}\n"
            f"输出目录: {output_dir}"
        )
        
        progress(1.0, desc="完成！")
        
        return output_pil, svg_path, json_path, html_path, stats
        
    except Exception as e:
        import traceback
        error_msg = f"错误: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, None, error_msg


def apply_preset(preset_name):
    """应用参数预设"""
    if preset_name in PRESETS:
        p = PRESETS[preset_name]
        return (
            p["num_strokes_s1"], p["num_strokes_s2"], p["num_strokes_s3"],
            p["max_width_s1"], p["max_width_s2"], p["max_width_s3"],
            p["num_iter"], p["lr"],
        )
    return gr.update()


# ===================== Gradio UI =====================

with gr.Blocks(
    title="PaintsUndo-Vector",
    theme=gr.themes.Soft(),
    css="""
    .main-title { text-align: center; margin-bottom: 10px; }
    .preset-btn { min-width: 100px; }
    """
) as demo:
    
    gr.Markdown(
        "# 🎨 PaintsUndo-Vector\n"
        "基于可微渲染优化的矢量笔画生成工具 — 输入图片，输出人类画家风格的矢量笔画\n\n"
        "上传一张图片，AI 会模拟人类画家的绘画过程，生成可编辑的矢量笔画。"
    )
    
    with gr.Row():
        # 左侧：输入
        with gr.Column(scale=1):
            gr.Markdown("### 输入")
            input_image = gr.Image(label="上传图片", type="pil")
            
            gr.Markdown("### 参数预设")
            with gr.Row():
                for preset_name in PRESETS:
                    gr.Button(
                        preset_name, variant="secondary", elem_classes=["preset-btn"]
                    ).click(
                        fn=lambda n=preset_name: apply_preset(n),
                        outputs=[
                            num_strokes_s1, num_strokes_s2, num_strokes_s3,
                            max_width_s1, max_width_s2, max_width_s3,
                            num_iter, lr,
                        ] if False else []  # 将在下面定义后连接
                    )
            
            gr.Markdown("### 参数设置")
            with gr.Accordion("阶段1: 铺底色", open=True):
                num_strokes_s1 = gr.Slider(8, 128, value=32, step=4, label="笔画数量")
                max_width_s1 = gr.Slider(10, 100, value=50.0, step=5, label="最大笔宽")
            
            with gr.Accordion("阶段2: 形体刻画", open=True):
                num_strokes_s2 = gr.Slider(32, 512, value=128, step=8, label="笔画数量")
                max_width_s2 = gr.Slider(3, 30, value=15.0, step=1, label="最大笔宽")
            
            with gr.Accordion("阶段3: 细节线稿", open=True):
                num_strokes_s3 = gr.Slider(128, 2048, value=512, step=16, label="笔画数量")
                max_width_s3 = gr.Slider(0.5, 10, value=3.0, step=0.5, label="最大笔宽")
            
            with gr.Accordion("通用设置", open=True):
                num_iter = gr.Slider(100, 2000, value=500, step=50, label="每阶段迭代次数")
                lr = gr.Slider(0.01, 2.0, value=1.0, step=0.01, label="学习率")
                device_choice = gr.Radio(
                    ["auto", "cpu", "cuda"],
                    value="auto",
                    label="计算设备"
                )
                enable_human_sort = gr.Checkbox(
                    value=True,
                    label="人类绘画排序（让笔画顺序更像人类画家）"
                )
            
            run_button = gr.Button("🚀 开始生成", variant="primary", size="lg")
        
        # 右侧：输出
        with gr.Column(scale=1):
            gr.Markdown("### 输出")
            output_image = gr.Image(label="渲染结果")
            
            with gr.Row():
                svg_file = gr.File(label="SVG 矢量文件")
                json_file = gr.File(label="JSON 笔画数据")
            
            html_file = gr.File(label="HTML 回放器")
            
            stats_text = gr.Textbox(label="统计信息", lines=8, max_lines=12)
    
    # 预设按钮连接
    preset_buttons = []
    with gr.Row():
        for preset_name in PRESETS:
            btn = gr.Button(preset_name, variant="secondary")
            btn.click(
                fn=lambda n=preset_name: apply_preset(n),
                outputs=[
                    num_strokes_s1, num_strokes_s2, num_strokes_s3,
                    max_width_s1, max_width_s2, max_width_s3,
                    num_iter, lr,
                ]
            )
            preset_buttons.append(btn)
    
    # 运行按钮
    run_button.click(
        fn=run_optimization,
        inputs=[
            input_image,
            gr.State("标准质量"),  # preset
            num_strokes_s1, num_strokes_s2, num_strokes_s3,
            max_width_s1, max_width_s2, max_width_s3,
            num_iter, lr, device_choice, enable_human_sort,
        ],
        outputs=[output_image, svg_file, json_file, html_file, stats_text]
    )
    
    gr.Markdown(
        "## 使用说明\n"
        "1. 上传一张图片（PNG/JPG）\n"
        "2. 选择参数预设或手动调整参数\n"
        "3. 点击「开始生成」\n"
        "4. 等待优化完成（可能需要几分钟到几十分钟）\n"
        "5. 下载 SVG 矢量文件、JSON 笔画数据或 HTML 回放器\n\n"
        "## 提示\n"
        "- **快速预览**：适合测试，质量较低但速度快\n"
        "- **标准质量**：平衡质量和速度\n"
        "- **精细还原**：最高质量，但需要较长时间\n"
        "- 阶段1笔画少、宽度大（铺底色）\n"
        "- 阶段2中等笔画（形体刻画）\n"
        "- 阶段3大量细笔画（细节线稿）\n"
        "- 更多迭代 = 更高质量但更慢\n"
        "- 使用 GPU 可大幅加速"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
