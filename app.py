#!/usr/bin/env python3
"""
PaintsUndo-Vector Gradio Web UI
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.optimizer import MultiStageOptimizer, DEFAULT_STAGE_CONFIGS
from core.renderer import DifferentiableRenderer
from export.svg_export import SVGExporter
from export.json_export import JSONExporter
from utils.image import load_image, save_image


def run_optimization(
    input_image,
    num_strokes_stage1=32,
    num_strokes_stage2=128,
    num_strokes_stage3=512,
    max_width_stage1=50.0,
    max_width_stage2=15.0,
    max_width_stage3=3.0,
    num_iter=500,
    learning_rate=0.01,
    device_choice="auto",
    progress=gr.Progress()
):
    if input_image is None:
        return None, None, None, "Please upload an image first"

    progress(0, desc="Initializing...")

    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    input_image = input_image.convert("RGB")

    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice

    canvas_size = (640, 480)
    img_np = np.array(input_image).astype(np.float32) / 255.0
    target_image = torch.from_numpy(img_np).permute(2, 0, 1).to(device)

    try:
        output_dir = tempfile.mkdtemp(prefix="paints_undo_out_")

        optimizer = MultiStageOptimizer(
            canvas_size=canvas_size,
            device=device,
            output_dir=output_dir
        )

        progress(0.1, desc="Stage 1: Base colors...")
        result = optimizer.optimize(target_image=target_image)

        progress(0.9, desc="Generating results...")

        all_strokes = result["strokes"]
        all_brush_names = result["brush_names"]

        renderer = DifferentiableRenderer(canvas_size=canvas_size, device=device)
        stroke_groups = []
        brush_set = set(all_brush_names)
        for bname in brush_set:
            idxs = [i for i, bn in enumerate(all_brush_names) if bn == bname]
            stroke_groups.append((bname, [all_strokes[i] for i in idxs]))

        final_canvas = renderer.render_multi_brush(stroke_groups)
        final_image = final_canvas[:3]

        result_image_path = os.path.join(output_dir, "result.png")
        save_image(final_image, result_image_path)

        svg_path = os.path.join(output_dir, "result.svg")
        SVGExporter(canvas_size=canvas_size).export(
            all_strokes, all_brush_names, svg_path, include_filters=True
        )

        json_path = os.path.join(output_dir, "strokes.json")
        JSONExporter().export(
            all_strokes, all_brush_names, json_path, canvas_size=canvas_size
        )

        stats = (
            "Done!\n\n"
            "Stage 1 strokes: " + str(num_strokes_stage1) + "\n"
            "Stage 2 strokes: " + str(num_strokes_stage2) + "\n"
            "Stage 3 strokes: " + str(num_strokes_stage3) + "\n"
            "Total strokes: " + str(len(all_strokes)) + "\n"
            "Device: " + device + "\n\n"
            "SVG: " + svg_path + "\n"
            "JSON: " + json_path
        )

        progress(1.0, desc="Done!")
        result_image = Image.open(result_image_path).convert("RGB")
        return result_image, svg_path, json_path, stats

    except Exception as e:
        import traceback
        return None, None, None, "Error:\n" + traceback.format_exc()


with gr.Blocks(title="PaintsUndo-Vector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# PaintsUndo-Vector\n"
        "Vector stroke generation via differentiable rendering optimization\n\n"
        "Upload image -> Adjust params -> Generate strokes -> Export SVG/JSON"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            input_image = gr.Image(label="Upload Image", type="pil")

            gr.Markdown("## Parameters")
            with gr.Accordion("Stroke Count", open=True):
                num_strokes_stage1 = gr.Slider(10, 64, 32, step=2, label="Stage 1 strokes (base)")
                num_strokes_stage2 = gr.Slider(32, 256, 128, step=8, label="Stage 2 strokes (form)")
                num_strokes_stage3 = gr.Slider(64, 1024, 512, step=32, label="Stage 3 strokes (detail)")

            with gr.Accordion("Stroke Width", open=True):
                max_width_stage1 = gr.Slider(10, 80, 50, step=5, label="Stage 1 max width")
                max_width_stage2 = gr.Slider(3, 40, 15, step=1, label="Stage 2 max width")
                max_width_stage3 = gr.Slider(1, 10, 3, step=0.5, label="Stage 3 max width")

            with gr.Accordion("Optimization", open=True):
                num_iter = gr.Slider(100, 2000, 500, step=50, label="Iterations per stage")
                learning_rate = gr.Slider(0.001, 0.1, 0.01, step=0.001, label="Learning rate")

            with gr.Accordion("Device", open=True):
                device_choice = gr.Radio(["auto", "cpu", "cuda"], value="auto", label="Device")

            run_button = gr.Button("Start Generation", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## Output")
            output_image = gr.Image(label="Result", type="pil")

            with gr.Accordion("Export Files", open=True):
                svg_file = gr.File(label="SVG Vector File")
                json_file = gr.File(label="JSON Stroke Data")

            stats_text = gr.Textbox(label="Stats", lines=8, max_lines=12)

    run_button.click(
        fn=run_optimization,
        inputs=[
            input_image,
            num_strokes_stage1, num_strokes_stage2, num_strokes_stage3,
            max_width_stage1, max_width_stage2, max_width_stage3,
            num_iter, learning_rate, device_choice
        ],
        outputs=[output_image, svg_file, json_file, stats_text]
    )

    gr.Markdown(
        "## Instructions\n"
        "1. Upload an image (PNG/JPG)\n"
        "2. Adjust parameters or use defaults\n"
        "3. Click Start Generation\n"
        "4. Wait for optimization (may take several minutes)\n"
        "5. Download SVG or JSON files\n\n"
        "## Tips\n"
        "- Stage 1: fewer strokes, wider width (base colors)\n"
        "- Stage 2: medium strokes, medium width (forms)\n"
        "- Stage 3: many strokes, thin width (details)\n"
        "- More iterations = better quality but slower\n"
        "- Use GPU if available for faster processing"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
