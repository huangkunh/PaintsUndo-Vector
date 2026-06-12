#!/usr/bin/env python3
"""
PaintsUndo-Vector 端到端运行脚本

使用方法:
    python run_test.py                          # 使用默认测试图片
    python run_test.py --input photo.png        # 使用自定义图片
    python run_test.py --input photo.png --size 256  # 指定画布大小
    python run_test.py --strokes 8 20 40        # 指定每阶段笔画数
    python run_test.py --iters 50 80 100        # 指定每阶段迭代数
"""

import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image, ImageDraw

from core.renderer import DifferentiableRenderer
from core.losses import PixelLoss
from brushes.base import BrushStroke
from export.svg_export import SVGExporter
from export.json_export import JSONExporter
from utils.image import load_image, save_image
from utils.init import initialize_strokes_stage1, initialize_strokes_stage2, initialize_strokes_stage3


def create_test_image(path: str, size: tuple):
    """创建测试图片"""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    w, h = size
    # 天空
    for y in range(h // 2):
        r = int(135 + (200 - 135) * y / (h // 2))
        g = int(206 + (220 - 206) * y / (h // 2))
        b = int(235 + (255 - 235) * y / (h // 2))
        draw.line([(0, y), (w - 1, y)], fill=(r, g, b))
    
    # 太阳
    sx, sy = int(w * 0.75), int(h * 0.15)
    sr = int(w * 0.08)
    draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 200, 50))
    
    # 山
    draw.polygon([(0, h * 0.6), (w * 0.3, h * 0.3), (w * 0.6, h * 0.6)], fill=(100, 140, 80))
    draw.polygon([(w * 0.3, h * 0.6), (w * 0.7, h * 0.25), (w, h * 0.6)], fill=(80, 120, 60))
    
    # 草地
    draw.rectangle([0, int(h * 0.6), w, h], fill=(80, 160, 60))
    
    # 房子
    hx, hy = int(w * 0.35), int(h * 0.45)
    hw, hh = int(w * 0.2), int(h * 0.15)
    draw.rectangle([hx, hy, hx + hw, hy + hh], fill=(180, 80, 60))
    draw.polygon([(hx - 5, hy), (hx + hw // 2, hy - hh // 2), (hx + hw + 5, hy)], fill=(160, 60, 40))
    draw.rectangle([hx + hw // 3, hy + hh // 2, hx + hw * 2 // 3, hy + hh], fill=(100, 60, 30))
    
    # 树
    tx, ty = int(w * 0.8), int(h * 0.5)
    draw.rectangle([tx - 3, ty, tx + 3, ty + int(h * 0.1)], fill=(100, 70, 30))
    draw.ellipse([tx - int(w * 0.06), ty - int(h * 0.1), tx + int(w * 0.06), ty + int(h * 0.02)], fill=(40, 130, 40))
    
    img.save(path)
    return img


def main():
    parser = argparse.ArgumentParser(description="PaintsUndo-Vector 运行脚本")
    parser.add_argument("--input", "-i", type=str, default=None, help="输入图片路径")
    parser.add_argument("--output", "-o", type=str, default="test_output", help="输出目录")
    parser.add_argument("--size", type=int, default=64, help="画布大小 (正方形)")
    parser.add_argument("--strokes", type=int, nargs=3, default=[6, 12, 20], help="每阶段笔画数")
    parser.add_argument("--iters", type=int, nargs=3, default=[30, 50, 60], help="每阶段迭代数")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cpu/cuda/auto)")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    canvas_size = (args.size, args.size)
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 50)
    print("PaintsUndo-Vector 运行")
    print("=" * 50)
    print(f"设备: {device}")
    print(f"画布: {canvas_size}")
    print(f"笔画数: {args.strokes}")
    print(f"迭代数: {args.iters}")
    
    # 加载或创建图片
    if args.input:
        from PIL import Image as PILImage
        img = PILImage.open(args.input).convert("RGB")
        img = img.resize(canvas_size, PILImage.LANCZOS)
        img.save(os.path.join(args.output, "input.png"))
        input_path = os.path.join(args.output, "input.png")
    else:
        input_path = os.path.join(args.output, "test_input.png")
        create_test_image(input_path, canvas_size)
    
    target = load_image(input_path, canvas_size, device)
    save_image(target, os.path.join(args.output, "target.png"))
    print(f"✓ 目标图像: {target.shape}")
    
    renderer = DifferentiableRenderer(canvas_size=canvas_size, device=device)
    pixel_loss = PixelLoss("l1")
    
    t0 = time.time()
    
    # Stage 1
    print("\n--- Stage 1: 铺底色 ---")
    s1, b1 = initialize_strokes_stage1(
        target, num_strokes=args.strokes[0], max_width=args.size*0.4, min_width=args.size*0.15,
        num_control_points=4, canvas_size=canvas_size, device=device
    )
    o1 = torch.optim.Adam([p for s in s1 for p in s.parameters()], lr=1.0)
    bl1, bc1 = 999, None
    for it in range(args.iters[0]):
        o1.zero_grad()
        c = renderer.render_strokes(s1, brush_name=b1[0])
        l = pixel_loss(c, target)
        l.backward(); o1.step()
        if l.item() < bl1: bl1 = l.item(); bc1 = c.detach().clone()
        if it % max(1, args.iters[0] // 5) == 0:
            print(f"  iter {it:3d}: loss={l.item():.4f}")
    save_image(bc1, os.path.join(args.output, "stage1.png"))
    print(f"✓ Stage 1: loss={bl1:.4f}")
    
    # Stage 2
    print("\n--- Stage 2: 形体刻画 ---")
    for s in s1:
        for p in s.parameters(): p.requires_grad = False
    s2, b2 = initialize_strokes_stage2(
        target, num_strokes=args.strokes[1], max_width=args.size*0.1, min_width=args.size*0.03,
        num_control_points=5, canvas_size=canvas_size, device=device
    )
    o2 = torch.optim.Adam([p for s in s2 for p in s.parameters()], lr=1.0)
    bl2, bc2 = 999, None
    for it in range(args.iters[1]):
        o2.zero_grad()
        c = renderer.render_strokes(s1 + s2, brush_name=(b1 + b2)[0])
        l = pixel_loss(c, target)
        l.backward(); o2.step()
        if l.item() < bl2: bl2 = l.item(); bc2 = c.detach().clone()
        if it % max(1, args.iters[1] // 5) == 0:
            print(f"  iter {it:3d}: loss={l.item():.4f}")
    save_image(bc2, os.path.join(args.output, "stage2.png"))
    print(f"✓ Stage 2: loss={bl2:.4f}")
    
    # Stage 3
    print("\n--- Stage 3: 细节线稿 ---")
    for s in s2:
        for p in s.parameters(): p.requires_grad = False
    s3, b3 = initialize_strokes_stage3(
        target, num_strokes=args.strokes[2], max_width=args.size*0.02, min_width=args.size*0.005,
        num_control_points=6, canvas_size=canvas_size, device=device
    )
    o3 = torch.optim.Adam([p for s in s3 for p in s.parameters()], lr=0.5)
    bl3, bc3 = 999, None
    for it in range(args.iters[2]):
        o3.zero_grad()
        c = renderer.render_strokes(s1 + s2 + s3, brush_name=(b1 + b2 + b3)[0])
        l = pixel_loss(c, target)
        l.backward(); o3.step()
        if l.item() < bl3: bl3 = l.item(); bc3 = c.detach().clone()
        if it % max(1, args.iters[2] // 5) == 0:
            print(f"  iter {it:3d}: loss={l.item():.4f}")
    save_image(bc3, os.path.join(args.output, "final_result.png"))
    print(f"✓ Stage 3: loss={bl3:.4f}")
    
    # Export
    all_s, all_b = s1 + s2 + s3, b1 + b2 + b3
    SVGExporter(canvas_size).export(all_s, all_b, os.path.join(args.output, "result.svg"))
    JSONExporter().export(all_s, all_b, os.path.join(args.output, "strokes.json"), canvas_size=canvas_size)
    print("✓ SVG + JSON 导出完成")
    
    # Comparison chart
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
        plt.rcParams['font.sans-serif'] = ['Noto Serif SC', 'DejaVu Sans']
    except:
        pass
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, img_t) in zip(axes, [
        ('Target', target), ('Stage 1', bc1), ('Stage 2', bc2), ('Stage 3', bc3)
    ]):
        arr = img_t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(arr); ax.set_title(title, fontsize=14); ax.axis('off')
    plt.suptitle(f'PaintsUndo-Vector ({len(all_s)} strokes, loss: {bl3:.4f})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "progress.png"), dpi=150, bbox_inches='tight')
    print("✓ 对比图已保存")
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"完成! 总笔画: {len(all_s)}, 最终损失: {bl3:.4f}, 耗时: {elapsed:.1f}s")
    print(f"输出目录: {args.output}/")


if __name__ == "__main__":
    main()
