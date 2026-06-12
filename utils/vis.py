"""
可视化工具
"""

from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw


def visualize_strokes(
    strokes_data: List[dict],
    canvas_size: Tuple[int, int] = (640, 480),
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    可视化笔画（简化版，用于调试）
    
    Args:
        strokes_data: 笔画数据列表，每个包含 points, color, width, opacity
        canvas_size: 画布大小
        background_color: 背景颜色
    """
    img = Image.new("RGB", canvas_size, background_color)
    draw = ImageDraw.Draw(img)
    
    for stroke_data in strokes_data:
        points = stroke_data.get("points", [])
        color = stroke_data.get("color", (0, 0, 0))
        width = stroke_data.get("width", 2)
        
        if len(points) < 2:
            continue
        
        # 转换为像素坐标
        pixel_points = [(p[0] * canvas_size[0], p[1] * canvas_size[1]) for p in points]
        
        # 绘制线段
        for i in range(len(pixel_points) - 1):
            draw.line(
                [pixel_points[i], pixel_points[i + 1]],
                fill=color,
                width=max(1, int(width)),
            )
    
    return img


def create_comparison_image(
    target: torch.Tensor,
    rendered: torch.Tensor,
) -> Image.Image:
    """创建对比图像：目标 vs 渲染"""
    target_np = (target.permute(1, 2, 0).cpu().clamp(0, 1).detach().numpy() * 255).astype(np.uint8)
    rendered_np = (rendered.permute(1, 2, 0).cpu().clamp(0, 1).detach().numpy() * 255).astype(np.uint8)
    
    target_img = Image.fromarray(target_np)
    rendered_img = Image.fromarray(rendered_np)
    
    # 水平拼接
    w, h = target_img.size
    comparison = Image.new("RGB", (w * 2, h))
    comparison.paste(target_img, (0, 0))
    comparison.paste(rendered_img, (w, 0))
    
    return comparison
