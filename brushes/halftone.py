"""
网点笔刷 - 基于 enazo 网点工作方式

enazo 网点定义：
[32, "💊", 2, render_func, {haveAlpha: true}]
[33, "网点", 2, render_func, {haveAlpha: true}]
[34, "网点马克", 2, render_func, {haveAlpha: true}]
[35, "网点填色", 2, render_func, {disCollectPoints: true}]

核心特征：
- 半色调网点效果
- 圆点阵列填充
- 模拟印刷网点效果
- 适合漫画/插画风格
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np

from brushes.base import BaseBrush, BrushStroke


class HalftoneBrush(BaseBrush):
    """
    网点笔刷
    
    基于 enazo 的网点渲染逻辑，半色调网点效果。
    """
    
    brush_id = 33
    brush_name = "网点"
    brush_type = 2
    have_alpha = True
    dis_collect_points = False
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
        dot_spacing: float = 8.0,
        max_dot_radius: float = 3.0,
    ):
        super().__init__(canvas_size=canvas_size, device=device)
        self.dot_spacing = dot_spacing
        self.max_dot_radius = max_dot_radius
    
    def render_stroke(
        self,
        stroke: BrushStroke,
        canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """渲染网点笔画"""
        if canvas is None:
            canvas = torch.zeros(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
            canvas[3] = 0.0
        
        control_points = stroke.get_pixel_control_points()
        width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        rendered = self._render_halftone_stroke(control_points, width, color, opacity)
        result = self._alpha_blend(rendered, canvas)
        
        return result
    
    def _render_halftone_stroke(
        self,
        control_points: torch.Tensor,
        width: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        可微渲染网点笔画。
        
        在笔画范围内生成半色调网点阵列。
        网点大小随位置变化，模拟印刷效果。
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        # 采样贝塞尔曲线
        num_samples = 64
        curve_points = self._sample_bezier_curve(control_points, num_samples)
        
        # 计算到中心线的距离
        y_coords = torch.linspace(0, H - 1, H, device=self.device)
        x_coords = torch.linspace(0, W - 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixels = torch.stack([grid_x, grid_y], dim=-1)
        
        center_dist = self._compute_min_distance(pixels, curve_points)
        
        # 笔画范围掩码
        half_width = width / 2.0
        stroke_mask = torch.clamp((half_width - center_dist) / max(1.0, half_width * 0.1), 0.0, 1.0)
        
        # 生成网点阵列
        # 计算每个像素到最近网格点的距离
        spacing = self.dot_spacing
        grid_x_idx = torch.round(grid_x / spacing) * spacing
        grid_y_idx = torch.round(grid_y / spacing) * spacing
        
        # 到最近网格点的距离
        dot_dist = torch.sqrt((grid_x - grid_x_idx) ** 2 + (grid_y - grid_y_idx) ** 2)
        
        # 网点半径（可以根据颜色深度变化）
        dot_radius = self.max_dot_radius
        
        # 网点 alpha
        dot_alpha = torch.clamp((dot_radius - dot_dist) / max(0.5, dot_radius * 0.1), 0.0, 1.0)
        
        # 在笔画范围内应用网点
        alpha = dot_alpha * stroke_mask * opacity * color[3]
        
        # 构建渲染结果
        rendered = torch.zeros(4, H, W, device=self.device)
        rendered[0] = color[0] * alpha
        rendered[1] = color[1] * alpha
        rendered[2] = color[2] * alpha
        rendered[3] = alpha
        
        return rendered
    
    def _sample_bezier_curve(self, control_points, num_samples=64):
        """采样贝塞尔曲线"""
        n = control_points.shape[0]
        if n < 2:
            return control_points
        
        t = torch.linspace(0, 1, num_samples, device=self.device).unsqueeze(1)
        points = control_points.unsqueeze(0).expand(num_samples, -1, -1).clone()
        
        for level in range(n - 1, 0, -1):
            for i in range(level):
                points[:, i, :] = (1 - t) * points[:, i, :] + t * points[:, i + 1, :]
        
        return points[:, 0, :]
    
    def _compute_min_distance(self, pixels, curve_points):
        """计算最近距离"""
        H, W, _ = pixels.shape
        flat_pixels = pixels.reshape(-1, 2)
        
        chunk_size = 4096
        min_dists = []
        
        for start in range(0, flat_pixels.shape[0], chunk_size):
            end = min(start + chunk_size, flat_pixels.shape[0])
            chunk = flat_pixels[start:end]
            dists = torch.cdist(chunk.unsqueeze(0), curve_points.unsqueeze(0)).squeeze(0)
            min_dist = dists.min(dim=1)[0]
            min_dists.append(min_dist)
        
        return torch.cat(min_dists).reshape(H, W)
    
    def _alpha_blend(self, foreground, background):
        """Alpha blending"""
        alpha_fg = foreground[3:4]
        alpha_bg = background[3:4]
        
        out_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
        out_alpha = torch.clamp(out_alpha, 0, 1)
        safe_alpha = torch.where(out_alpha > 1e-6, out_alpha, torch.ones_like(out_alpha))
        
        out_rgb = (foreground[:3] * alpha_fg + background[:3] * alpha_bg * (1 - alpha_fg)) / safe_alpha
        
        result = torch.cat([out_rgb, out_alpha], dim=0)
        return torch.clamp(result, 0, 1)
