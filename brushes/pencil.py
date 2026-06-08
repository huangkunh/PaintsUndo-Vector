"""
铅笔笔刷 - 基于 enazo 铅笔工作方式

enazo 铅笔定义：
[6, "铅笔", 2, render_func, {haveAlpha: true}]

核心特征：
- 细线条，模拟铅笔质感
- 使用二次贝塞尔曲线插值
- 支持透明度叠加
- 笔画宽度较细，通常 1-3 像素
- 边缘有轻微的粗糙感（通过噪声模拟）

渲染逻辑：
1. 与马克笔类似的贝塞尔曲线插值
2. 但宽度更细，边缘有铅笔特有的粗糙质感
3. 通过在距离场中添加噪声来模拟铅笔纹理
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np

from brushes.base import BaseBrush, BrushStroke


class PencilBrush(BaseBrush):
    """
    铅笔笔刷
    
    基于 enazo 的铅笔渲染逻辑，细线条 + 铅笔纹理效果。
    """
    
    brush_id = 6
    brush_name = "铅笔"
    brush_type = 2
    have_alpha = True
    dis_collect_points = False
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
        texture_strength: float = 0.3,
    ):
        super().__init__(canvas_size=canvas_size, device=device)
        self.texture_strength = texture_strength
    
    def render_stroke(
        self,
        stroke: BrushStroke,
        canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """渲染铅笔笔画"""
        if canvas is None:
            canvas = torch.zeros(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
            canvas[3] = 0.0
        
        control_points = stroke.get_pixel_control_points()
        width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        rendered = self._render_pencil_stroke(control_points, width, color, opacity)
        result = self._alpha_blend(rendered, canvas)
        
        return result
    
    def _render_pencil_stroke(
        self,
        control_points: torch.Tensor,
        width: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        可微渲染铅笔笔画。
        
        与马克笔类似，但：
        1. 宽度更细
        2. 边缘添加铅笔纹理噪声
        3. 透明度变化更自然
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        # 创建像素坐标网格
        y_coords = torch.linspace(0, H - 1, H, device=self.device)
        x_coords = torch.linspace(0, W - 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixels = torch.stack([grid_x, grid_y], dim=-1)
        
        # 采样贝塞尔曲线
        num_samples = 80  # 铅笔需要更多采样点以获得平滑细线
        curve_points = self._sample_bezier_curve(control_points, num_samples)
        
        # 计算最近距离
        min_dist = self._compute_min_distance(pixels, curve_points)
        
        # 铅笔纹理：在距离场中添加轻微噪声
        noise = torch.randn_like(min_dist) * self.texture_strength
        textured_dist = min_dist + noise * (width / 4.0)
        
        # 细线条渲染 - 更锐利的边缘
        half_width = width / 2.0
        alpha = torch.clamp((half_width - textured_dist) / max(0.5, half_width * 0.05), 0.0, 1.0)
        
        # 铅笔特有的透明度衰减
        alpha = alpha * opacity * color[3]
        
        # 构建渲染结果
        rendered = torch.zeros(4, H, W, device=self.device)
        rendered[0] = color[0] * alpha
        rendered[1] = color[1] * alpha
        rendered[2] = color[2] * alpha
        rendered[3] = alpha
        
        return rendered
    
    def _sample_bezier_curve(self, control_points, num_samples=80):
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
    
    def to_svg_path(self, stroke: BrushStroke) -> str:
        """铅笔使用直线段连接（更硬朗的线条感）"""
        cp = stroke.control_points.detach().cpu().numpy()
        w = stroke.canvas_width
        h = stroke.canvas_height
        
        points = cp.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        
        if len(points) < 2:
            return ""
        
        path = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
        for i in range(1, len(points)):
            path += f" L {points[i][0]:.2f},{points[i][1]:.2f}"
        
        return path
    
    def get_svg_attributes(self, stroke: BrushStroke) -> Dict[str, str]:
        """铅笔特有属性"""
        attrs = super().get_svg_attributes(stroke)
        attrs["stroke-linecap"] = "butt"  # 铅笔使用平头
        return attrs
