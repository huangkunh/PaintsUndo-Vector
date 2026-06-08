"""
水彩笔刷 - 基于 glm 水彩工作方式

glm 水彩定义：
[10, "水彩", 2, render_func, {haveAlpha: true}]

核心特征：
- 水彩晕染效果，边缘有扩散
- 透明度叠加，模拟水彩的透明特性
- 笔画宽度随压力变化
- 边缘有水迹扩散效果

渲染逻辑（从 glm 源码提取）：
1. 基础笔画使用二次贝塞尔曲线
2. 在笔画周围添加高斯模糊模拟水彩扩散
3. 边缘添加噪声模拟水迹
4. 多层叠加模拟水彩的透明覆盖效果
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BaseBrush, BrushStroke


class WatercolorBrush(BaseBrush):
    """
    水彩笔刷
    
    基于 glm 的水彩渲染逻辑，具有晕染扩散效果。
    """
    
    brush_id = 10
    brush_name = "水彩"
    brush_type = 2
    have_alpha = True
    dis_collect_points = False
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
        spread_radius: float = 3.0,
        bleed_strength: float = 0.4,
    ):
        super().__init__(canvas_size=canvas_size, device=device)
        self.spread_radius = spread_radius
        self.bleed_strength = bleed_strength
    
    def render_stroke(
        self,
        stroke: BrushStroke,
        canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """渲染水彩笔画"""
        if canvas is None:
            canvas = torch.zeros(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
            canvas[3] = 0.0
        
        control_points = stroke.get_pixel_control_points()
        width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        rendered = self._render_watercolor_stroke(control_points, width, color, opacity)
        result = self._alpha_blend(rendered, canvas)
        
        return result
    
    def _render_watercolor_stroke(
        self,
        control_points: torch.Tensor,
        width: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        可微渲染水彩笔画。
        
        水彩效果通过以下步骤实现：
        1. 渲染基础笔画（类似马克笔）
        2. 对基础笔画进行高斯模糊（模拟扩散）
        3. 添加边缘噪声（模拟水迹）
        4. 多层叠加（模拟水彩透明覆盖）
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        # 创建像素坐标网格
        y_coords = torch.linspace(0, H - 1, H, device=self.device)
        x_coords = torch.linspace(0, W - 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixels = torch.stack([grid_x, grid_y], dim=-1)
        
        # 采样贝塞尔曲线
        num_samples = 64
        curve_points = self._sample_bezier_curve(control_points, num_samples)
        
        # 计算最近距离
        min_dist = self._compute_min_distance(pixels, curve_points)
        
        # 基础笔画 alpha
        half_width = width / 2.0
        base_alpha = torch.clamp((half_width - min_dist) / max(1.0, half_width * 0.1), 0.0, 1.0)
        
        # 水彩扩散：使用可微的高斯模糊
        # 将 alpha 转换为图像格式进行模糊
        alpha_img = base_alpha.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 创建高斯核
        kernel_size = int(self.spread_radius * 6) | 1  # 确保奇数
        sigma = self.spread_radius
        blurred_alpha = F.gaussian_blur(alpha_img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        blurred_alpha = blurred_alpha.squeeze(0).squeeze(0)  # [H, W]
        
        # 混合基础 alpha 和模糊 alpha
        # 水彩效果：中心浓，边缘淡且扩散
        watercolor_alpha = base_alpha * (1 - self.bleed_strength) + blurred_alpha * self.bleed_strength
        
        # 添加边缘噪声模拟水迹
        noise = torch.randn_like(watercolor_alpha) * 0.1
        watercolor_alpha = watercolor_alpha + noise * (1 - base_alpha) * self.bleed_strength
        watercolor_alpha = torch.clamp(watercolor_alpha, 0, 1)
        
        # 应用透明度
        alpha = watercolor_alpha * opacity * color[3]
        
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
    
    def get_svg_attributes(self, stroke: BrushStroke) -> Dict[str, str]:
        """水彩特有属性"""
        attrs = super().get_svg_attributes(stroke)
        # 水彩使用 SVG 滤镜模拟扩散效果
        attrs["filter"] = "url(#watercolor-filter)"
        return attrs
