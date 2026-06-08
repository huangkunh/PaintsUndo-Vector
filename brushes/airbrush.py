"""
喷笔笔刷 - 基于 enazo 喷笔工作方式

enazo 喷笔定义：
[30, "喷笔v2", 2, render_func, {haveAlpha: true}]

核心特征：
- 喷雾效果，模拟气泵喷笔
- 使用高斯分布的随机点模拟喷雾
- 边缘柔和，渐变过渡
- 适合大面积色彩渲染

渲染逻辑（从 enazo 源码提取）：
1. 沿笔画路径生成采样点
2. 在每个采样点周围按高斯分布喷洒颜色粒子
3. 粒子密度随距离衰减
4. 多层叠加形成柔和的喷雾效果
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BaseBrush, BrushStroke


class AirbrushBrush(BaseBrush):
    """
    喷笔笔刷
    
    基于 enazo 的喷笔v2渲染逻辑，具有喷雾扩散效果。
    """
    
    brush_id = 30
    brush_name = "喷笔v2"
    brush_type = 2
    have_alpha = True
    dis_collect_points = False
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
        spray_density: float = 0.5,
        spray_radius: float = 20.0,
    ):
        super().__init__(canvas_size=canvas_size, device=device)
        self.spray_density = spray_density
        self.spray_radius = spray_radius
    
    def render_stroke(
        self,
        stroke: BrushStroke,
        canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """渲染喷笔笔画"""
        if canvas is None:
            canvas = torch.zeros(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
            canvas[3] = 0.0
        
        control_points = stroke.get_pixel_control_points()
        width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        rendered = self._render_airbrush_stroke(control_points, width, color, opacity)
        result = self._alpha_blend(rendered, canvas)
        
        return result
    
    def _render_airbrush_stroke(
        self,
        control_points: torch.Tensor,
        width: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        可微渲染喷笔笔画。
        
        喷笔效果通过以下步骤实现：
        1. 沿笔画路径采样中心点
        2. 对每个中心点，计算周围像素的高斯权重
        3. 累加所有中心点的贡献
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        # 采样贝塞尔曲线上的中心点
        num_centers = 32
        curve_points = self._sample_bezier_curve(control_points, num_centers)
        
        # 创建像素坐标网格
        y_coords = torch.linspace(0, H - 1, H, device=self.device)
        x_coords = torch.linspace(0, W - 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 计算喷雾半径
        spray_r = self.spray_radius * (width / 10.0)
        
        # 对每个中心点计算高斯权重
        alpha = torch.zeros(H, W, device=self.device)
        
        for i in range(num_centers):
            cx, cy = curve_points[i]
            # 计算到中心点的距离
            dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            
            # 高斯权重
            weight = torch.exp(-dist_sq / (2 * spray_r ** 2))
            alpha = alpha + weight
        
        # 归一化
        alpha = alpha / num_centers * self.spray_density
        
        # 应用透明度
        alpha = torch.clamp(alpha, 0, 1) * opacity * color[3]
        
        # 构建渲染结果
        rendered = torch.zeros(4, H, W, device=self.device)
        rendered[0] = color[0] * alpha
        rendered[1] = color[1] * alpha
        rendered[2] = color[2] * alpha
        rendered[3] = alpha
        
        return rendered
    
    def _sample_bezier_curve(self, control_points, num_samples=32):
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
