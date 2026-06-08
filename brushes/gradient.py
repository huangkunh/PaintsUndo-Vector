"""
渐变笔刷 - 基于 enazo 渐变工作方式

enazo 渐变定义：
[39, "渐变", 2, render_func, {disCollectPoints: true}]
[40, "渐变背景", 2, render_func, {disCollectPoints: true, hide: true}]

核心特征：
- 线性渐变填充
- 从起点到终点的颜色渐变
- 支持透明度渐变（从有色到透明）
- 渐变背景使用 destination-over 合成模式

渲染逻辑（从 enazo 源码提取）：
1. 创建从起点到终点的线性渐变
2. 起点颜色为指定颜色（完全不透明）
3. 终点颜色为指定颜色（完全透明）
4. 使用 fillRect 填充整个画布
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np

from brushes.base import BaseBrush, BrushStroke


class GradientBrush(BaseBrush):
    """
    渐变笔刷
    
    基于 enazo 的渐变渲染逻辑，创建线性渐变效果。
    """
    
    brush_id = 39
    brush_name = "渐变"
    brush_type = 2
    have_alpha = False
    dis_collect_points = True
    
    def render_stroke(
        self,
        stroke: BrushStroke,
        canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """渲染渐变笔画"""
        if canvas is None:
            canvas = torch.zeros(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
            canvas[3] = 0.0
        
        control_points = stroke.get_pixel_control_points()
        color = stroke.color
        opacity = stroke.opacity
        
        rendered = self._render_gradient(control_points, color, opacity)
        result = self._alpha_blend(rendered, canvas)
        
        return result
    
    def _render_gradient(
        self,
        control_points: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        可微渲染线性渐变。
        
        从控制点起点到终点创建线性渐变：
        - 起点颜色：指定颜色（完全不透明）
        - 终点颜色：指定颜色（完全透明）
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        # 获取渐变起止点
        start_point = control_points[0]  # [2]
        end_point = control_points[-1]   # [2]
        
        # 计算渐变方向
        direction = end_point - start_point
        length = torch.sqrt((direction ** 2).sum() + 1e-8)
        direction = direction / length
        
        # 创建像素坐标网格
        y_coords = torch.linspace(0, H - 1, H, device=self.device)
        x_coords = torch.linspace(0, W - 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 计算每个像素在渐变方向上的投影
        dx = grid_x - start_point[0]
        dy = grid_y - start_point[1]
        projection = (dx * direction[0] + dy * direction[1]) / length
        projection = torch.clamp(projection, 0, 1)
        
        # 渐变 alpha：从1到0
        gradient_alpha = (1 - projection) * opacity * color[3]
        
        # 构建渲染结果
        rendered = torch.zeros(4, H, W, device=self.device)
        rendered[0] = color[0] * gradient_alpha
        rendered[1] = color[1] * gradient_alpha
        rendered[2] = color[2] * gradient_alpha
        rendered[3] = gradient_alpha
        
        return rendered
    
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
