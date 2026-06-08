"""
马克笔笔刷 - 基于 enazo 马克笔工作方式

enazo 马克笔定义：
[0, "马克笔", 2, render_func, {haveAlpha: true, disCollectPoints: true}]

核心特征：
- 使用二次贝塞尔曲线插值（quadraticCurveTo）
- 支持透明度叠加（haveAlpha: true）
- 禁用点收集（disCollectPoints: true），直接使用原始点序列
- globalAlpha 控制整体透明度

渲染逻辑（从 enazo 源码提取）：
1. save() 保存画布状态
2. 设置 globalAlpha = opacity
3. beginPath() + moveTo(points[0])
4. 如果只有2个点：lineTo(points[0])（单点）
5. 如果有4个点：lineTo(points[2], points[3])
6. 如果有更多点：使用 quadraticCurveTo 进行平滑插值
7. stroke() 绘制
8. restore() 恢复画布状态
"""

import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np

from brushes.base import BaseBrush, BrushStroke


class MarkerBrush(BaseBrush):
    """
    马克笔笔刷
    
    基于 enazo 的马克笔渲染逻辑，使用二次贝塞尔曲线进行平滑插值，
    支持透明度叠加效果。
    """
    
    brush_id = 0
    brush_name = "马克笔"
    brush_type = 2
    have_alpha = True
    dis_collect_points = True
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
    ):
        super().__init__(canvas_size=canvas_size, device=device)
    
    def render_stroke(
        self,
        stroke: BrushStroke,
        canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        渲染马克笔笔画。
        
        使用 pydiffvg 风格的可微渲染：
        1. 从控制点构建贝塞尔曲线路径
        2. 设置笔画宽度和颜色
        3. 渲染到画布上（alpha blending）
        """
        if canvas is None:
            canvas = torch.zeros(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
            canvas[3] = 0.0
        
        # 获取笔画参数
        control_points = stroke.get_pixel_control_points()  # [N, 2]
        width = stroke.width
        color = stroke.color  # [4] RGBA
        opacity = stroke.opacity
        
        # 使用可微渲染绘制贝塞尔曲线
        rendered = self._render_bezier_stroke(control_points, width, color, opacity)
        
        # Alpha blending 合成到画布
        result = self._alpha_blend(rendered, canvas)
        
        return result
    
    def _render_bezier_stroke(
        self,
        control_points: torch.Tensor,
        width: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        可微渲染贝塞尔曲线笔画。
        
        使用距离场方法实现可微的光栅化：
        1. 对每个像素计算到贝塞尔曲线的最近距离
        2. 根据距离和笔画宽度计算像素的 alpha 值
        3. 应用颜色和透明度
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        # 创建像素坐标网格
        y_coords = torch.linspace(0, H - 1, H, device=self.device)
        x_coords = torch.linspace(0, W - 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixels = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        
        # 采样贝塞尔曲线上的点
        num_samples = 64
        curve_points = self._sample_bezier_curve(control_points, num_samples)  # [num_samples, 2]
        
        # 计算每个像素到曲线的最近距离
        min_dist = self._compute_min_distance(pixels, curve_points)  # [H, W]
        
        # 使用 smoothstep 函数创建抗锯齿边缘
        half_width = width / 2.0
        # smoothstep: 平滑过渡边缘
        alpha = torch.clamp((half_width - min_dist) / max(1.0, half_width * 0.1), 0.0, 1.0)
        
        # 应用透明度
        alpha = alpha * opacity * color[3]
        
        # 构建渲染结果
        rendered = torch.zeros(4, H, W, device=self.device)
        rendered[0] = color[0] * alpha
        rendered[1] = color[1] * alpha
        rendered[2] = color[2] * alpha
        rendered[3] = alpha
        
        return rendered
    
    def _sample_bezier_curve(
        self,
        control_points: torch.Tensor,
        num_samples: int = 64,
    ) -> torch.Tensor:
        """
        采样贝塞尔曲线上的点。
        
        使用 de Casteljau 算法进行任意阶贝塞尔曲线求值。
        对应 enazo 中的 quadraticCurveTo 插值逻辑。
        """
        n = control_points.shape[0]
        if n < 2:
            return control_points
        
        t = torch.linspace(0, 1, num_samples, device=self.device).unsqueeze(1)  # [num_samples, 1]
        
        # de Casteljau 算法
        points = control_points.unsqueeze(0).expand(num_samples, -1, -1).clone()  # [num_samples, n, 2]
        
        for level in range(n - 1, 0, -1):
            for i in range(level):
                points[:, i, :] = (1 - t) * points[:, i, :] + t * points[:, i + 1, :]
        
        return points[:, 0, :]  # [num_samples, 2]
    
    def _compute_min_distance(
        self,
        pixels: torch.Tensor,
        curve_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每个像素到曲线点集的最近距离。
        
        为了内存效率，使用分块计算。
        """
        H, W, _ = pixels.shape
        num_points = curve_points.shape[0]
        
        # 展平像素坐标
        flat_pixels = pixels.reshape(-1, 2)  # [H*W, 2]
        
        # 分块计算以节省内存
        chunk_size = 4096
        min_dists = []
        
        for start in range(0, flat_pixels.shape[0], chunk_size):
            end = min(start + chunk_size, flat_pixels.shape[0])
            chunk = flat_pixels[start:end]  # [chunk, 2]
            
            # 计算距离 [chunk, num_points]
            dists = torch.cdist(chunk.unsqueeze(0), curve_points.unsqueeze(0)).squeeze(0)
            min_dist = dists.min(dim=1)[0]  # [chunk]
            min_dists.append(min_dist)
        
        return torch.cat(min_dists).reshape(H, W)
    
    def _alpha_blend(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alpha blending 合成操作。
        
        对应 enazo 中的 globalAlpha 叠加逻辑：
        result = foreground * alpha_fg + background * (1 - alpha_fg)
        """
        alpha_fg = foreground[3:4]  # [1, H, W]
        alpha_bg = background[3:4]  # [1, H, W]
        
        # 标准_alpha blending
        out_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
        out_alpha = torch.clamp(out_alpha, 0, 1)
        
        # 避免除零
        safe_alpha = torch.where(out_alpha > 1e-6, out_alpha, torch.ones_like(out_alpha))
        
        out_rgb = (foreground[:3] * alpha_fg + background[:3] * alpha_bg * (1 - alpha_fg)) / safe_alpha
        
        result = torch.cat([out_rgb, out_alpha], dim=0)
        return torch.clamp(result, 0, 1)
    
    def to_svg_path(self, stroke: BrushStroke) -> str:
        """
        将马克笔笔画转换为 SVG 路径。
        
        使用二次贝塞尔曲线（Q 命令），对应 enazo 的 quadraticCurveTo。
        """
        cp = stroke.control_points.detach().cpu().numpy()
        w = stroke.canvas_width
        h = stroke.canvas_height
        
        points = cp.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        
        if len(points) < 2:
            return ""
        
        # 马克笔使用二次贝塞尔曲线插值
        path = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
        
        if len(points) == 2:
            path += f" L {points[1][0]:.2f},{points[1][1]:.2f}"
        else:
            # 使用二次贝塞尔曲线，每两个控制点一组
            i = 1
            while i < len(points) - 1:
                if i + 1 < len(points):
                    # Q 控制点 终点
                    path += f" Q {points[i][0]:.2f},{points[i][1]:.2f} {points[i+1][0]:.2f},{points[i+1][1]:.2f}"
                    i += 2
                else:
                    path += f" L {points[i][0]:.2f},{points[i][1]:.2f}"
                    i += 1
        
        return path
