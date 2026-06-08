"""
可微渲染器 - 核心渲染管线（高质量纯 PyTorch 实现）

将笔画参数通过可微渲染管线转换为像素图像，
支持梯度回传以优化笔画参数。

渲染管线：
笔画参数 → 贝塞尔曲线构建 → 笔刷特效处理 → 距离场光栅化 → 画布合成 → 输出图像

关键技术：
1. 基于 De Casteljau 算法的可微贝塞尔曲线求值
2. 基于距离场（SDF）的可微光栅化
3. 抗锯齿：使用 smoothstep 替代硬阈值
4. 多层 alpha blending 模拟真实绘画叠加
5. 支持变宽笔画（压感笔刷）
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from brushes.base import BaseBrush, BrushStroke
from brushes import BRUSH_REGISTRY


class DifferentiableRenderer(nn.Module):
    """
    可微渲染器
    
    管理多种笔刷的渲染，将笔画参数列表渲染为最终图像。
    支持梯度回传，用于优化笔画参数。
    
    核心改进：
    - 使用距离场（SDF）实现高质量抗锯齿光栅化
    - 支持变宽笔画渲染
    - 多层合成模拟真实绘画叠加效果
    - 高效的批量渲染
    """
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        antialias: float = 2.0,
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.device = device
        self.background_color = background_color
        self.antialias = antialias  # 抗锯齿过渡宽度
        
        # 笔刷缓存
        self._brush_cache: Dict[str, BaseBrush] = {}
        
        # 预计算像素坐标网格
        self._init_coordinate_grid()
    
    def _init_coordinate_grid(self):
        """预计算像素坐标网格，避免重复计算"""
        W, H = self.canvas_size
        y_coords = torch.linspace(0, H - 1, H, device=self.device)
        x_coords = torch.linspace(0, W - 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('pixel_grid_x', grid_x)
        self.register_buffer('pixel_grid_y', grid_y)
        self.register_buffer('pixel_grid', torch.stack([grid_x, grid_y], dim=-1))  # [H, W, 2]
    
    def get_brush(self, brush_name: str) -> BaseBrush:
        """获取或创建笔刷实例"""
        if brush_name not in self._brush_cache:
            brush_cls = BRUSH_REGISTRY[brush_name]
            self._brush_cache[brush_name] = brush_cls(
                canvas_size=self.canvas_size,
                device=self.device,
            )
        return self._brush_cache[brush_name]
    
    def render_strokes(
        self,
        strokes: List[BrushStroke],
        brush_name: str = "marker",
    ) -> torch.Tensor:
        """
        渲染一组笔画为最终 RGB 图像。
        """
        brush = self.get_brush(brush_name)
        
        canvas = torch.ones(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
        canvas[3] = 1.0
        canvas[0] = self.background_color[0]
        canvas[1] = self.background_color[1]
        canvas[2] = self.background_color[2]
        
        for stroke in strokes:
            canvas = brush.render_stroke(stroke, canvas)
        
        return canvas[:3]
    
    def render_strokes_with_alpha(
        self,
        strokes: List[BrushStroke],
        brush_name: str = "marker",
    ) -> torch.Tensor:
        """渲染一组笔画，返回 RGBA 图像"""
        brush = self.get_brush(brush_name)
        
        canvas = torch.ones(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
        canvas[3] = 1.0
        canvas[0] = self.background_color[0]
        canvas[1] = self.background_color[1]
        canvas[2] = self.background_color[2]
        
        for stroke in strokes:
            canvas = brush.render_stroke(stroke, canvas)
        
        return canvas
    
    def render_multi_brush(
        self,
        stroke_groups: List[Tuple[str, List[BrushStroke]]],
    ) -> torch.Tensor:
        """
        使用多种笔刷渲染笔画组。
        """
        canvas = torch.ones(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
        canvas[3] = 1.0
        canvas[0] = self.background_color[0]
        canvas[1] = self.background_color[1]
        canvas[2] = self.background_color[2]
        
        for brush_name, strokes in stroke_groups:
            brush = self.get_brush(brush_name)
            for stroke in strokes:
                canvas = brush.render_stroke(stroke, canvas)
        
        return canvas[:3]
    
    # ==================== 可微渲染核心函数 ====================
    
    def de_casteljau(self, control_points: torch.Tensor, num_samples: int = 64) -> torch.Tensor:
        """
        De Casteljau 算法求值贝塞尔曲线（可微）。
        
        Args:
            control_points: [N, 2] 控制点
            num_samples: 采样点数
            
        Returns:
            [num_samples, 2] 曲线上的采样点
        """
        n = control_points.shape[0]
        if n < 2:
            return control_points.expand(num_samples, -1).clone()
        
        t = torch.linspace(0, 1, num_samples, device=control_points.device).unsqueeze(1)
        
        # De Casteljau 递推
        points = control_points.unsqueeze(0).expand(num_samples, -1, -1).clone()
        
        for level in range(n - 1, 0, -1):
            for i in range(level):
                points[:, i, :] = (1 - t) * points[:, i, :] + t * points[:, i + 1, :]
        
        return points[:, 0, :]
    
    def compute_sdf_to_curve(
        self,
        curve_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算像素网格到曲线的带符号距离场（SDF）。
        
        使用分块计算避免内存溢出。
        
        Args:
            curve_points: [M, 2] 曲线采样点（像素坐标）
            
        Returns:
            [H, W] 距离场
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        pixels = self.pixel_grid  # [H, W, 2]
        
        # 分块计算以节省内存
        chunk_size = 8192
        flat_pixels = pixels.reshape(-1, 2)
        min_dists = []
        
        for start in range(0, flat_pixels.shape[0], chunk_size):
            end = min(start + chunk_size, flat_pixels.shape[0])
            chunk = flat_pixels[start:end]
            # [chunk, 1, 2] vs [1, M, 2] -> [chunk, M]
            dists = torch.cdist(chunk.unsqueeze(0).float(), curve_points.unsqueeze(0).float()).squeeze(0)
            min_dist = dists.min(dim=1)[0]
            min_dists.append(min_dist)
        
        return torch.cat(min_dists).reshape(H, W)
    
    def rasterize_stroke_sdf(
        self,
        stroke: BrushStroke,
        num_curve_samples: int = 64,
    ) -> torch.Tensor:
        """
        使用 SDF 方法光栅化单条笔画（可微）。
        
        核心渲染算法：
        1. 从控制点采样贝塞尔曲线
        2. 计算每个像素到曲线的距离
        3. 使用 smoothstep 函数生成抗锯齿的 alpha 值
        4. 应用笔画颜色和透明度
        
        Args:
            stroke: 笔画参数
            num_curve_samples: 曲线采样点数
            
        Returns:
            [4, H, W] RGBA 渲染结果
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        # 获取笔画参数
        control_points = stroke.get_pixel_control_points()  # [N, 2]
        width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        # 采样贝塞尔曲线
        curve_points = self.de_casteljau(control_points, num_curve_samples)  # [M, 2]
        
        # 计算距离场
        dist = self.compute_sdf_to_curve(curve_points)  # [H, W]
        
        # 使用 smoothstep 抗锯齿
        half_width = width / 2.0
        # smoothstep(0, antialias, half_width - dist)
        # 当 dist < half_width - antialias 时 alpha = 1
        # 当 dist > half_width 时 alpha = 0
        # 中间平滑过渡
        alpha = torch.clamp((half_width - dist) / max(self.antialias, 0.5), 0.0, 1.0)
        
        # 应用透明度
        alpha = alpha * opacity * color[3]
        
        # 构建渲染结果
        rendered = torch.zeros(4, H, W, device=self.device)
        rendered[0] = color[0] * alpha
        rendered[1] = color[1] * alpha
        rendered[2] = color[2] * alpha
        rendered[3] = alpha
        
        return rendered
    
    def rasterize_variable_width_stroke(
        self,
        stroke: BrushStroke,
        pressure_fn=None,
        num_curve_samples: int = 64,
    ) -> torch.Tensor:
        """
        光栅化变宽笔画（用于压感笔刷）。
        
        沿曲线方向，笔画宽度根据压力函数变化。
        模拟真实绘画中笔压变化的效果。
        
        Args:
            stroke: 笔画参数
            pressure_fn: 压力函数 f(t) -> width_factor，t ∈ [0, 1]
            num_curve_samples: 曲线采样点数
            
        Returns:
            [4, H, W] RGBA 渲染结果
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        control_points = stroke.get_pixel_control_points()
        base_width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        # 采样贝塞尔曲线
        curve_points = self.de_casteljau(control_points, num_curve_samples)  # [M, 2]
        
        # 计算每个曲线点的局部宽度
        t_values = torch.linspace(0, 1, num_curve_samples, device=self.device)
        if pressure_fn is not None:
            width_factors = pressure_fn(t_values)
        else:
            # 默认钟形压力曲线
            width_factors = torch.sin(torch.pi * t_values)
        
        local_widths = base_width * width_factors  # [M]
        
        # 计算距离场
        dist = self.compute_sdf_to_curve(curve_points)  # [H, W]
        
        # 计算每个像素到最近曲线点的索引，以获取局部宽度
        # 使用近似方法：找到最近点后使用其宽度
        flat_pixels = self.pixel_grid.reshape(-1, 2)
        flat_curve = curve_points
        
        chunk_size = 8192
        min_dists = []
        nearest_indices = []
        
        for start in range(0, flat_pixels.shape[0], chunk_size):
            end = min(start + chunk_size, flat_pixels.shape[0])
            chunk = flat_pixels[start:end]
            dists = torch.cdist(chunk.unsqueeze(0).float(), flat_curve.unsqueeze(0).float()).squeeze(0)
            min_idx = dists.argmin(dim=1)
            min_dist = dists.gather(1, min_idx.unsqueeze(1)).squeeze(1)
            min_dists.append(min_dist)
            nearest_indices.append(min_idx)
        
        dist = torch.cat(min_dists).reshape(H, W)
        nearest_idx = torch.cat(nearest_indices).reshape(H, W)
        
        # 获取每个像素的局部宽度
        local_width_map = local_widths[nearest_idx]  # [H, W]
        half_width = local_width_map / 2.0
        
        # 抗锯齿
        alpha = torch.clamp((half_width - dist) / max(self.antialias, 0.5), 0.0, 1.0)
        alpha = alpha * opacity * color[3]
        
        rendered = torch.zeros(4, H, W, device=self.device)
        rendered[0] = color[0] * alpha
        rendered[1] = color[1] * alpha
        rendered[2] = color[2] * alpha
        rendered[3] = alpha
        
        return rendered
    
    @staticmethod
    def alpha_blend(foreground: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        """
        可微 alpha blending（Porter-Duff over 操作）。
        
        模拟真实绘画中颜料叠加的效果：
        - 新笔画覆盖旧笔画
        - 透明度决定覆盖程度
        - 颜色混合遵循物理模型
        
        Args:
            foreground: [4, H, W] RGBA 前景
            background: [4, H, W] RGBA 背景
            
        Returns:
            [4, H, W] RGBA 合成结果
        """
        alpha_fg = foreground[3:4]
        alpha_bg = background[3:4]
        
        # Porter-Duff over 操作
        out_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
        out_alpha = torch.clamp(out_alpha, 0, 1)
        
        # 避免除零
        safe_alpha = torch.where(out_alpha > 1e-6, out_alpha, torch.ones_like(out_alpha))
        
        # 预乘 alpha 颜色混合
        out_rgb = (foreground[:3] * alpha_fg + background[:3] * alpha_bg * (1 - alpha_fg)) / safe_alpha
        
        result = torch.cat([out_rgb, out_alpha], dim=0)
        return torch.clamp(result, 0, 1)
