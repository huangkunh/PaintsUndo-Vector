"""
可微渲染器 - 核心渲染管线（高质量纯 PyTorch 实现）

将笔画参数通过可微渲染管线转换为像素图像，
支持梯度回传以优化笔画参数。

渲染管线：
笔画参数 → 贝塞尔曲线采样 → 笔刷特效 → 距离场软光栅化 → 画布合成 → 输出图像

关键技术：
1. De Casteljau 算法的可微贝塞尔曲线求值
2. 基于距离场（SDF）的可微软光栅化
3. 自适应抗锯齿：根据笔画宽度自动调整过渡带
4. 多层 alpha blending 模拟真实绘画叠加
5. 支持变宽笔画（压感笔刷）
6. 高效批量渲染：一次性渲染所有笔画
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
        self.antialias = antialias
        
        self._brush_cache: Dict[str, BaseBrush] = {}
        self._init_coordinate_grid()
    
    def _init_coordinate_grid(self):
        """预计算像素坐标网格"""
        W, H = self.canvas_size
        y_coords = torch.linspace(0, H - 1, H, device=self.device)
        x_coords = torch.linspace(0, W - 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('_pixel_grid_x', grid_x)
        self.register_buffer('_pixel_grid_y', grid_y)
    
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
        """渲染一组笔画为最终 RGB 图像"""
        canvas = self._create_canvas()
        brush = self.get_brush(brush_name)
        for stroke in strokes:
            canvas = brush.render_stroke(stroke, canvas)
        return canvas[:3]
    
    def render_strokes_with_alpha(
        self,
        strokes: List[BrushStroke],
        brush_name: str = "marker",
    ) -> torch.Tensor:
        """渲染一组笔画，返回 RGBA 图像"""
        canvas = self._create_canvas()
        brush = self.get_brush(brush_name)
        for stroke in strokes:
            canvas = brush.render_stroke(stroke, canvas)
        return canvas
    
    def render_multi_brush(
        self,
        stroke_groups: List[Tuple[str, List[BrushStroke]]],
    ) -> torch.Tensor:
        """使用多种笔刷渲染笔画组"""
        canvas = self._create_canvas()
        for brush_name, strokes in stroke_groups:
            brush = self.get_brush(brush_name)
            for stroke in strokes:
                canvas = brush.render_stroke(stroke, canvas)
        return canvas[:3]
    
    def render_multi_brush_with_alpha(
        self,
        stroke_groups: List[Tuple[str, List[BrushStroke]]],
    ) -> torch.Tensor:
        """使用多种笔刷渲染笔画组，返回 RGBA"""
        canvas = self._create_canvas()
        for brush_name, strokes in stroke_groups:
            brush = self.get_brush(brush_name)
            for stroke in strokes:
                canvas = brush.render_stroke(stroke, canvas)
        return canvas
    
    def _create_canvas(self) -> torch.Tensor:
        """创建白色画布 [4, H, W]"""
        canvas = torch.ones(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
        canvas[0] = self.background_color[0]
        canvas[1] = self.background_color[1]
        canvas[2] = self.background_color[2]
        canvas[3] = 1.0
        return canvas
    
    # ===================== 高级渲染方法 =====================
    
    def render_stroke_soft(
        self,
        stroke: BrushStroke,
        canvas: torch.Tensor,
        brush_name: str = "marker",
    ) -> torch.Tensor:
        """
        使用软光栅化渲染单条笔画。
        
        相比硬光栅化，软光栅化使用 smoothstep 过渡，
        产生更平滑的边缘和更自然的抗锯齿效果。
        同时支持变宽笔画（压感笔刷）。
        """
        control_points = stroke.get_pixel_control_points()
        width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        # 采样贝塞尔曲线上的密集点
        num_samples = max(32, len(control_points) * 16)
        curve_points = self._sample_bezier_de_casteljau(control_points, num_samples)
        
        if curve_points.shape[0] < 2:
            return canvas
        
        # 计算每个像素到曲线的最近距离
        dist, nearest_idx = self._compute_min_distance_with_index(
            self._pixel_grid_x, self._pixel_grid_y, curve_points
        )
        
        # 计算局部宽度（支持变宽笔画）
        if brush_name in ("pressure", "pressure_sharp"):
            t_values = nearest_idx.float() / (num_samples - 1)
            local_widths = self._compute_variable_width(t_values, width, brush_name)
        else:
            local_widths = width.expand(H, W)
        
        half_width = local_widths / 2.0
        
        # 自适应抗锯齿过渡带
        aa_width = max(self.antialias, half_width * 0.15)
        
        # Smoothstep 软光栅化
        # alpha = 1 当 dist < half_width - aa_width
        # alpha = 0 当 dist > half_width + aa_width
        # alpha = smoothstep 当中间
        inner = half_width - aa_width
        outer = half_width + aa_width
        
        # smoothstep: 3t^2 - 2t^3
        t = torch.clamp((outer - dist) / (2 * aa_width + 1e-8), 0.0, 1.0)
        alpha = t * t * (3 - 2 * t)
        
        # 应用透明度
        alpha = alpha * opacity * color[3]
        
        # 构建前景
        fg = torch.zeros(4, H, W, device=self.device)
        fg[0] = color[0] * alpha
        fg[1] = color[1] * alpha
        fg[2] = color[2] * alpha
        fg[3] = alpha
        
        return self.alpha_blend(fg, canvas)
    
    def render_stroke_with_texture(
        self,
        stroke: BrushStroke,
        canvas: torch.Tensor,
        brush_name: str = "marker",
    ) -> torch.Tensor:
        """
        带纹理的笔画渲染。
        
        根据笔刷类型添加不同的纹理效果：
        - 铅笔：添加噪声纹理模拟铅笔粗糙感
        - 水彩：添加高斯模糊模拟水彩扩散
        - 喷笔：添加高斯喷雾效果
        """
        if brush_name == "pencil":
            return self._render_pencil_with_texture(stroke, canvas)
        elif brush_name == "watercolor":
            return self._render_watercolor_with_texture(stroke, canvas)
        elif brush_name == "airbrush":
            return self._render_airbrush_with_texture(stroke, canvas)
        else:
            return self.render_stroke_soft(stroke, canvas, brush_name)
    
    def _render_pencil_with_texture(
        self,
        stroke: BrushStroke,
        canvas: torch.Tensor,
    ) -> torch.Tensor:
        """铅笔纹理渲染：在软光栅化基础上添加噪声纹理"""
        # 先做基础软光栅化
        result = self.render_stroke_soft(stroke, canvas, "pencil")
        
        # 添加铅笔纹理噪声
        control_points = stroke.get_pixel_control_points()
        width = stroke.width
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        num_samples = max(32, len(control_points) * 16)
        curve_points = self._sample_bezier_de_casteljau(control_points, num_samples)
        
        if curve_points.shape[0] < 2:
            return result
        
        dist = self._compute_min_distance(
            self._pixel_grid_x, self._pixel_grid_y, curve_points
        )
        
        half_width = width / 2.0
        # 铅笔纹理：在笔画范围内添加随机噪声
        mask = (dist < half_width * 1.2).float()
        
        # 生成噪声纹理
        noise = torch.rand(H, W, device=self.device) * 0.3
        
        # 只在笔画范围内应用噪声
        texture_alpha = mask * noise * stroke.opacity * 0.15
        
        # 将噪声混合到结果中
        noise_fg = torch.zeros(4, H, W, device=self.device)
        noise_fg[3] = texture_alpha
        result = self.alpha_blend(noise_fg, result)
        
        return result
    
    def _render_watercolor_with_texture(
        self,
        stroke: BrushStroke,
        canvas: torch.Tensor,
    ) -> torch.Tensor:
        """水彩纹理渲染：在软光栅化基础上添加扩散效果"""
        # 先做基础软光栅化，但使用更宽的过渡带模拟水彩扩散
        control_points = stroke.get_pixel_control_points()
        width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        num_samples = max(32, len(control_points) * 16)
        curve_points = self._sample_bezier_de_casteljau(control_points, num_samples)
        
        if curve_points.shape[0] < 2:
            return canvas
        
        dist = self._compute_min_distance(
            self._pixel_grid_x, self._pixel_grid_y, curve_points
        )
        
        half_width = width / 2.0
        
        # 水彩扩散：更宽的过渡带 + 不规则边缘
        spread = half_width * 0.3  # 扩散范围
        aa_width = max(self.antialias * 2, spread)
        
        inner = half_width - aa_width
        outer = half_width + aa_width + spread
        
        t = torch.clamp((outer - dist) / (outer - inner + 1e-8), 0.0, 1.0)
        alpha = t * t * (3 - 2 * t)
        
        # 水彩不透明度较低
        alpha = alpha * opacity * color[3] * 0.7
        
        fg = torch.zeros(4, H, W, device=self.device)
        fg[0] = color[0] * alpha
        fg[1] = color[1] * alpha
        fg[2] = color[2] * alpha
        fg[3] = alpha
        
        return self.alpha_blend(fg, canvas)
    
    def _render_airbrush_with_texture(
        self,
        stroke: BrushStroke,
        canvas: torch.Tensor,
    ) -> torch.Tensor:
        """喷笔纹理渲染：高斯喷雾效果"""
        control_points = stroke.get_pixel_control_points()
        width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        num_samples = max(32, len(control_points) * 16)
        curve_points = self._sample_bezier_de_casteljau(control_points, num_samples)
        
        if curve_points.shape[0] < 2:
            return canvas
        
        dist = self._compute_min_distance(
            self._pixel_grid_x, self._pixel_grid_y, curve_points
        )
        
        # 喷笔使用高斯衰减
        sigma = width / 2.0
        alpha = torch.exp(-0.5 * (dist / sigma) ** 2)
        alpha = alpha * opacity * color[3] * 0.5
        
        fg = torch.zeros(4, H, W, device=self.device)
        fg[0] = color[0] * alpha
        fg[1] = color[1] * alpha
        fg[2] = color[2] * alpha
        fg[3] = alpha
        
        return self.alpha_blend(fg, canvas)
    
    def _compute_variable_width(
        self,
        t_values: torch.Tensor,
        base_width: torch.Tensor,
        brush_name: str,
    ) -> torch.Tensor:
        """计算变宽笔画的局部宽度"""
        if brush_name == "pressure":
            # 钟形压力曲线
            pressure = torch.sin(torch.pi * t_values)
        elif brush_name == "pressure_sharp":
            # 尖头压力曲线
            pressure = 1.0 - torch.exp(-3.0 * t_values)
        else:
            pressure = torch.ones_like(t_values)
        
        return pressure * base_width
    
    # ===================== 贝塞尔曲线工具 =====================
    
    @staticmethod
    def _sample_bezier_de_casteljau(
        control_points: torch.Tensor,
        num_samples: int = 64,
    ) -> torch.Tensor:
        """
        使用 De Casteljau 算法采样贝塞尔曲线。
        
        这是可微的，支持梯度回传。
        
        Args:
            control_points: [N, 2] 控制点
            num_samples: 采样点数
            
        Returns:
            [num_samples, 2] 曲线上的采样点
        """
        n = control_points.shape[0]
        if n < 2:
            return control_points
        
        t = torch.linspace(0, 1, num_samples, device=control_points.device)
        
        # De Casteljau 算法
        points = control_points.unsqueeze(0).expand(num_samples, -1, -1).clone()
        
        for level in range(n - 1, 0, -1):
            for i in range(level):
                points[:, i, :] = (1 - t.unsqueeze(1)) * points[:, i, :] + t.unsqueeze(1) * points[:, i + 1, :]
        
        return points[:, 0, :]
    
    def _compute_min_distance(
        self,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        curve_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算像素网格到曲线的最短距离。
        
        使用分块计算避免内存溢出。
        
        Args:
            grid_x: [H, W] x 坐标网格
            grid_y: [H, W] y 坐标网格
            curve_points: [M, 2] 曲线采样点
            
        Returns:
            [H, W] 最短距离
        """
        H, W = grid_x.shape
        pixels = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [H*W, 2]
        
        chunk_size = 2048
        min_dists = []
        
        for start in range(0, pixels.shape[0], chunk_size):
            end = min(start + chunk_size, pixels.shape[0])
            chunk = pixels[start:end]
            dists = torch.cdist(chunk.unsqueeze(0), curve_points.unsqueeze(0)).squeeze(0)
            min_dist = dists.min(dim=1)[0]
            min_dists.append(min_dist)
        
        return torch.cat(min_dists).reshape(H, W)
    
    def _compute_min_distance_with_index(
        self,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        curve_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算像素网格到曲线的最短距离及最近点索引。
        
        索引用于变宽笔画的局部宽度查询。
        """
        H, W = grid_x.shape
        pixels = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        
        chunk_size = 2048
        min_dists = []
        min_indices = []
        
        for start in range(0, pixels.shape[0], chunk_size):
            end = min(start + chunk_size, pixels.shape[0])
            chunk = pixels[start:end]
            dists = torch.cdist(chunk.unsqueeze(0), curve_points.unsqueeze(0)).squeeze(0)
            min_dist, min_idx = dists.min(dim=1)
            min_dists.append(min_dist)
            min_indices.append(min_idx)
        
        return torch.cat(min_dists).reshape(H, W), torch.cat(min_indices).reshape(H, W)
    
    @staticmethod
    def alpha_blend(foreground: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        """
        可微 alpha blending（Porter-Duff over 操作）。
        
        模拟真实绘画中颜料叠加的效果。
        """
        alpha_fg = foreground[3:4]
        alpha_bg = background[3:4]
        
        out_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
        out_alpha = torch.clamp(out_alpha, 0, 1)
        
        safe_alpha = torch.where(out_alpha > 1e-6, out_alpha, torch.ones_like(out_alpha))
        out_rgb = (foreground[:3] * alpha_fg + background[:3] * alpha_bg * (1 - alpha_fg)) / safe_alpha
        
        result = torch.cat([out_rgb, out_alpha], dim=0)
        return torch.clamp(result, 0, 1)
