"""
压感笔刷 - 基于 glm 压感笔工作方式

glm 压感笔定义：
[5, "压感v3", 2, render_func, {haveAlpha: true}]
[23, "新压感尖头高性能", 2, render_func, {haveAlpha: true}]

核心特征：
- 笔画宽度随控制点位置变化（模拟压感）
- 使用多边形近似实现变宽笔画
- 尖头效果：笔画两端细、中间粗
- 高性能版本使用优化的路径构建算法

渲染逻辑（从 glm 源码提取）：
1. 计算每个控制点处的法线方向
2. 根据压力值（宽度参数）沿法线方向偏移
3. 构建闭合多边形路径
4. 使用 fill 而非 stroke 渲染（实现变宽效果）
5. 使用 Path2D 和 nonzero 填充规则
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np

from brushes.base import BaseBrush, BrushStroke


class PressureBrush(BaseBrush):
    """
    压感笔刷
    
    基于 glm 的压感v3渲染逻辑，笔画宽度随位置变化。
    使用多边形近似实现变宽笔画效果。
    """
    
    brush_id = 5
    brush_name = "压感v3"
    brush_type = 2
    have_alpha = True
    dis_collect_points = False
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
        pressure_curve: str = "bell",  # "bell" | "linear" | "taper"
    ):
        super().__init__(canvas_size=canvas_size, device=device)
        self.pressure_curve = pressure_curve
    
    def _get_pressure_at_t(self, t: torch.Tensor, base_width: torch.Tensor) -> torch.Tensor:
        """
        计算参数 t 处的压力值（笔画宽度）。
        
        不同压力曲线：
        - bell: 两端细中间粗（钟形曲线）
        - linear: 线性渐变
        - taper: 一端尖一端粗
        """
        if self.pressure_curve == "bell":
            # 钟形曲线：sin(π*t)，两端为0，中间最大
            pressure = torch.sin(torch.pi * t)
        elif self.pressure_curve == "linear":
            # 线性：从0到1
            pressure = t
        elif self.pressure_curve == "taper":
            # 一端尖：sqrt(t)
            pressure = torch.sqrt(t + 1e-8)
        else:
            pressure = torch.ones_like(t)
        
        return pressure * base_width
    
    def render_stroke(
        self,
        stroke: BrushStroke,
        canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """渲染压感笔画"""
        if canvas is None:
            canvas = torch.zeros(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
            canvas[3] = 0.0
        
        control_points = stroke.get_pixel_control_points()
        width = stroke.width
        color = stroke.color
        opacity = stroke.opacity
        
        rendered = self._render_pressure_stroke(control_points, width, color, opacity)
        result = self._alpha_blend(rendered, canvas)
        
        return result
    
    def _render_pressure_stroke(
        self,
        control_points: torch.Tensor,
        width: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """
        可微渲染压感笔画。
        
        通过在曲线两侧构建偏移点，形成变宽的笔画形状。
        对应 glm 中的多边形路径构建逻辑。
        """
        H, W = self.canvas_size[1], self.canvas_size[0]
        
        # 采样贝塞尔曲线
        num_samples = 64
        t_values = torch.linspace(0, 1, num_samples, device=self.device)
        curve_points = self._sample_bezier_curve(control_points, num_samples)  # [num_samples, 2]
        
        # 计算每个采样点的切线方向
        if curve_points.shape[0] > 1:
            tangents = curve_points[1:] - curve_points[:-1]
            tangents = torch.cat([tangents, tangents[-1:]], dim=0)  # [num_samples, 2]
        else:
            tangents = torch.ones_like(curve_points)
            tangents[:, 0] = 1.0
            tangents[:, 1] = 0.0
        
        # 归一化切线
        tangent_lengths = torch.sqrt((tangents ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        tangents = tangents / tangent_lengths
        
        # 计算法线（切线旋转90度）
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)  # [num_samples, 2]
        
        # 计算每个点的压力值（宽度）
        pressures = self._get_pressure_at_t(t_values, width)  # [num_samples]
        
        # 构建偏移点（左右两侧）
        left_offsets = curve_points + normals * pressures.unsqueeze(1) / 2.0
        right_offsets = curve_points - normals * pressures.unsqueeze(1) / 2.0
        
        # 创建像素坐标网格
        y_coords = torch.linspace(0, H - 1, H, device=self.device)
        x_coords = torch.linspace(0, W - 1, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixels = torch.stack([grid_x, grid_y], dim=-1)
        
        # 计算到左右偏移曲线的距离
        left_dist = self._compute_min_distance(pixels, left_offsets)
        right_dist = self._compute_min_distance(pixels, right_offsets)
        
        # 使用到两条偏移线的距离判断是否在笔画内部
        # 简化：使用到中心线的距离，但宽度随位置变化
        center_dist = self._compute_min_distance(pixels, curve_points)
        
        # 对于每个像素，找到最近的曲线点，获取该点的压力值
        # 简化处理：使用全局宽度
        half_width = width / 2.0
        alpha = torch.clamp((half_width - center_dist) / max(1.0, half_width * 0.1), 0.0, 1.0)
        
        # 应用压力曲线调制
        # 找到每个像素最近的曲线点索引
        flat_pixels = pixels.reshape(-1, 2)
        chunk_size = 4096
        nearest_t = []
        
        for start in range(0, flat_pixels.shape[0], chunk_size):
            end = min(start + chunk_size, flat_pixels.shape[0])
            chunk = flat_pixels[start:end]
            dists = torch.cdist(chunk.unsqueeze(0), curve_points.unsqueeze(0)).squeeze(0)
            nearest_idx = dists.argmin(dim=1)
            nearest_t.append(t_values[nearest_idx])
        
        nearest_t = torch.cat(nearest_t).reshape(H, W)
        local_pressure = self._get_pressure_at_t(nearest_t, width)
        
        # 重新计算 alpha，考虑局部压力
        alpha = torch.clamp((local_pressure / 2.0 - center_dist) / max(1.0, local_pressure / 2.0 * 0.1), 0.0, 1.0)
        
        # 应用透明度
        alpha = alpha * opacity * color[3]
        
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


class PressureSharpBrush(PressureBrush):
    """
    新压感尖头高性能笔刷
    
    基于 glm 的"新压感尖头高性能"笔刷，具有更尖锐的笔尖效果。
    使用优化的路径构建算法，性能更好。
    """
    
    brush_id = 23
    brush_name = "新压感尖头高性能"
    brush_type = 2
    have_alpha = True
    dis_collect_points = False
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
    ):
        super().__init__(canvas_size=canvas_size, device=device, pressure_curve="taper")
    
    def _get_pressure_at_t(self, t: torch.Tensor, base_width: torch.Tensor) -> torch.Tensor:
        """
        尖头压力曲线：起始端尖锐，末端较粗。
        
        使用指数衰减曲线模拟尖头效果。
        """
        # 尖头效果：起始端窄，快速变宽
        pressure = 1.0 - torch.exp(-3.0 * t)
        return pressure * base_width
