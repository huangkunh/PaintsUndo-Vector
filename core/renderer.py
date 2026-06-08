"""
可微渲染器 - 核心渲染管线

将笔画参数通过可微渲染管线转换为像素图像，
支持梯度回传以优化笔画参数。

渲染管线：
笔画参数 → 贝塞尔曲线构建 → 笔刷特效处理 → 光栅化渲染 → 画布合成 → 输出图像
"""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
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
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.device = device
        self.background_color = background_color
        
        # 笔刷缓存
        self._brush_cache: Dict[str, BaseBrush] = {}
    
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
        渲染一组笔画为最终图像。
        
        Args:
            strokes: 笔画列表
            brush_name: 使用的笔刷名称
            
        Returns:
            渲染后的图像 [C, H, W]，RGB 格式
        """
        brush = self.get_brush(brush_name)
        
        # 创建白色背景画布
        canvas = torch.ones(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
        canvas[3] = 1.0  # 完全不透明
        canvas[0] = self.background_color[0]
        canvas[1] = self.background_color[1]
        canvas[2] = self.background_color[2]
        
        # 逐条渲染笔画
        for stroke in strokes:
            canvas = brush.render_stroke(stroke, canvas)
        
        # 返回 RGB 图像
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
        
        Args:
            stroke_groups: [(brush_name, strokes), ...] 列表
            
        Returns:
            渲染后的 RGB 图像
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
