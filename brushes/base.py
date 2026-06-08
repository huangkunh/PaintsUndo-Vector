"""
笔刷基类 - 定义可微笔刷的统一接口

基于 glm 笔刷工作方式，每种笔刷需要实现：
1. 参数化定义（控制点、宽度、颜色等）
2. 可微渲染方法
3. SVG 导出方法

glm 笔刷工作方式的核心概念：
- 笔刷接收 (canvas_context, points, color, size, opacity) 参数
- points 是 [x0, y0, x1, y1, ...] 的扁平坐标数组
- 不同笔刷使用不同的曲线插值策略（二次贝塞尔、直线等）
- 笔刷可以有特殊属性：haveAlpha（支持透明度）、disCollectPoints（禁用点收集）等
"""

import math
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BrushStroke(nn.Module):
    """
    单条笔画的参数化表示。
    
    一条笔画由以下可微参数组成：
    - control_points: 贝塞尔曲线控制点 [N, 2]
    - width: 笔画基础宽度（标量）
    - color: RGBA 颜色 [4]
    - opacity: 透明度（标量）
    - pressure: 压感曲线参数（用于变宽笔画）
    
    对应 glm 中的笔画数据结构：
    points 数组 + color + size + opacity
    """
    
    def __init__(
        self,
        num_control_points: int = 5,
        canvas_size: Tuple[int, int] = (640, 480),
        init_width: float = 10.0,
        init_color: Optional[torch.Tensor] = None,
        init_opacity: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_control_points = num_control_points
        self.canvas_width, self.canvas_height = canvas_size
        self.device = device
        
        # 初始化控制点 - 随机分布在画布上
        # 使用 sigmoid 约束控制点在画布范围内
        raw_points = torch.randn(num_control_points, 2, device=device) * 0.1
        self.raw_control_points = nn.Parameter(raw_points)
        
        # 笔画宽度 - 使用 softplus 保证正值
        self.raw_width = nn.Parameter(torch.tensor(math.log(math.exp(init_width) - 1), device=device))
        
        # RGBA 颜色 - 使用 sigmoid 约束在 [0, 1]
        if init_color is not None:
            raw_c = torch.logit(torch.clamp(init_color.to(device), 0.01, 0.99))
            self.raw_color = nn.Parameter(raw_c)
        else:
            self.raw_color = nn.Parameter(torch.randn(4, device=device) * 0.5)
        
        # 透明度 - 使用 sigmoid 约束在 [0, 1]
        self.raw_opacity = nn.Parameter(
            torch.tensor(math.log(init_opacity / (1 - init_opacity + 1e-8)), device=device)
        )
    
    @property
    def control_points(self) -> torch.Tensor:
        """获取归一化后的控制点坐标 [0, 1]"""
        return torch.sigmoid(self.raw_control_points)
    
    @property
    def width(self) -> torch.Tensor:
        """获取笔画宽度（正值）"""
        return torch.nn.functional.softplus(self.raw_width)
    
    @property
    def color(self) -> torch.Tensor:
        """获取 RGBA 颜色 [0, 1]"""
        return torch.sigmoid(self.raw_color)
    
    @property
    def opacity(self) -> torch.Tensor:
        """获取透明度 [0, 1]"""
        return torch.sigmoid(self.raw_opacity)
    
    def get_pixel_control_points(self) -> torch.Tensor:
        """获取像素坐标的控制点"""
        cp = self.control_points
        cp_pixel = cp.clone()
        cp_pixel[:, 0] *= self.canvas_width
        cp_pixel[:, 1] *= self.canvas_height
        return cp_pixel
    
    def get_length(self) -> torch.Tensor:
        """计算笔画的近似长度（用于正则化）"""
        cp = self.control_points
        diffs = cp[1:] - cp[:-1]
        distances = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-8)
        return distances.sum()
    
    def to_dict(self) -> Dict[str, Any]:
        """将笔画参数导出为字典"""
        return {
            "control_points": self.control_points.detach().cpu().numpy().tolist(),
            "width": self.width.detach().cpu().item(),
            "color": self.color.detach().cpu().numpy().tolist(),
            "opacity": self.opacity.detach().cpu().item(),
            "num_control_points": self.num_control_points,
        }


class BaseBrush(nn.Module):
    """
    笔刷基类 - 定义可微笔刷的统一接口。
    
    基于 glm 笔刷工作方式，每种笔刷需要实现：
    - render(): 可微渲染方法，将笔画参数渲染为像素图像
    - to_svg_path(): 将笔画转换为 SVG 路径
    - initialize_strokes(): 初始化一组笔画
    
    glm 笔刷定义格式：
    [id, "名称", 类型, 渲染函数, {属性}]
    
    属性包括：
    - haveAlpha: 支持透明度叠加
    - disCollectPoints: 禁用点收集（如渐变、区域填充）
    - hide: 隐藏笔刷
    """
    
    # 笔刷元信息（子类覆盖）
    brush_id: int = -1
    brush_name: str = "base"
    brush_type: int = 2  # 2=线条型, 3=填充型
    have_alpha: bool = True
    dis_collect_points: bool = False
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.device = device
    
    def create_stroke(
        self,
        init_width: float = 10.0,
        init_color: Optional[torch.Tensor] = None,
        init_opacity: float = 1.0,
        num_control_points: int = 5,
    ) -> BrushStroke:
        """创建一条新的笔画"""
        return BrushStroke(
            num_control_points=num_control_points,
            canvas_size=self.canvas_size,
            init_width=init_width,
            init_color=init_color,
            init_opacity=init_opacity,
            device=self.device,
        )
    
    def initialize_strokes(
        self,
        num_strokes: int,
        target_image: Optional[torch.Tensor] = None,
        init_width_range: Tuple[float, float] = (5.0, 20.0),
        num_control_points: int = 5,
    ) -> List[BrushStroke]:
        """
        初始化一组笔画。
        
        如果提供了目标图像，会使用启发式策略（如边缘检测、颜色聚类）
        来初始化笔画的位置和颜色，加速收敛。
        """
        strokes = []
        for i in range(num_strokes):
            width = np.random.uniform(init_width_range[0], init_width_range[1])
            
            if target_image is not None:
                color = self._sample_color_from_image(target_image)
            else:
                color = torch.rand(4)
                color[3] = 1.0
            
            stroke = self.create_stroke(
                init_width=width,
                init_color=color,
                num_control_points=num_control_points,
            )
            strokes.append(stroke)
        
        return strokes
    
    def _sample_color_from_image(self, image: torch.Tensor) -> torch.Tensor:
        """从图像中随机采样一个颜色"""
        C, H, W = image.shape
        y = np.random.randint(0, H)
        x = np.random.randint(0, W)
        color = image[:, y, x].clone()
        color[:3] += torch.randn(3, device=image.device) * 0.05
        color[:3] = color[:3].clamp(0, 1)
        alpha = torch.tensor([1.0], device=image.device)
        color = torch.cat([color[:3], alpha])
        return color
    
    def render_stroke(
        self,
        stroke: BrushStroke,
        canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        渲染单条笔画到画布上。
        
        子类必须实现此方法。
        """
        raise NotImplementedError("Subclasses must implement render_stroke()")
    
    def render_strokes(
        self,
        strokes: List[BrushStroke],
        canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """渲染多条笔画到画布上。"""
        if canvas is None:
            canvas = torch.ones(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
            canvas[3] = 0.0
        
        for stroke in strokes:
            canvas = self.render_stroke(stroke, canvas)
        
        return canvas
    
    def to_svg_path(self, stroke: BrushStroke) -> str:
        """将笔画转换为 SVG 路径字符串。"""
        cp = stroke.control_points.detach().cpu().numpy()
        w = stroke.canvas_width
        h = stroke.canvas_height
        
        points = cp.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        
        if len(points) < 2:
            return ""
        
        path = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
        
        if len(points) == 2:
            path += f" L {points[1][0]:.2f},{points[1][1]:.2f}"
        elif len(points) >= 3:
            for i in range(1, len(points) - 1, 2):
                if i + 1 < len(points):
                    path += f" Q {points[i][0]:.2f},{points[i][1]:.2f} {points[i+1][0]:.2f},{points[i+1][1]:.2f}"
                else:
                    path += f" L {points[i][0]:.2f},{points[i][1]:.2f}"
        
        return path
    
    def get_svg_attributes(self, stroke: BrushStroke) -> Dict[str, str]:
        """获取 SVG 笔画的属性"""
        color = stroke.color.detach().cpu().numpy()
        r, g, b = (color[:3] * 255).astype(int)
        opacity = stroke.opacity.detach().cpu().item()
        width = stroke.width.detach().cpu().item()
        
        return {
            "fill": "none",
            "stroke": f"rgb({r},{g},{b})",
            "stroke-width": f"{width:.2f}",
            "stroke-opacity": f"{opacity:.3f}",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
        }
