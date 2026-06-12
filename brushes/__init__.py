"""
笔刷模块初始化

基于 glm 绘画应用的笔刷工作方式，提供多种可微笔刷实现。
每种笔刷定义包含：
- 渲染函数：接收点序列、颜色、大小、透明度参数
- 点收集策略：不同笔刷有不同的采样方式
- 特效参数：如水彩的扩散半径、喷笔的喷雾密度等
"""

from brushes.base import BaseBrush, BrushStroke
from brushes.marker import MarkerBrush
from brushes.pencil import PencilBrush
from brushes.watercolor import WatercolorBrush
from brushes.pressure import PressureBrush, PressureSharpBrush
from brushes.airbrush import AirbrushBrush
from brushes.gradient import GradientBrush
from brushes.hatching import HatchingBrush
from brushes.halftone import HalftoneBrush

# 笔刷注册表
BRUSH_REGISTRY = {
    "marker": MarkerBrush,
    "pencil": PencilBrush,
    "watercolor": WatercolorBrush,
    "pressure": PressureBrush,
    "pressure_sharp": PressureSharpBrush,
    "airbrush": AirbrushBrush,
    "gradient": GradientBrush,
    "hatching": HatchingBrush,
    "halftone": HalftoneBrush,
}


def get_brush(name: str) -> type:
    """根据名称获取笔刷类"""
    if name not in BRUSH_REGISTRY:
        raise ValueError(f"Unknown brush: {name}. Available: {list(BRUSH_REGISTRY.keys())}")
    return BRUSH_REGISTRY[name]


def list_brushes() -> list:
    """列出所有可用笔刷"""
    return list(BRUSH_REGISTRY.keys())
