"""
SVG 导出模块

将优化后的笔画参数导出为 SVG 文件，
支持多种笔刷的 SVG 路径生成和滤镜效果。

对应方案B阶段六：导出矢量笔画与回放
"""

import os
from typing import List, Optional, Tuple, Dict

import svgwrite
import numpy as np

from brushes.base import BrushStroke
from brushes import BRUSH_REGISTRY


class SVGExporter:
    """
    SVG 导出器
    
    将笔画列表导出为 SVG 文件，支持：
    - 多种笔刷的 SVG 路径
    - SVG 滤镜（水彩扩散、铅笔纹理等）
    - 分阶段导出
    - 动画回放（SMIL 动画）
    """
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        background_color: str = "#FFFFFF",
    ):
        self.canvas_width, self.canvas_height = canvas_size
        self.background_color = background_color
    
    def export(
        self,
        strokes: List[BrushStroke],
        brush_names: List[str],
        output_path: str,
        include_filters: bool = True,
        include_animation: bool = False,
    ):
        """
        导出笔画为 SVG 文件。
        
        Args:
            strokes: 笔画列表
            brush_names: 对应的笔刷名称列表
            output_path: 输出文件路径
            include_filters: 是否包含 SVG 滤镜
            include_animation: 是否包含 SMIL 动画
        """
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{self.canvas_width}px", f"{self.canvas_height}px"),
            viewBox=f"0 0 {self.canvas_width} {self.canvas_height}",
        )
        
        # 添加背景
        dwg.add(dwg.rect(
            insert=(0, 0),
            size=(self.canvas_width, self.canvas_height),
            fill=self.background_color,
        ))
        
        # 添加滤镜定义
        if include_filters:
            self._add_filters(dwg)
        
        # 添加笔画
        for idx, (stroke, brush_name) in enumerate(zip(strokes, brush_names)):
            self._add_stroke_to_svg(dwg, stroke, brush_name, idx, include_animation)
        
        dwg.save()
        print(f"SVG exported to: {output_path}")
    
    def export_stages(
        self,
        stages_history: List[Dict],
        output_dir: str,
    ):
        """
        分阶段导出 SVG 文件。
        
        每个阶段生成一个独立的 SVG 文件，
        对应方案B中的 History_1 → step_1.svg, History_2 → step_2.svg 等。
        """
        os.makedirs(output_dir, exist_ok=True)
        
        accumulated_strokes = []
        accumulated_brushes = []
        
        for stage_idx, stage_data in enumerate(stages_history):
            accumulated_strokes.extend(stage_data["strokes"])
            accumulated_brushes.extend(stage_data["brush_names"])
            
            output_path = os.path.join(output_dir, f"step_{stage_idx + 1}.svg")
            self.export(
                accumulated_strokes,
                accumulated_brushes,
                output_path,
            )
        
        # 最终完整版
        final_path = os.path.join(output_dir, "step_final.svg")
        self.export(
            accumulated_strokes,
            accumulated_brushes,
            final_path,
        )
    
    def _add_stroke_to_svg(
        self,
        dwg: svgwrite.Drawing,
        stroke: BrushStroke,
        brush_name: str,
        index: int,
        include_animation: bool = False,
    ):
        """添加单条笔画到 SVG"""
        brush_cls = BRUSH_REGISTRY[brush_name]
        brush = brush_cls(canvas_size=(self.canvas_width, self.canvas_height))
        
        path_data = brush.to_svg_path(stroke)
        if not path_data:
            return
        
        attrs = brush.get_svg_attributes(stroke)
        
        path = dwg.path(d=path_data, **attrs)
        
        # 添加动画
        if include_animation:
            # SMIL 动画：笔画逐步绘制
            duration = 0.5  # 每条笔画0.5秒
            delay = index * duration
            path.add(dwg.animate(
                attributeName="stroke-dashoffset",
                from_="1000",
                to="0",
                dur=f"{duration}s",
                begin=f"{delay}s",
                fill="freeze",
            ))
            path["stroke-dasharray"] = "1000"
        
        dwg.add(path)
    
    def _add_filters(self, dwg: svgwrite.Drawing):
        """添加 SVG 滤镜定义"""
        defs = dwg.defs
        
        # 水彩扩散滤镜
        watercolor_filter = defs.filter(
            id="watercolor-filter",
            size=("150%", "150%"),
            x="-25%", y="-25%",
        )
        watercolor_filter.feGaussianBlur(
            in_="SourceGraphic",
            stdDeviation="2",
            result="blur",
        )
        watercolor_filter.feComposite(
            in_="blur",
            in2="SourceGraphic",
            operator="over",
        )
        defs.add(watercolor_filter)
        
        # 铅笔纹理滤镜
        pencil_filter = defs.filter(
            id="pencil-filter",
            size=("120%", "120%"),
            x="-10%", y="-10%",
        )
        pencil_filter.feTurbulence(
            type="fractalNoise",
            baseFrequency="0.5",
            numOctaves="4",
            result="noise",
        )
        pencil_filter.feDisplacementMap(
            in_="SourceGraphic",
            in2="noise",
            scale="1",
        )
        defs.add(pencil_filter)
