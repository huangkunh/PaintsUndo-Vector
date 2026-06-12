"""
SVG 导出模块 - 高质量矢量笔画导出

将优化后的笔画参数导出为 SVG 文件，
支持多种笔刷的 SVG 路径生成和滤镜效果。

改进：
1. 更精确的贝塞尔曲线 SVG 路径
2. 变宽笔画 SVG 实现（使用多边形近似）
3. 水彩/喷笔 SVG 滤镜
4. 绘画过程动画（SMIL + CSS）
5. 分层导出（每个阶段一个 SVG 组）
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
    - 变宽笔画（多边形近似）
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
        stages_history: Optional[List[Dict]] = None,
    ):
        """
        导出笔画为 SVG 文件。
        
        Args:
            strokes: 笔画列表
            brush_names: 对应的笔刷名称列表
            output_path: 输出文件路径
            include_filters: 是否包含 SVG 滤镜
            include_animation: 是否包含 SMIL 动画
            stages_history: 阶段历史（用于分层导出）
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
        
        # 分层导出
        if stages_history:
            for stage_idx, stage_data in enumerate(stages_history):
                stage_group = dwg.g(id=f"stage_{stage_idx + 1}")
                stage_strokes = stage_data["strokes"]
                stage_brushes = stage_data["brush_names"]
                
                for i, (stroke, brush_name) in enumerate(zip(stage_strokes, stage_brushes)):
                    self._add_stroke_to_group(dwg, stage_group, stroke, brush_name, i,
                                             include_animation=include_animation)
                
                dwg.add(stage_group)
        else:
            for i, (stroke, brush_name) in enumerate(zip(strokes, brush_names)):
                self._add_stroke_to_svg(dwg, stroke, brush_name, i,
                                       include_animation=include_animation)
        
        dwg.save()
        print(f"SVG exported to: {output_path}")
    
    def _add_stroke_to_svg(
        self,
        dwg: svgwrite.Drawing,
        stroke: BrushStroke,
        brush_name: str,
        index: int,
        include_animation: bool = False,
    ):
        """添加单条笔画到 SVG"""
        brush = BRUSH_REGISTRY.get(brush_name)
        if brush is None:
            return
        
        # 获取笔刷实例
        brush_inst = brush(canvas_size=(self.canvas_width, self.canvas_height))
        
        # 变宽笔画使用多边形近似
        if brush_name in ("pressure", "pressure_sharp"):
            self._add_variable_width_stroke(dwg, stroke, brush_name, index,
                                           include_animation)
        else:
            path_d = brush_inst.to_svg_path(stroke)
            if not path_d:
                return
            
            attrs = brush_inst.get_svg_attributes(stroke)
            path = dwg.path(d=path_d, **attrs)
            
            if include_animation:
                self._add_path_animation(path, index)
            
            dwg.add(path)
    
    def _add_stroke_to_group(
        self,
        dwg: svgwrite.Drawing,
        group: svgwrite.container.Group,
        stroke: BrushStroke,
        brush_name: str,
        index: int,
        include_animation: bool = False,
    ):
        """添加单条笔画到 SVG 组"""
        brush = BRUSH_REGISTRY.get(brush_name)
        if brush is None:
            return
        
        brush_inst = brush(canvas_size=(self.canvas_width, self.canvas_height))
        
        if brush_name in ("pressure", "pressure_sharp"):
            self._add_variable_width_stroke_to_group(dwg, group, stroke, brush_name,
                                                    index, include_animation)
        else:
            path_d = brush_inst.to_svg_path(stroke)
            if not path_d:
                return
            
            attrs = brush_inst.get_svg_attributes(stroke)
            path = dwg.path(d=path_d, **attrs)
            
            if include_animation:
                self._add_path_animation(path, index)
            
            group.add(path)
    
    def _add_variable_width_stroke(
        self,
        dwg,
        stroke: BrushStroke,
        brush_name: str,
        index: int,
        include_animation: bool = False,
    ):
        """添加变宽笔画（使用多边形近似）"""
        cp = stroke.control_points.detach().cpu().numpy()
        w = self.canvas_width
        h = self.canvas_height
        
        points = cp.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        
        if len(points) < 2:
            return
        
        # 计算变宽多边形
        polygon_points = self._compute_variable_width_polygon(
            points, stroke, brush_name
        )
        
        if len(polygon_points) < 3:
            return
        
        color = stroke.color.detach().cpu().numpy()
        r, g, b = (color[:3] * 255).astype(int)
        opacity = stroke.opacity.detach().cpu().item()
        
        polygon_str = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in polygon_points)
        
        polygon = dwg.polygon(
            points=[(p[0], p[1]) for p in polygon_points],
            fill=f"rgb({r},{g},{b})",
            fill_opacity=f"{opacity:.3f}",
            stroke="none",
        )
        
        if include_animation:
            self._add_polygon_animation(polygon, index)
        
        dwg.add(polygon)
    
    def _add_variable_width_stroke_to_group(
        self,
        dwg,
        group,
        stroke: BrushStroke,
        brush_name: str,
        index: int,
        include_animation: bool = False,
    ):
        """添加变宽笔画到组"""
        cp = stroke.control_points.detach().cpu().numpy()
        w = self.canvas_width
        h = self.canvas_height
        
        points = cp.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        
        if len(points) < 2:
            return
        
        polygon_points = self._compute_variable_width_polygon(
            points, stroke, brush_name
        )
        
        if len(polygon_points) < 3:
            return
        
        color = stroke.color.detach().cpu().numpy()
        r, g, b = (color[:3] * 255).astype(int)
        opacity = stroke.opacity.detach().cpu().item()
        
        polygon = dwg.polygon(
            points=[(p[0], p[1]) for p in polygon_points],
            fill=f"rgb({r},{g},{b})",
            fill_opacity=f"{opacity:.3f}",
            stroke="none",
        )
        
        group.add(polygon)
    
    def _compute_variable_width_polygon(
        self,
        points: np.ndarray,
        stroke: BrushStroke,
        brush_name: str,
    ) -> np.ndarray:
        """
        计算变宽笔画的多边形近似。
        
        沿笔画路径计算法线方向，根据压力曲线偏移，
        生成闭合多边形实现变宽效果。
        """
        n = len(points)
        base_width = stroke.width.detach().cpu().item()
        
        # 计算每个点的切线方向
        tangents = np.zeros_like(points)
        for i in range(n):
            if i == 0:
                tangents[i] = points[1] - points[0]
            elif i == n - 1:
                tangents[i] = points[-1] - points[-2]
            else:
                tangents[i] = points[i + 1] - points[i - 1]
        
        # 归一化
        lengths = np.sqrt((tangents ** 2).sum(axis=1, keepdims=True)) + 1e-8
        tangents = tangents / lengths
        
        # 法线方向（切线旋转90度）
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        
        # 计算每个点的宽度（压力曲线）
        widths = np.zeros(n)
        for i in range(n):
            t = i / (n - 1)
            if brush_name == "pressure_sharp":
                # 尖头：起始端窄
                pressure = 1.0 - np.exp(-3.0 * t)
            else:
                # 钟形：两端细中间粗
                pressure = np.sin(np.pi * t)
            widths[i] = base_width * pressure / 2.0
        
        # 构建多边形
        left_points = points + normals * widths[:, np.newaxis]
        right_points = points - normals * widths[:, np.newaxis]
        
        # 闭合多边形
        polygon = np.concatenate([left_points, right_points[::-1]], axis=0)
        
        return polygon
    
    def _add_path_animation(self, path, index: int):
        """添加路径绘制动画"""
        duration = 0.5
        delay = index * duration * 0.3
        path.add(path.root.animate(
            attributeName="stroke-dashoffset",
            from_="1000",
            to="0",
            dur=f"{duration}s",
            begin=f"{delay}s",
            fill="freeze",
        ))
        path["stroke-dasharray"] = "1000"
    
    def _add_polygon_animation(self, polygon, index: int):
        """添加多边形淡入动画"""
        duration = 0.3
        delay = index * 0.15
        polygon.add(polygon.root.animate(
            attributeName="fill-opacity",
            from_="0",
            to=polygon.get("fill-opacity", "1"),
            dur=f"{duration}s",
            begin=f"{delay}s",
            fill="freeze",
        ))
    
    def _add_filters(self, dwg: svgwrite.Drawing):
        """添加 SVG 滤镜定义（使用 xml.etree 兼容 svgwrite）"""
        from xml.etree.ElementTree import SubElement
        
        # 获取 defs 的底层 XML 元素
        defs = dwg.defs
        defs_xml = defs.get_xml()
        
        # 水彩扩散滤镜
        wc_filter = SubElement(defs_xml, '{http://www.w3.org/2000/svg}filter',
            id='watercolor-filter', x='-25%', y='-25%', width='150%', height='150%')
        SubElement(wc_filter, '{http://www.w3.org/2000/svg}feGaussianBlur',
            attrib={'in': 'SourceGraphic', 'stdDeviation': '2', 'result': 'blur'})
        SubElement(wc_filter, '{http://www.w3.org/2000/svg}feComposite',
            attrib={'in': 'blur', 'in2': 'SourceGraphic', 'operator': 'over'})
        
        # 铅笔纹理滤镜
        pn_filter = SubElement(defs_xml, '{http://www.w3.org/2000/svg}filter',
            id='pencil-filter', x='-10%', y='-10%', width='120%', height='120%')
        SubElement(pn_filter, '{http://www.w3.org/2000/svg}feTurbulence',
            attrib={'type': 'fractalNoise', 'baseFrequency': '0.5', 'numOctaves': '4', 'result': 'noise'})
        SubElement(pn_filter, '{http://www.w3.org/2000/svg}feDisplacementMap',
            attrib={'in': 'SourceGraphic', 'in2': 'noise', 'scale': '1'})
        
        # 喷笔柔和滤镜
        ab_filter = SubElement(defs_xml, '{http://www.w3.org/2000/svg}filter',
            id='airbrush-filter', x='-10%', y='-10%', width='120%', height='120%')
        SubElement(ab_filter, '{http://www.w3.org/2000/svg}feGaussianBlur',
            attrib={'in': 'SourceGraphic', 'stdDeviation': '3'})
