"""
可微渲染器 - PyTorch 实现

复刻 JS Canvas 引擎的核心渲染逻辑：
1. Catmull-Rom 样条插值（压感/无压感两种）
2. 压感→线宽映射: lineWidth = width * (pressureMin + pressure * pressurePower)
3. 混合模式: source-over, destination-over, destination-out
4. 笔刷渲染: 马克笔(0), 压感v3(5), 铅笔(6), 水彩(9), 区域(18), 粗头(54)
"""

import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict

from engine_reverser.constants import *


def catmull_rom_no_pressure(points: torch.Tensor, tension: float = 0.5,
                             subdivisions: int = 8, closed: bool = False) -> torch.Tensor:
    """
    无压感 Catmull-Rom 样条插值 (对应 JS 中的 c() 函数)
    
    Args:
        points: [N, 2] 控制点 (x, y)
        tension: 张力参数
        subdivisions: 每段细分数
        closed: 是否闭合
    Returns:
        [M, 2] 插值后的点
    """
    N = points.shape[0]
    if N < 2:
        return points
    
    # 扩展端点
    if closed:
        extended = torch.cat([points[-2:], points, points[:2]], dim=0)
    else:
        extended = torch.cat([points[:1], points, points[-1:]], dim=0)
    
    result = []
    for i in range(1, extended.shape[0] - 2):
        p0 = extended[i - 1]
        p1 = extended[i]
        p2 = extended[i + 1]
        p3 = extended[i + 2]
        
        for j in range(subdivisions):
            t = j / subdivisions
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom 基函数
            c0 = 2 * t3 - 3 * t2 + 1
            c1 = -2 * t3 + 3 * t2
            c2 = t3 - 2 * t2 + t
            c3 = t3 - t2
            
            # 切线
            tan1 = (p2 - p0) * tension
            tan2 = (p3 - p1) * tension
            
            pt = c0 * p1 + c1 * p2 + c2 * tan1 + c3 * tan2
            result.append(pt)
    
    result.append(extended[-2])
    return torch.stack(result)


def catmull_rom_pressure(points: torch.Tensor, tension: float = 0.5,
                          subdivisions: int = 6, min_width: float = 2.0) -> torch.Tensor:
    """
    压感 Catmull-Rom 样条插值 (对应 JS 中的 f() 函数)
    
    Args:
        points: [N, 3] 控制点 (x, y, pressure)
        tension: 张力参数
        subdivisions: 每段细分数
        min_width: 最小宽度参数
    Returns:
        [M, 3] 插值后的点 (x, y, pressure)
    """
    N = points.shape[0]
    if N < 2:
        return points
    
    # 扩展端点 (步长为3: x, y, p)
    extended = torch.cat([points[:1], points, points[-1:]], dim=0)
    
    result = []
    for i in range(1, extended.shape[0] - 2):
        p0 = extended[i - 1]
        p1 = extended[i]
        p2 = extended[i + 1]
        p3 = extended[i + 2]
        
        for j in range(subdivisions):
            t = j / subdivisions
            t2 = t * t
            t3 = t2 * t
            
            c0 = 2 * t3 - 3 * t2 + 1
            c1 = -2 * t3 + 3 * t2
            c2 = t3 - 2 * t2 + t
            c3 = t3 - t2
            
            tan1 = (p2 - p0) * tension
            tan2 = (p3 - p1) * tension
            
            pt = c0 * p1 + c1 * p2 + c2 * tan1 + c3 * tan2
            result.append(pt)
    
    result.append(extended[-2])
    return torch.stack(result)


def smooth_pressure(pressures: torch.Tensor, window: int = 5) -> torch.Tensor:
    """
    平滑压感值 (对应 JS 中的 5 点平均平滑)
    公式: smoothed[i] = (p[i-2] + p[i-1] + p[i] + p[i+1] + p[i+2]) / 5
    """
    N = pressures.shape[0]
    if N < 5:
        return pressures
    
    smoothed = pressures.clone()
    for i in range(2, N - 2):
        smoothed[i] = (pressures[i-2] + pressures[i-1] + pressures[i] + 
                       pressures[i+1] + pressures[i+2]) / 5.0
    
    # 边界处理
    if N >= 3:
        smoothed[1] = (pressures[0] + pressures[1] + pressures[2]) / 3.0
        smoothed[N-2] = (pressures[N-3] + pressures[N-2] + pressures[N-1]) / 3.0
    
    return smoothed


class DifferentiableCanvasRenderer:
    """
    可微画布渲染器
    
    在 PyTorch 中复刻 JS Canvas 引擎的渲染逻辑。
    支持:
    - 多种笔刷 (马克笔、压感、铅笔、水彩、区域、粗头)
    - 混合模式 (source-over, destination-over, destination-out)
    - 压感→线宽映射
    - Catmull-Rom 样条插值
    """
    
    def __init__(self, width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT,
                 device: str = "cpu"):
        self.width = width
        self.height = height
        self.device = device
        
        # 预计算像素坐标网格
        y_coords = torch.arange(height, device=device, dtype=torch.float32)
        x_coords = torch.arange(width, device=device, dtype=torch.float32)
        self.pixel_y, self.pixel_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # [H, W]
    
    def render_action_sequence(self, actions: List, background: str = "#f8ecdb") -> torch.Tensor:
        """
        渲染完整的 Action 序列
        
        Args:
            actions: Action 列表，格式如 [["background", 0], ["colour", "#222"], ...]
            background: 默认背景色
        Returns:
            [3, H, W] 渲染结果
        """
        # 状态机
        state = {
            "color": "#222222",
            "width": 4.0,
            "alpha": 1.0,
            "blend": BLEND_SOURCE_OVER,
            "background": background,
        }
        
        # 初始化画布 (RGBA)
        canvas = self._color_to_tensor(state["background"]).unsqueeze(1).unsqueeze(2).expand(3, self.height, self.width).clone()
        
        for action in actions:
            action_type = action[0]
            payload = action[1] if len(action) > 1 else None
            
            if action_type == "background":
                bg_color = self._get_background_color(payload)
                state["background"] = bg_color
                canvas = self._color_to_tensor(bg_color).unsqueeze(1).unsqueeze(2).expand(3, self.height, self.width).clone()
            
            elif action_type in ("colour", "color"):
                if isinstance(payload, str) and payload.startswith("#"):
                    state["color"] = payload
                elif isinstance(payload, int):
                    state["color"] = DEFAULT_COLORS[payload % len(DEFAULT_COLORS)]
            
            elif action_type == "width":
                state["width"] = float(payload)
            
            elif action_type == "radius":
                radius_list = [2, 4, 10, 20]
                idx = min(int(payload), len(radius_list) - 1)
                state["width"] = float(radius_list[idx])
            
            elif action_type == "alpha":
                state["alpha"] = float(payload)
            
            elif action_type == "blend":
                state["blend"] = payload
            
            elif action_type == "line":
                line_data = payload
                brush_id = int(line_data[0])
                canvas = self._render_line(canvas, line_data, brush_id, state)
        
        return canvas.clamp(0, 1)
    
    def render_with_keyframes(self, actions: List, background: str = "#f8ecdb",
                               keyframe_callback=None) -> torch.Tensor:
        """
        渲染 Action 序列，支持关键帧回调
        
        Args:
            actions: Action 列表
            background: 默认背景色
            keyframe_callback: 回调函数 (action_idx, action, canvas_tensor)
        Returns:
            [3, H, W] 渲染结果
        """
        state = {
            "color": "#222222",
            "width": 4.0,
            "alpha": 1.0,
            "blend": BLEND_SOURCE_OVER,
            "background": background,
        }
        
        canvas = self._color_to_tensor(state["background"]).unsqueeze(1).unsqueeze(2).expand(3, self.height, self.width).clone()
        
        for i, action in enumerate(actions):
            action_type = action[0]
            payload = action[1] if len(action) > 1 else None
            
            if action_type == "background":
                bg_color = self._get_background_color(payload)
                state["background"] = bg_color
                canvas = self._color_to_tensor(bg_color).unsqueeze(1).unsqueeze(2).expand(3, self.height, self.width).clone()
            elif action_type in ("colour", "color"):
                if isinstance(payload, str) and payload.startswith("#"):
                    state["color"] = payload
                elif isinstance(payload, int):
                    state["color"] = DEFAULT_COLORS[payload % len(DEFAULT_COLORS)]
            elif action_type == "width":
                state["width"] = float(payload)
            elif action_type == "radius":
                radius_list = [2, 4, 10, 20]
                idx = min(int(payload), len(radius_list) - 1)
                state["width"] = float(radius_list[idx])
            elif action_type == "alpha":
                state["alpha"] = float(payload)
            elif action_type == "blend":
                state["blend"] = payload
            elif action_type == "line":
                line_data = payload
                brush_id = int(line_data[0])
                canvas = self._render_line(canvas, line_data, brush_id, state)
            
            # 关键帧回调
            if keyframe_callback is not None:
                keyframe_callback(i, action, canvas.clone())
        
        return canvas.clamp(0, 1)
    
    def _render_line(self, canvas: torch.Tensor, line_data: List, brush_id: int,
                     state: Dict) -> torch.Tensor:
        """渲染一条笔画"""
        brush_info = BRUSHES.get(brush_id, ("未知", 2, False))
        brush_name, step, has_pressure = brush_info
        width = state["width"]
        alpha = state["alpha"]
        blend = state["blend"]
        color = state["color"]
        
        # 解析坐标点
        coords = line_data[1:]
        if has_pressure:
            # 步长3: x, y, p
            points = []
            pressures = []
            for i in range(0, len(coords) - 2, 3):
                points.append([coords[i], coords[i+1]])
                pressures.append(coords[i+2])
            if len(points) < 1:
                return canvas
            points_t = torch.tensor(points, device=self.device, dtype=torch.float32)
            pressures_t = torch.tensor(pressures, device=self.device, dtype=torch.float32)
        else:
            # 步长2: x, y
            points = []
            for i in range(0, len(coords) - 1, 2):
                points.append([coords[i], coords[i+1]])
            if len(points) < 1:
                return canvas
            points_t = torch.tensor(points, device=self.device, dtype=torch.float32)
            pressures_t = None
        
        # 根据笔刷类型渲染
        if brush_id == 0:  # 马克笔
            return self._render_marker(canvas, points_t, width, alpha, blend, color)
        elif brush_id == 5:  # 压感v3
            return self._render_pressure_v3(canvas, points_t, pressures_t, width, alpha, blend, color)
        elif brush_id == 6:  # 铅笔
            return self._render_pencil(canvas, points_t, pressures_t, width, alpha, blend, color)
        elif brush_id == 9:  # 水彩
            return self._render_watercolor(canvas, points_t, pressures_t, width, alpha, blend, color)
        elif brush_id == 18:  # 区域
            return self._render_area(canvas, points_t, width, alpha, blend, color)
        elif brush_id == 54:  # 粗头
            return self._render_thick(canvas, points_t, pressures_t, width, alpha, blend, color)
        else:
            # 默认: 用压感v3或马克笔渲染
            if has_pressure and pressures_t is not None:
                return self._render_pressure_v3(canvas, points_t, pressures_t, width, alpha, blend, color)
            else:
                return self._render_marker(canvas, points_t, width, alpha, blend, color)
    
    def _render_marker(self, canvas: torch.Tensor, points: torch.Tensor,
                       width: float, alpha: float, blend: str, color: str) -> torch.Tensor:
        """
        马克笔渲染 (brush 0)
        
        JS 逻辑: quadraticCurveTo 平滑曲线, stroke
        """
        if points.shape[0] < 2:
            return canvas
        
        # Catmull-Rom 插值
        interpolated = catmull_rom_no_pressure(points, CATMULL_ROM_TENSION, 
                                                CATMULL_ROM_SUBDIVISIONS_NOPRESS)
        
        # 渲染线段
        color_t = self._color_to_tensor(color)  # [3]
        line_width = width * BRUSH_MIN_WIDTH  # 简化
        
        return self._draw_smooth_line(canvas, interpolated, line_width, alpha, blend, color_t)
    
    def _render_pressure_v3(self, canvas: torch.Tensor, points: torch.Tensor,
                            pressures: torch.Tensor, width: float, alpha: float,
                            blend: str, color: str) -> torch.Tensor:
        """
        压感v3渲染 (brush 5)
        
        JS 逻辑:
        1. Catmull-Rom 插值 (f函数, 步长3)
        2. 5点平滑压感
        3. lineWidth = width * (Z + pressure * H) = width * (0.2 + pressure * 0.2)
        4. globalAlpha = (pressure / 8) * alpha
        5. 逐段 lineTo + stroke
        """
        if points.shape[0] < 1:
            return canvas
        
        # 单点: 画圆
        if points.shape[0] == 1 and pressures is not None and pressures.shape[0] == 1:
            p = pressures[0].item()
            radius = width * p * 0.1
            color_t = self._color_to_tensor(color)
            return self._draw_circle(canvas, points[0, 0], points[0, 1], radius, alpha, blend, color_t)
        
        # Catmull-Rom 插值 (带压感)
        points_with_p = torch.cat([points, pressures.unsqueeze(1)], dim=1)  # [N, 3]
        min_w = max(0.6 * width, 2.0)
        interpolated = catmull_rom_pressure(points_with_p, CATMULL_ROM_TENSION,
                                             CATMULL_ROM_SUBDIVISIONS, min_w)
        
        interp_points = interpolated[:, :2]   # [M, 2]
        interp_press = interpolated[:, 2]      # [M]
        
        # 5点平滑压感
        interp_press = smooth_pressure(interp_press)
        
        # 逐段渲染
        color_t = self._color_to_tensor(color)
        result = canvas
        for i in range(1, interp_points.shape[0]):
            p = interp_press[i].item()
            line_w = width * (PRESSURE_BRUSH_MIN_WIDTH + p * PRESSURE_BRUSH_POWER)
            seg_alpha = (p / 8.0) * alpha
            
            x1, y1 = interp_points[i-1, 0].item(), interp_points[i-1, 1].item()
            x2, y2 = interp_points[i, 0].item(), interp_points[i, 1].item()
            
            result = self._draw_line_segment(result, x1, y1, x2, y2, line_w, seg_alpha, blend, color_t)
        
        return result
    
    def _render_pencil(self, canvas: torch.Tensor, points: torch.Tensor,
                       pressures: torch.Tensor, width: float, alpha: float,
                       blend: str, color: str) -> torch.Tensor:
        """
        铅笔渲染 (brush 6)
        
        JS 逻辑:
        1. g() 函数处理点 (随机偏移)
        2. Catmull-Rom 插值
        3. lineWidth = width * (0.8 + pressure * 0.07)
        4. globalAlpha = (pressure/8)^3 * alpha
        5. 逐点画短线段
        """
        if points.shape[0] < 1:
            return canvas
        
        # Catmull-Rom 插值
        points_with_p = torch.cat([points, pressures.unsqueeze(1)], dim=1)
        interpolated = catmull_rom_pressure(points_with_p, 1.0, 10, 0.2)
        
        interp_points = interpolated[:, :2]
        interp_press = interpolated[:, 2]
        interp_press = smooth_pressure(interp_press)
        
        color_t = self._color_to_tensor(color)
        result = canvas
        for i in range(1, interp_points.shape[0]):
            p = interp_press[i].item()
            line_w = width * (PENCIL_WIDTH_FACTOR + p * PENCIL_PRESSURE_FACTOR)
            seg_alpha = (p / 8.0) ** PENCIL_ALPHA_POWER * alpha
            
            x1, y1 = interp_points[i-1, 0].item(), interp_points[i-1, 1].item()
            x2, y2 = interp_points[i, 0].item(), interp_points[i, 1].item()
            
            result = self._draw_line_segment(result, x1, y1, x2, y2, line_w, seg_alpha, blend, color_t)
        
        return result
    
    def _render_watercolor(self, canvas: torch.Tensor, points: torch.Tensor,
                           pressures: torch.Tensor, width: float, alpha: float,
                           blend: str, color: str) -> torch.Tensor:
        """
        水彩渲染 (brush 9) - 简化版
        
        JS 逻辑: 使用模糊纹理印章，逐点绘制
        简化: 用半透明圆形模拟水彩效果
        """
        if points.shape[0] < 1:
            return canvas
        
        points_with_p = torch.cat([points, pressures.unsqueeze(1)], dim=1)
        interpolated = catmull_rom_pressure(points_with_p, 1.0, 8, 0.2)
        
        interp_points = interpolated[:, :2]
        interp_press = interpolated[:, 2]
        
        color_t = self._color_to_tensor(color)
        result = canvas
        for i in range(interp_points.shape[0]):
            p = interp_press[i].item()
            # 水彩: 较大的半透明圆
            radius = width * (WATERCOLOR_WIDTH_FACTOR + p * WATERCOLOR_PRESSURE_FACTOR)
            seg_alpha = 0.03 * alpha  # 水彩透明度很低
            
            x, y = interp_points[i, 0].item(), interp_points[i, 1].item()
            result = self._draw_circle(result, x, y, radius, seg_alpha, blend, color_t)
        
        return result
    
    def _render_area(self, canvas: torch.Tensor, points: torch.Tensor,
                     width: float, alpha: float, blend: str, color: str) -> torch.Tensor:
        """
        区域渲染 (brush 18) - 填充多边形
        
        JS 逻辑: closePath + fill
        """
        if points.shape[0] < 3:
            return canvas
        
        color_t = self._color_to_tensor(color)
        return self._fill_polygon(canvas, points, alpha, blend, color_t)
    
    def _render_thick(self, canvas: torch.Tensor, points: torch.Tensor,
                      pressures: torch.Tensor, width: float, alpha: float,
                      blend: str, color: str) -> torch.Tensor:
        """
        粗头渲染 (brush 54)
        
        JS 逻辑:
        1. 计算累积路径长度
        2. lineWidth = width * pressure * 0.1
        3. 逐段渲染
        """
        if points.shape[0] < 1:
            return canvas
        
        # 单点: 画圆
        if points.shape[0] == 1 and pressures is not None:
            p = pressures[0].item()
            radius = min(3.0, width, p * width * 0.1)
            color_t = self._color_to_tensor(color)
            return self._draw_circle(canvas, points[0, 0], points[0, 1], radius, alpha, blend, color_t)
        
        # 两点: 简单线段
        if points.shape[0] == 2 and pressures is not None and pressures.shape[0] == 2:
            p = pressures[0].item()
            line_w = p * width * 0.1
            color_t = self._color_to_tensor(color)
            return self._draw_line_segment(canvas, points[0,0].item(), points[0,1].item(),
                                           points[1,0].item(), points[1,1].item(),
                                           line_w, alpha, blend, color_t)
        
        # 多点: Catmull-Rom + 逐段渲染
        points_with_p = torch.cat([points, pressures.unsqueeze(1)], dim=1)
        interpolated = catmull_rom_pressure(points_with_p, CATMULL_ROM_TENSION,
                                             CATMULL_ROM_SUBDIVISIONS, 2.0)
        
        interp_points = interpolated[:, :2]
        interp_press = interpolated[:, 2]
        interp_press = smooth_pressure(interp_press)
        
        color_t = self._color_to_tensor(color)
        result = canvas
        for i in range(1, interp_points.shape[0]):
            p = interp_press[i].item()
            line_w = width * p * THICK_HEAD_WIDTH_FACTOR
            
            x1, y1 = interp_points[i-1, 0].item(), interp_points[i-1, 1].item()
            x2, y2 = interp_points[i, 0].item(), interp_points[i, 1].item()
            
            result = self._draw_line_segment(result, x1, y1, x2, y2, line_w, alpha, blend, color_t)
        
        return result
    
    # ========== 底层绘制原语 ==========
    
    def _draw_line_segment(self, canvas: torch.Tensor, x1: float, y1: float,
                           x2: float, y2: float, line_width: float, alpha: float,
                           blend: str, color: torch.Tensor) -> torch.Tensor:
        """绘制一条线段（可微近似）"""
        # 计算每个像素到线段的最短距离
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        
        if seg_len_sq < 1e-6:
            # 退化为点
            return self._draw_circle(canvas, x1, y1, line_width / 2, alpha, blend, color)
        
        # 参数化投影
        t = ((self.pixel_x - x1) * dx + (self.pixel_y - y1) * dy) / seg_len_sq
        t = t.clamp(0, 1)
        
        # 最近点
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # 距离
        dist = ((self.pixel_x - closest_x) ** 2 + (self.pixel_y - closest_y) ** 2).sqrt()
        
        # 抗锯齿
        half_w = line_width / 2.0
        aa = max(half_w * 0.15, 0.5)
        
        # Smoothstep
        t_aa = ((half_w + aa - dist) / (2 * aa + 1e-8)).clamp(0, 1)
        mask = t_aa * t_aa * (3 - 2 * t_aa) * alpha
        
        return self._apply_blend(canvas, mask, color, blend)
    
    def _draw_smooth_line(self, canvas: torch.Tensor, points: torch.Tensor,
                          line_width: float, alpha: float, blend: str,
                          color: torch.Tensor) -> torch.Tensor:
        """绘制平滑曲线（逐段渲染）"""
        result = canvas
        for i in range(1, points.shape[0]):
            x1, y1 = points[i-1, 0].item(), points[i-1, 1].item()
            x2, y2 = points[i, 0].item(), points[i, 1].item()
            result = self._draw_line_segment(result, x1, y1, x2, y2, line_width, alpha, blend, color)
        return result
    
    def _draw_circle(self, canvas: torch.Tensor, cx: float, cy: float,
                     radius: float, alpha: float, blend: str,
                     color: torch.Tensor) -> torch.Tensor:
        """绘制圆形"""
        dist = ((self.pixel_x - cx) ** 2 + (self.pixel_y - cy) ** 2).sqrt()
        
        aa = max(radius * 0.15, 0.5)
        t = ((radius + aa - dist) / (2 * aa + 1e-8)).clamp(0, 1)
        mask = t * t * (3 - 2 * t) * alpha
        
        return self._apply_blend(canvas, mask, color, blend)
    
    def _fill_polygon(self, canvas: torch.Tensor, points: torch.Tensor,
                      alpha: float, blend: str, color: torch.Tensor) -> torch.Tensor:
        """填充多边形（简化：用包围盒 + 点在多边形内判断）"""
        # 射线法判断点是否在多边形内
        n = points.shape[0]
        mask = torch.zeros(self.height, self.width, device=self.device)
        
        px = self.pixel_x
        py = self.pixel_y
        
        for i in range(n):
            j = (i + 1) % n
            xi, yi = points[i, 0], points[i, 1]
            xj, yj = points[j, 0], points[j, 1]
            
            # 射线交叉判断
            cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-8) + xi)
            mask = torch.logical_xor(mask.bool(), cond.bool()).float()
        
        mask = mask * alpha
        return self._apply_blend(canvas, mask, color, blend)
    
    def _apply_blend(self, canvas: torch.Tensor, mask: torch.Tensor,
                     color: torch.Tensor, blend: str) -> torch.Tensor:
        """
        应用混合模式
        
        source-over: fg * alpha + bg * (1 - alpha)  (默认)
        destination-over: bg * alpha + fg * (1 - alpha)  (底色在下层)
        destination-out: bg * (1 - alpha)  (橡皮擦)
        """
        mask_3d = mask.unsqueeze(0)  # [1, H, W]
        
        if blend == BLEND_SOURCE_OVER:
            # 前景覆盖背景
            result = color.unsqueeze(1).unsqueeze(2) * mask_3d + canvas * (1 - mask_3d)
        elif blend == BLEND_DESTINATION_OVER:
            # 背景覆盖前景（底色在下层）
            result = canvas * mask_3d + color.unsqueeze(1).unsqueeze(2) * (1 - mask_3d)
            # 实际上 destination-over 是: 先画fg，再把bg放在fg下面
            # result = fg * fg_alpha + bg * (1 - fg_alpha)
            # 但在我们的实现中，canvas是当前状态，color是新绘制的
            # destination-over: 新绘制的内容在已有内容之下
            fg_alpha = 1 - mask_3d  # 新内容的不透明度（反向）
            result = canvas * (1 - fg_alpha) + color.unsqueeze(1).unsqueeze(2) * fg_alpha
            # 修正: destination-over 意味着新像素在已有像素之下
            # 已有像素的 alpha 保持，新像素只在已有像素透明的地方显示
            result = canvas * mask_3d + color.unsqueeze(1).unsqueeze(2) * (1 - mask_3d) * (1 - mask_3d)
            # 简化: 直接用 source-over 但反转 alpha
            result = color.unsqueeze(1).unsqueeze(2) * (1 - mask_3d) + canvas * mask_3d
        elif blend == BLEND_DESTINATION_OUT:
            # 橡皮擦: 移除已有内容
            result = canvas * (1 - mask_3d)
        elif blend == BLEND_MULTIPLY:
            result = canvas * color.unsqueeze(1).unsqueeze(2) * mask_3d + canvas * (1 - mask_3d)
        elif blend == BLEND_DARKEN:
            blended = torch.min(canvas, color.unsqueeze(1).unsqueeze(2))
            result = blended * mask_3d + canvas * (1 - mask_3d)
        else:
            # 默认 source-over
            result = color.unsqueeze(1).unsqueeze(2) * mask_3d + canvas * (1 - mask_3d)
        
        return result.clamp(0, 1)
    
    def _color_to_tensor(self, color_str: str) -> torch.Tensor:
        """将颜色字符串转为 [3] 张量"""
        if color_str.startswith("#"):
            hex_color = color_str[1:]
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                return torch.tensor([r, g, b], device=self.device)
        return torch.tensor([0.0, 0.0, 0.0], device=self.device)
    
    def _get_background_color(self, index: int) -> str:
        """获取背景色"""
        if isinstance(index, int) and 0 <= index < len(BACKGROUND_COLORS):
            return BACKGROUND_COLORS[index]
        return BACKGROUND_COLORS[0]
