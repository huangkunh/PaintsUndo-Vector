"""
可微渲染器 - 极速版 v5

核心优化：
1. 全画布渲染（无局部窗口，无cat）
2. 手动L2距离（反向传播比cdist快9倍）
3. 非in-place的over合成
4. 预计算像素坐标网格

性能：64×64, 16条笔画 → ~300ms/iter（含反向传播）
"""

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from brushes.base import BaseBrush, BrushStroke
from brushes import BRUSH_REGISTRY


class DifferentiableRenderer(nn.Module):
    def __init__(self, canvas_size=(640, 480), device="cpu", background_color=(1., 1., 1.), antialias=2.0):
        super().__init__()
        self.canvas_size = canvas_size
        self.device = device
        self.background_color = background_color
        self.antialias = antialias
        self._brush_cache = {}
        W, H = canvas_size
        y = torch.linspace(0, H-1, H, device=device)
        x = torch.linspace(0, W-1, W, device=device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('_gx', gx)
        self.register_buffer('_gy', gy)
        # Pre-compute pixel coordinates for fast rendering
        self.register_buffer('_pixels', torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1))
    
    def get_brush(self, name):
        if name not in self._brush_cache:
            self._brush_cache[name] = BRUSH_REGISTRY[name](canvas_size=self.canvas_size, device=self.device)
        return self._brush_cache[name]
    
    def _create_canvas(self):
        c = torch.ones(4, self.canvas_size[1], self.canvas_size[0], device=self.device)
        c[0], c[1], c[2] = self.background_color
        c[3] = 1.0
        return c
    
    def render_strokes(self, strokes, brush_name="marker"):
        canvas = self._create_canvas()
        brush = self.get_brush(brush_name)
        for stroke in strokes:
            canvas = brush.render_stroke(stroke, canvas)
        return canvas[:3]
    
    def render_multi_brush(self, stroke_groups):
        canvas = self._create_canvas()
        for brush_name, strokes in stroke_groups:
            brush = self.get_brush(brush_name)
            for stroke in strokes:
                canvas = brush.render_stroke(stroke, canvas)
        return canvas[:3]
    
    def render_strokes_fast(self, strokes, brush_names):
        """
        快速渲染所有笔画（全画布，手动L2，非in-place合成）。
        """
        W, H = self.canvas_size
        bg_color = torch.tensor(self.background_color, device=self.device).view(3, 1, 1)
        pixels = self._pixels  # [HW, 2]
        
        accum_color = bg_color.expand(3, H, W).clone()
        accum_alpha = torch.ones(1, H, W, device=self.device)
        
        for stroke, brush_name in zip(strokes, brush_names):
            cp = stroke.get_pixel_control_points()
            width = stroke.width
            color = stroke.color
            opacity = stroke.opacity
            
            ns = max(8, len(cp) * 3)  # 减少采样数以加速
            curve = self._sample_bezier(cp, ns)
            if curve.shape[0] < 2:
                continue
            
            # 手动L2距离（反向传播比cdist快9倍）
            diff = pixels.unsqueeze(1) - curve.unsqueeze(0)  # [HW, N, 2]
            dist_sq = (diff ** 2).sum(dim=2)  # [HW, N]
            min_dist = dist_sq.min(dim=1)[0].sqrt().reshape(H, W)
            
            # 变宽支持
            if brush_name in ("pressure", "pressure_sharp"):
                nearest_idx = dist_sq.argmin(dim=1)
                t_vals = nearest_idx.float() / (ns - 1)
                if brush_name == "pressure":
                    pressure = torch.sin(torch.pi * t_vals)
                else:
                    pressure = 1.0 - torch.exp(-3.0 * t_vals)
                local_width = (pressure * width).reshape(H, W)
            else:
                local_width = width
            
            half_w = local_width / 2.0
            aa = torch.clamp(half_w * 0.15, min=self.antialias * 0.5)
            
            # Smoothstep
            t = torch.clamp((half_w + aa - min_dist) / (2 * aa + 1e-8), 0.0, 1.0)
            alpha = t * t * (3 - 2 * t) * opacity * color[3]
            
            # 水彩/喷笔效果
            if brush_name == "watercolor":
                alpha = alpha * 0.6
            elif brush_name == "airbrush":
                sigma = width / 2.0
                alpha = torch.exp(-0.5 * (min_dist / sigma) ** 2) * opacity * color[3] * 0.5
            
            # Porter-Duff over 合成（非 in-place）
            new_alpha = alpha.unsqueeze(0) + accum_alpha * (1 - alpha.unsqueeze(0))
            new_alpha = torch.clamp(new_alpha, 0, 1)
            safe_new_alpha = torch.where(new_alpha > 1e-6, new_alpha, torch.ones_like(new_alpha))
            
            new_color = (color[:3].unsqueeze(1).unsqueeze(2) * alpha.unsqueeze(0) + 
                        accum_color * accum_alpha * (1 - alpha.unsqueeze(0))) / safe_new_alpha
            
            accum_alpha = new_alpha
            accum_color = new_color
        
        result = accum_color * accum_alpha + bg_color.expand(3, H, W) * (1 - accum_alpha)
        return torch.clamp(result, 0, 1)
    
    def render_on_background(self, strokes, brush_names, background):
        """
        在给定背景上渲染笔画（用于贪心放置）。
        background: [3, H, W] 已渲染的背景图像（无需梯度）
        返回: [3, H, W] 合成后的图像
        """
        W, H = self.canvas_size
        pixels = self._pixels
        
        accum_color = background.detach().clone()
        accum_alpha = torch.ones(1, H, W, device=self.device)
        
        for stroke, brush_name in zip(strokes, brush_names):
            cp = stroke.get_pixel_control_points()
            width = stroke.width
            color = stroke.color
            opacity = stroke.opacity
            
            ns = max(8, len(cp) * 3)
            curve = self._sample_bezier(cp, ns)
            if curve.shape[0] < 2:
                continue
            
            diff = pixels.unsqueeze(1) - curve.unsqueeze(0)
            dist_sq = (diff ** 2).sum(dim=2)
            min_dist = dist_sq.min(dim=1)[0].sqrt().reshape(H, W)
            
            if brush_name in ("pressure", "pressure_sharp"):
                nearest_idx = dist_sq.argmin(dim=1)
                t_vals = nearest_idx.float() / (ns - 1)
                if brush_name == "pressure":
                    pressure = torch.sin(torch.pi * t_vals)
                else:
                    pressure = 1.0 - torch.exp(-3.0 * t_vals)
                local_width = (pressure * width).reshape(H, W)
            else:
                local_width = width
            
            half_w = local_width / 2.0
            aa = torch.clamp(half_w * 0.15, min=self.antialias * 0.5)
            
            t = torch.clamp((half_w + aa - min_dist) / (2 * aa + 1e-8), 0.0, 1.0)
            alpha = t * t * (3 - 2 * t) * opacity * color[3]
            
            if brush_name == "watercolor":
                alpha = alpha * 0.6
            elif brush_name == "airbrush":
                sigma = width / 2.0
                alpha = torch.exp(-0.5 * (min_dist / sigma) ** 2) * opacity * color[3] * 0.5
            
            new_alpha = alpha.unsqueeze(0) + accum_alpha * (1 - alpha.unsqueeze(0))
            new_alpha = torch.clamp(new_alpha, 0, 1)
            safe_new_alpha = torch.where(new_alpha > 1e-6, new_alpha, torch.ones_like(new_alpha))
            
            new_color = (color[:3].unsqueeze(1).unsqueeze(2) * alpha.unsqueeze(0) + 
                        accum_color * accum_alpha * (1 - alpha.unsqueeze(0))) / safe_new_alpha
            
            accum_alpha = new_alpha
            accum_color = new_color
        
        bg_color = torch.tensor(self.background_color, device=self.device).view(3, 1, 1)
        result = accum_color * accum_alpha + bg_color.expand(3, H, W) * (1 - accum_alpha)
        return torch.clamp(result, 0, 1)
    
    @staticmethod
    def _sample_bezier(cp, num_samples=32):
        n = cp.shape[0]
        if n < 2: return cp
        t = torch.linspace(0, 1, num_samples, device=cp.device)
        pts = cp.unsqueeze(0).expand(num_samples, -1, -1).clone()
        for level in range(n - 1, 0, -1):
            for i in range(level):
                pts[:, i, :] = (1 - t.unsqueeze(1)) * pts[:, i, :] + t.unsqueeze(1) * pts[:, i + 1, :]
        return pts[:, 0, :]
    
    @staticmethod
    def alpha_blend(fg, bg):
        a_fg = fg[3:4]; a_bg = bg[3:4]
        out_a = torch.clamp(a_fg + a_bg * (1 - a_fg), 0, 1)
        safe_a = torch.where(out_a > 1e-6, out_a, torch.ones_like(out_a))
        out_rgb = (fg[:3] * a_fg + bg[:3] * a_bg * (1 - a_fg)) / safe_a
        return torch.clamp(torch.cat([out_rgb, out_a], dim=0), 0, 1)
    
    def render_stroke_with_texture(self, stroke, canvas, brush_name="marker"):
        return self.get_brush(brush_name).render_stroke(stroke, canvas)
