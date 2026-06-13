"""
逆向笔画重建器 - 核心推理模块

不依赖训练好的 Transformer（零样本），通过优化方法生成 Action 序列：
1. 多尺度色块分析目标图像
2. 按人类画师逻辑生成 Action 序列
3. 通过可微渲染器验证并优化

绘画顺序: 底色(destination-over) → 阴影 → 线稿(source-over) → 高光
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import colorsys
from typing import List, Tuple, Dict, Optional

from engine_reverser.constants import *
from engine_reverser.renderer import DifferentiableCanvasRenderer


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """RGB [0,1] → #RRGGBB"""
    ri = max(0, min(255, int(r * 255)))
    gi = max(0, min(255, int(g * 255)))
    bi = max(0, min(255, int(b * 255)))
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def quantize_color(r: float, g: float, b: float, levels: int = 16) -> str:
    """量化颜色到指定级数"""
    ri = max(0, min(255, int(round(r * 255 / (256 // levels)) * (256 // levels))))
    gi = max(0, min(255, int(round(g * 255 / (256 // levels)) * (256 // levels))))
    bi = max(0, min(255, int(round(b * 255 / (256 // levels)) * (256 // levels))))
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def get_luminance(r: float, g: float, b: float) -> float:
    """计算亮度"""
    return 0.299 * r + 0.587 * g + 0.114 * b


class StrokeReverser:
    """
    逆向笔画重建器
    
    输入: 目标图像
    输出: 引擎 Action 序列
    
    策略:
    1. 分析目标图像的多尺度特征
    2. 按人类画师逻辑生成 Action:
       - background: 设置背景色
       - destination-over: 铺底色（大色块）
       - source-over: 刻画形体（中等色块）
       - source-over: 线稿细节（细线条）
       - source-over: 高光（小色块）
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.renderer = DifferentiableCanvasRenderer(CANVAS_WIDTH, CANVAS_HEIGHT, device)
    
    def reconstruct(self, target: torch.Tensor) -> List[List]:
        """
        从目标图像逆向重建 Action 序列
        
        Args:
            target: [3, H, W] 目标图像 (0-1)
        Returns:
            Action 序列
        """
        C, H, W = target.shape
        assert W == CANVAS_WIDTH and H == CANVAS_HEIGHT, f"图像尺寸必须为 {CANVAS_WIDTH}x{CANVAS_HEIGHT}"
        
        actions = []
        
        # ===== 阶段 1: 设置背景色 =====
        bg_color = self._analyze_background(target)
        bg_idx = self._find_closest_bg_index(bg_color)
        actions.append(["background", bg_idx])
        
        # ===== 阶段 2: 铺底色 (destination-over) =====
        base_actions = self._generate_base_color_actions(target, bg_color)
        actions.extend(base_actions)
        
        # ===== 阶段 3: 阴影/中间调 (source-over, 低透明度) =====
        shadow_actions = self._generate_shadow_actions(target)
        actions.extend(shadow_actions)
        
        # ===== 阶段 4: 形体刻画 (source-over, 压感笔刷) =====
        shape_actions = self._generate_shape_actions(target)
        actions.extend(shape_actions)
        
        # ===== 阶段 5: 线稿 (source-over, 铅笔) =====
        line_actions = self._generate_line_art_actions(target)
        actions.extend(line_actions)
        
        # ===== 阶段 6: 高光 (source-over, 小色块) =====
        highlight_actions = self._generate_highlight_actions(target)
        actions.extend(highlight_actions)
        
        # ===== 阶段 7: 精修 (source-over, 细笔刷) =====
        detail_actions = self._generate_detail_actions(target)
        actions.extend(detail_actions)
        
        return actions
    
    def _analyze_background(self, target: torch.Tensor) -> Tuple[float, float, float]:
        """分析背景色（取边缘像素的中位数）"""
        # 取四边像素
        top = target[:, 0, :].mean(dim=1)
        bottom = target[:, -1, :].mean(dim=1)
        left = target[:, :, 0].mean(dim=1)
        right = target[:, :, -1].mean(dim=1)
        
        bg = (top + bottom + left + right) / 4.0
        return bg[0].item(), bg[1].item(), bg[2].item()
    
    def _find_closest_bg_index(self, bg_color: Tuple[float, float, float]) -> int:
        """找到最接近的背景色索引"""
        min_dist = float('inf')
        best_idx = 0
        for idx, bg_hex in enumerate(BACKGROUND_COLORS):
            r = int(bg_hex[1:3], 16) / 255.0
            g = int(bg_hex[3:5], 16) / 255.0
            b = int(bg_hex[5:7], 16) / 255.0
            dist = (r - bg_color[0])**2 + (g - bg_color[1])**2 + (b - bg_color[2])**2
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        return best_idx
    
    def _generate_base_color_actions(self, target: torch.Tensor,
                                      bg_color: Tuple[float, float, float]) -> List[List]:
        """
        生成铺底色 Action (destination-over)
        
        用大色块覆盖整个画布，建立整体色调。
        destination-over 模式确保底色在下层。
        """
        actions = []
        C, H, W = target.shape
        
        # 设置混合模式
        actions.append(["blend", BLEND_DESTINATION_OVER])
        actions.append(["alpha", 0.9])
        
        # 用区域笔刷(18)铺大色块
        patch_size = 64  # 大色块
        h_blocks = H // patch_size
        w_blocks = W // patch_size
        
        # 按亮度排序：暗色先画
        patches = []
        for r in range(h_blocks):
            for c in range(w_blocks):
                y1, y2 = r * patch_size, (r + 1) * patch_size
                x1, x2 = c * patch_size, (c + 1) * patch_size
                avg_color = target[:, y1:y2, x1:x2].mean(dim=(1, 2))
                lum = get_luminance(avg_color[0].item(), avg_color[1].item(), avg_color[2].item())
                patches.append((r, c, avg_color, lum))
        
        patches.sort(key=lambda x: x[3])  # 暗色优先
        
        for r, c, avg_color, lum in patches:
            y1, y2 = r * patch_size, (r + 1) * patch_size
            x1, x2 = c * patch_size, (c + 1) * patch_size
            
            color_hex = quantize_color(avg_color[0].item(), avg_color[1].item(), avg_color[2].item())
            actions.append(["colour", color_hex])
            actions.append(["width", patch_size / 2])
            actions.append(["line", [BRUSH_AREA, x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]])
        
        # 恢复混合模式
        actions.append(["blend", BLEND_SOURCE_OVER])
        actions.append(["alpha", 1.0])
        
        return actions
    
    def _generate_shadow_actions(self, target: torch.Tensor) -> List[List]:
        """生成阴影 Action (source-over, 低透明度)"""
        actions = []
        C, H, W = target.shape
        
        # 计算暗部区域
        gray = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
        dark_mask = gray < 0.4
        
        patch_size = 32
        h_blocks = H // patch_size
        w_blocks = W // patch_size
        
        actions.append(["blend", BLEND_SOURCE_OVER])
        actions.append(["alpha", 0.6])
        
        for r in range(h_blocks):
            for c in range(w_blocks):
                y1, y2 = r * patch_size, (r + 1) * patch_size
                x1, x2 = c * patch_size, (c + 1) * patch_size
                
                patch_dark = dark_mask[y1:y2, x1:x2].float().mean()
                if patch_dark > 0.3:  # 超过30%是暗部
                    avg_color = target[:, y1:y2, x1:x2].mean(dim=(1, 2))
                    color_hex = quantize_color(avg_color[0].item(), avg_color[1].item(), avg_color[2].item())
                    actions.append(["colour", color_hex])
                    actions.append(["width", 20])
                    
                    # 用马克笔(0)画阴影
                    cx1, cy1 = x1 + 5, y1 + 5
                    cx2, cy2 = x2 - 5, y2 - 5
                    actions.append(["line", [BRUSH_MARKER, cx1, cy1, cx2, cy2]])
        
        actions.append(["alpha", 1.0])
        return actions
    
    def _generate_shape_actions(self, target: torch.Tensor) -> List[List]:
        """生成形体刻画 Action (source-over, 压感笔刷)"""
        actions = []
        C, H, W = target.shape
        
        patch_size = 16
        h_blocks = H // patch_size
        w_blocks = W // patch_size
        
        actions.append(["blend", BLEND_SOURCE_OVER])
        actions.append(["alpha", 0.85])
        
        # 按亮度排序
        patches = []
        for r in range(h_blocks):
            for c in range(w_blocks):
                y1, y2 = r * patch_size, (r + 1) * patch_size
                x1, x2 = c * patch_size, (c + 1) * patch_size
                avg_color = target[:, y1:y2, x1:x2].mean(dim=(1, 2))
                lum = get_luminance(avg_color[0].item(), avg_color[1].item(), avg_color[2].item())
                patches.append((r, c, avg_color, lum))
        
        patches.sort(key=lambda x: x[3])
        
        for r, c, avg_color, lum in patches:
            y1, y2 = r * patch_size, (r + 1) * patch_size
            x1, x2 = c * patch_size, (c + 1) * patch_size
            
            color_hex = quantize_color(avg_color[0].item(), avg_color[1].item(), avg_color[2].item())
            actions.append(["colour", color_hex])
            actions.append(["width", 12])
            
            # 用压感v3(5)画笔画
            # 生成一条穿过色块的笔画，带压感
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            angle = np.random.uniform(0, 2 * np.pi)
            half_len = patch_size * 0.6
            
            px1 = int(cx - half_len * np.cos(angle))
            py1 = int(cy - half_len * np.sin(angle))
            px2 = int(cx)
            py2 = int(cy)
            px3 = int(cx + half_len * np.cos(angle))
            py3 = int(cy + half_len * np.sin(angle))
            
            # 压感: 起笔轻→中间重→收笔轻
            p1 = round(np.random.uniform(1, 3), 1)
            p2 = round(np.random.uniform(4, 7), 1)
            p3 = round(np.random.uniform(1, 3), 1)
            
            actions.append(["line", [BRUSH_PRESSURE, px1, py1, p1, px2, py2, p2, px3, py3, p3]])
        
        actions.append(["alpha", 1.0])
        return actions
    
    def _generate_line_art_actions(self, target: torch.Tensor) -> List[List]:
        """生成线稿 Action (source-over, 铅笔)"""
        actions = []
        C, H, W = target.shape
        
        # 简单边缘检测
        gray = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
        # Sobel
        gx = gray[1:, :] - gray[:-1, :]
        gy = gray[:, 1:] - gray[:, :-1]
        # Pad
        gx = F.pad(gx, (0, 0, 0, 1))
        gy = F.pad(gy, (0, 1, 0, 0))
        edge_mag = (gx ** 2 + gy ** 2).sqrt()
        
        # 在强边缘处画线
        patch_size = 8
        h_blocks = H // patch_size
        w_blocks = W // patch_size
        
        actions.append(["blend", BLEND_SOURCE_OVER])
        actions.append(["alpha", 0.9])
        actions.append(["colour", "#222222"])
        actions.append(["width", 3])
        
        for r in range(h_blocks):
            for c in range(w_blocks):
                y1, y2 = r * patch_size, (r + 1) * patch_size
                x1, x2 = c * patch_size, (c + 1) * patch_size
                
                patch_edge = edge_mag[y1:y2, x1:x2].mean()
                
                if patch_edge > 0.08:  # 强边缘
                    # 取边缘方向
                    avg_gx = gx[y1:y2, x1:x2].mean()
                    avg_gy = gy[y1:y2, x1:x2].mean()
                    angle = math.atan2(avg_gy, avg_gx)
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    half_len = patch_size * 0.5
                    
                    px1 = int(cx - half_len * math.cos(angle))
                    py1 = int(cy - half_len * math.sin(angle))
                    px2 = int(cx + half_len * math.cos(angle))
                    py2 = int(cy + half_len * math.sin(angle))
                    
                    # 铅笔压感
                    p1 = round(np.random.uniform(2, 4), 1)
                    p2 = round(np.random.uniform(3, 6), 1)
                    
                    actions.append(["line", [BRUSH_PENCIL, px1, py1, p1, px2, py2, p2]])
        
        actions.append(["alpha", 1.0])
        return actions
    
    def _generate_highlight_actions(self, target: torch.Tensor) -> List[List]:
        """生成高光 Action (source-over, 小色块)"""
        actions = []
        C, H, W = target.shape
        
        gray = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
        bright_mask = gray > 0.75
        
        patch_size = 16
        h_blocks = H // patch_size
        w_blocks = W // patch_size
        
        actions.append(["blend", BLEND_SOURCE_OVER])
        actions.append(["alpha", 0.7])
        
        for r in range(h_blocks):
            for c in range(w_blocks):
                y1, y2 = r * patch_size, (r + 1) * patch_size
                x1, x2 = c * patch_size, (c + 1) * patch_size
                
                patch_bright = bright_mask[y1:y2, x1:x2].float().mean()
                if patch_bright > 0.3:
                    avg_color = target[:, y1:y2, x1:x2].mean(dim=(1, 2))
                    color_hex = quantize_color(avg_color[0].item(), avg_color[1].item(), avg_color[2].item())
                    actions.append(["colour", color_hex])
                    actions.append(["width", 8])
                    
                    cx1, cy1 = x1 + 3, y1 + 3
                    cx2, cy2 = x2 - 3, y2 - 3
                    actions.append(["line", [BRUSH_MARKER, cx1, cy1, cx2, cy2]])
        
        actions.append(["alpha", 1.0])
        return actions
    
    def _generate_detail_actions(self, target: torch.Tensor) -> List[List]:
        """生成精修 Action (source-over, 细笔刷)"""
        actions = []
        C, H, W = target.shape
        
        patch_size = 8
        h_blocks = H // patch_size
        w_blocks = W // patch_size
        
        actions.append(["blend", BLEND_SOURCE_OVER])
        actions.append(["alpha", 0.95])
        
        for r in range(h_blocks):
            for c in range(w_blocks):
                y1, y2 = r * patch_size, (r + 1) * patch_size
                x1, x2 = c * patch_size, (c + 1) * patch_size
                
                avg_color = target[:, y1:y2, x1:x2].mean(dim=(1, 2))
                color_hex = quantize_color(avg_color[0].item(), avg_color[1].item(), avg_color[2].item())
                actions.append(["colour", color_hex])
                actions.append(["width", 6])
                
                # 用压感笔刷画短笔画
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                px1 = int(x1 + 2)
                py1 = int(y1 + 2)
                px2 = int(x2 - 2)
                py2 = int(y2 - 2)
                
                p1 = round(np.random.uniform(2, 5), 1)
                p2 = round(np.random.uniform(3, 6), 1)
                
                actions.append(["line", [BRUSH_PRESSURE, px1, py1, p1, px2, py2, p2]])
        
        actions.append(["alpha", 1.0])
        return actions
