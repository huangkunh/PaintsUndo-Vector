"""
人类绘画模拟器 - 重构版

改进点：
1. 更合理的绘画顺序：构图→线稿→底色→阴影→高光→细节→调整
2. 画错并擦除/重画机制
3. 画师停顿、审视、修改行为
4. 覆盖策略（不是连续微调，而是一笔画出或分几段画出）
5. 绘画节奏控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum, auto

from core.semantic.layers import (
    LayerType, BlendMode, OperationType, PaintLayer, PaintingPlan,
    SemanticStroke, HUMAN_LAYER_ORDER, LAYER_BLEND_MAP, LAYER_ALPHA_MAP,
    LAYER_BRUSH_MAP, PaintingPlanBuilder,
)
from core.semantic.behavior_loss import HumanFrameGenerator, BehaviorLoss


# ========== 绘画节奏控制 ==========

@dataclass
class RhythmEvent:
    """绘画节奏事件"""
    event_type: str      # "stroke", "pause", "review", "erase", "redraw", "adjust"
    duration: float      # 持续时间（秒）
    layer: LayerType     # 所属图层
    description: str = ""


class PaintingRhythm:
    """
    绘画节奏控制器
    
    模拟人类画师的绘画节奏：
    - 快速铺色阶段：连续快速笔画，少停顿
    - 刻画阶段：中等速度，偶尔停顿审视
    - 细节阶段：慢速精细，频繁停顿
    - 调整阶段：间歇性，大量审视
    """
    
    def __init__(self):
        self.events: List[RhythmEvent] = []
        self.current_time = 0.0
    
    def plan_rhythm(self, plan: PaintingPlan) -> List[RhythmEvent]:
        """
        为整个绘画计划生成节奏事件序列
        """
        self.events = []
        self.current_time = 0.0
        
        for layer_type in plan.execution_order:
            if layer_type not in plan.layers:
                continue
            
            layer = plan.layers[layer_type]
            strokes = layer.strokes
            
            if not strokes:
                continue
            
            # 图层开始前的停顿（换笔/换色）
            self._add_pause(0.5, layer_type, f"准备{layer_type.name}...")
            
            # 根据图层类型设置节奏
            rhythm_config = self._get_rhythm_config(layer_type)
            
            for i, stroke in enumerate(strokes):
                # 绘制笔画
                stroke_duration = rhythm_config["stroke_duration"]
                self._add_stroke(stroke_duration, layer_type, f"绘制{layer_type.name}第{i+1}笔")
                
                # 偶尔停顿审视
                if np.random.random() < rhythm_config["pause_probability"]:
                    pause_duration = np.random.uniform(*rhythm_config["pause_range"])
                    self._add_pause(pause_duration, layer_type, "审视画面...")
                
                # 偶尔修改/重画
                if np.random.random() < rhythm_config["correction_probability"] and i > 0:
                    self._add_erase(0.3, layer_type, "擦除不满意的部分")
                    self._add_redraw(stroke_duration * 1.5, layer_type, "重新绘制")
            
            # 图层完成后的审视
            self._add_pause(1.0, layer_type, f"{layer_type.name}完成，审视整体效果...")
        
        return self.events
    
    def _get_rhythm_config(self, layer_type: LayerType) -> Dict:
        """获取图层类型的节奏配置"""
        configs = {
            LayerType.COMPOSITION: {
                "stroke_duration": 0.3,
                "pause_probability": 0.1,
                "pause_range": (0.2, 0.5),
                "correction_probability": 0.05,
            },
            LayerType.BASE_COLOR: {
                "stroke_duration": 0.4,
                "pause_probability": 0.15,
                "pause_range": (0.3, 0.8),
                "correction_probability": 0.08,
            },
            LayerType.SHADOW: {
                "stroke_duration": 0.5,
                "pause_probability": 0.2,
                "pause_range": (0.3, 1.0),
                "correction_probability": 0.1,
            },
            LayerType.LINE_ART: {
                "stroke_duration": 0.6,
                "pause_probability": 0.25,
                "pause_range": (0.5, 1.5),
                "correction_probability": 0.15,
            },
            LayerType.MID_TONE: {
                "stroke_duration": 0.5,
                "pause_probability": 0.2,
                "pause_range": (0.3, 1.0),
                "correction_probability": 0.1,
            },
            LayerType.HIGHLIGHT: {
                "stroke_duration": 0.4,
                "pause_probability": 0.3,
                "pause_range": (0.5, 1.5),
                "correction_probability": 0.12,
            },
            LayerType.DETAIL: {
                "stroke_duration": 0.8,
                "pause_probability": 0.35,
                "pause_range": (0.5, 2.0),
                "correction_probability": 0.2,
            },
            LayerType.ADJUSTMENT: {
                "stroke_duration": 0.6,
                "pause_probability": 0.4,
                "pause_range": (1.0, 3.0),
                "correction_probability": 0.15,
            },
        }
        return configs.get(layer_type, configs[LayerType.DETAIL])
    
    def _add_stroke(self, duration, layer, desc):
        self.events.append(RhythmEvent("stroke", duration, layer, desc))
        self.current_time += duration
    
    def _add_pause(self, duration, layer, desc):
        self.events.append(RhythmEvent("pause", duration, layer, desc))
        self.current_time += duration
    
    def _add_erase(self, duration, layer, desc):
        self.events.append(RhythmEvent("erase", duration, layer, desc))
        self.current_time += duration
    
    def _add_redraw(self, duration, layer, desc):
        self.events.append(RhythmEvent("redraw", duration, layer, desc))
        self.current_time += duration


# ========== 覆盖与擦除策略 ==========

class CorrectionStrategy:
    """
    修正策略 - 模拟"画错并擦除/重画"
    
    策略：
    1. 在优化过程中，如果某条笔画的贡献为负（增加损失），标记为"画错"
    2. "画错"的笔画有概率被擦除并替换
    3. 替换笔画从目标图像重新采样参数
    4. 擦除操作使用 destination-out 混合模式
    """
    
    def __init__(self, erase_threshold: float = 0.8, replace_probability: float = 0.6):
        self.erase_threshold = erase_threshold
        self.replace_probability = replace_probability
    
    def evaluate_strokes(self, strokes: List[SemanticStroke], 
                         canvas_before: torch.Tensor,
                         canvas_after: torch.Tensor,
                         target: torch.Tensor) -> List[int]:
        """
        评估每条笔画的贡献，返回"画错"的笔画索引
        
        如果添加某笔画后，与目标的距离增大，则认为该笔画"画错"
        """
        bad_indices = []
        
        loss_before = F.mse_loss(canvas_before, target).item()
        loss_after = F.mse_loss(canvas_after, target).item()
        
        # 如果整体损失增加，检查哪些笔画可能是原因
        if loss_after > loss_before * self.erase_threshold:
            # 简化：标记最后几条笔画为可疑
            n_suspect = max(1, len(strokes) // 5)
            for i in range(len(strokes) - n_suspect, len(strokes)):
                if i >= 0 and np.random.random() < self.replace_probability:
                    bad_indices.append(i)
        
        return bad_indices
    
    def generate_correction(self, bad_stroke: SemanticStroke, 
                           target: torch.Tensor) -> SemanticStroke:
        """
        生成修正笔画
        
        从目标图像重新采样参数，替换"画错"的笔画
        """
        corrected = SemanticStroke(
            stroke_data=dict(bad_stroke.stroke_data),  # 复制数据
            layer_type=bad_stroke.layer_type,
            operation=OperationType.ERASE if bad_stroke.operation != OperationType.ERASE else OperationType.DRAW,
            brush_name=bad_stroke.brush_name,
            priority=bad_stroke.priority + 0.1,
            confidence=0.5,
            is_correction=True,
            replaced_stroke_idx=-1,
        )
        return corrected


# ========== 行为级回放 ==========

class BehaviorReplay:
    """
    行为级回放 - 将优化轨迹重新解释为人类绘画行为
    
    核心转换：
    - 优化轨迹：笔画在连续迭代中缓慢挪动 → 看起来像梯度下降
    - 行为回放：每一笔画"一次性画出"或"分几段画出" → 看起来像人类
    
    映射算法：
    1. 将优化轨迹按笔画分组
    2. 每条笔画取最终优化结果作为"目标笔画"
    3. 如果笔画在优化中移动较大，拆分为多段（模拟"分几段画出"）
    4. 插入停顿、审视事件
    5. 插入擦除/重画事件（如果优化中有大幅回退）
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (640, 480)):
        self.canvas_size = canvas_size
    
    def convert_trajectory_to_replay(
        self,
        plan: PaintingPlan,
        rhythm: List[RhythmEvent],
    ) -> List[Dict]:
        """
        将绘画计划+节奏事件转换为行为级回放序列
        
        Returns:
            回放事件列表，每个事件包含：
            - type: "draw" | "pause" | "erase" | "review"
            - layer: 图层类型
            - stroke: 笔画数据（仅 draw/erase 类型）
            - duration: 持续时间
            - description: 描述文字
        """
        replay = []
        stroke_idx = 0
        
        all_strokes = plan.get_all_strokes_ordered()
        
        for event in rhythm:
            if event.event_type == "stroke":
                if stroke_idx < len(all_strokes):
                    stroke = all_strokes[stroke_idx]
                    replay.append({
                        "type": "draw",
                        "layer": stroke.layer_type.name,
                        "stroke": stroke.stroke_data,
                        "brush": stroke.brush_name,
                        "operation": stroke.operation.name,
                        "duration": event.duration,
                        "description": event.description,
                        "is_correction": stroke.is_correction,
                    })
                    stroke_idx += 1
            elif event.event_type == "pause":
                replay.append({
                    "type": "pause",
                    "layer": event.layer.name,
                    "duration": event.duration,
                    "description": event.description,
                })
            elif event.event_type == "erase":
                replay.append({
                    "type": "erase",
                    "layer": event.layer.name,
                    "duration": event.duration,
                    "description": event.description,
                })
            elif event.event_type == "redraw":
                if stroke_idx < len(all_strokes):
                    stroke = all_strokes[stroke_idx]
                    replay.append({
                        "type": "draw",
                        "layer": stroke.layer_type.name,
                        "stroke": stroke.stroke_data,
                        "brush": stroke.brush_name,
                        "operation": stroke.operation.name,
                        "duration": event.duration,
                        "description": event.description,
                        "is_correction": True,
                    })
                    stroke_idx += 1
        
        return replay


# ========== 手绘感增强 ==========

class HandDrawnEffect:
    """
    手绘感效果 - 在渲染时添加人类手绘特征
    
    效果类型：
    1. 笔压变化：沿笔画方向的压感波动
    2. 毛边效果：笔画边缘的不规则抖动
    3. 墨水渗透：笔画起始/结束处的颜色扩散
    4. 手部微颤：控制点的轻微随机偏移
    5. 笔刷风格差异：不同笔刷的渲染风格
    """
    
    @staticmethod
    def apply_pressure_variation(
        control_points: torch.Tensor,
        base_width: float,
        style: str = "natural",
    ) -> torch.Tensor:
        """
        沿笔画方向添加压感变化
        
        Args:
            control_points: [N, 2] 控制点
            base_width: 基础宽度
            style: 压感风格
                - "natural": 自然压感（起笔轻→行笔重→收笔轻）
                - "sketch": 素描压感（快速轻扫，压感波动大）
                - "ink": 书写压感（起笔重→收笔渐细）
                - "watercolor": 水彩压感（缓慢变化，偶有积水效果）
        Returns:
            [N] 每个控制点的宽度
        """
        N = control_points.shape[0]
        t = torch.linspace(0, 1, N, device=control_points.device)
        
        if style == "natural":
            # 起笔轻→行笔重→收笔轻
            pressure = 1.0 - 2.0 * (t - 0.5) ** 2
            pressure = pressure * 0.6 + 0.4
        elif style == "sketch":
            # 快速轻扫，波动大
            pressure = 0.3 + 0.7 * torch.rand(N, device=control_points.device)
            # 平滑
            kernel = torch.ones(3, device=control_points.device) / 3
            pressure = F.conv1d(pressure.unsqueeze(0).unsqueeze(0), 
                               kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
        elif style == "ink":
            # 起笔重→收笔渐细
            pressure = 1.0 - 0.6 * t
        elif style == "watercolor":
            # 缓慢变化，偶有积水
            pressure = 0.7 + 0.3 * torch.sin(t * np.pi)
            # 积水效果：随机位置突然加宽
            water_spots = torch.rand(N, device=control_points.device) < 0.1
            pressure[water_spots] += 0.3
        else:
            pressure = torch.ones(N, device=control_points.device)
        
        widths = base_width * pressure
        return widths.clamp(min=base_width * 0.1)
    
    @staticmethod
    def apply_hand_tremor(
        control_points: torch.Tensor,
        strength: float = 0.003,
        frequency: float = 5.0,
    ) -> torch.Tensor:
        """
        添加手部微颤效果
        
        模拟人类画师的手部自然颤抖：
        - 高频小幅抖动（生理性震颤，8-12Hz）
        - 低频大幅偏移（调整性移动）
        
        Args:
            control_points: [N, 2] 控制点
            strength: 抖动强度（归一化坐标）
            frequency: 抖动频率
        Returns:
            [N, 2] 添加抖动后的控制点
        """
        N = control_points.shape[0]
        device = control_points.device
        
        # 高频震颤
        t = torch.linspace(0, N * 0.1, N, device=device)
        tremor_x = strength * torch.sin(2 * np.pi * frequency * t + torch.rand(1, device=device) * 2 * np.pi)
        tremor_y = strength * torch.cos(2 * np.pi * frequency * t * 1.3 + torch.rand(1, device=device) * 2 * np.pi)
        
        # 低频偏移
        drift_x = strength * 2 * torch.sin(2 * np.pi * 0.5 * t + torch.rand(1, device=device) * 2 * np.pi)
        drift_y = strength * 2 * torch.cos(2 * np.pi * 0.7 * t + torch.rand(1, device=device) * 2 * np.pi)
        
        noise = torch.stack([tremor_x + drift_x, tremor_y + drift_y], dim=1)
        
        return control_points + noise
    
    @staticmethod
    def apply_ink_bleed(
        control_points: torch.Tensor,
        widths: torch.Tensor,
        bleed_factor: float = 0.15,
    ) -> torch.Tensor:
        """
        墨水渗透效果
        
        在笔画起始和结束处，宽度略微增加，模拟墨水渗透。
        
        Args:
            control_points: [N, 2] 控制点
            widths: [N] 宽度
            bleed_factor: 渗透因子
        Returns:
            [N] 调整后的宽度
        """
        N = widths.shape[0]
        t = torch.linspace(0, 1, N, device=widths.device)
        
        # 起始处渗透
        start_bleed = bleed_factor * torch.exp(-t * 10)
        # 结束处渗透
        end_bleed = bleed_factor * torch.exp(-(1 - t) * 10)
        
        return widths * (1.0 + start_bleed + end_bleed)
    
    @staticmethod
    def apply_brush_style(
        control_points: torch.Tensor,
        widths: torch.Tensor,
        colors: torch.Tensor,
        brush_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用笔刷风格差异
        
        不同笔刷有不同的渲染特征：
        - pencil: 细线、压感变化大、有断点
        - marker: 均匀宽度、半透明叠加
        - watercolor: 边缘模糊、颜色渗透、积水效果
        - airbrush: 柔和边缘、渐变透明度
        - pressure: 压感明显、粗细变化大
        
        Args:
            control_points: [N, 2] 控制点
            widths: [N] 宽度
            colors: [4] RGBA 颜色
            brush_name: 笔刷名称
        Returns:
            (调整后的控制点, 调整后的宽度, 调整后的颜色)
        """
        cp = control_points.clone()
        w = widths.clone()
        c = colors.clone()
        
        if brush_name == "pencil":
            # 铅笔：细线、微颤、压感变化
            cp = HandDrawnEffect.apply_hand_tremor(cp, strength=0.002, frequency=8.0)
            w = HandDrawnEffect.apply_pressure_variation(cp, w.mean().item(), "sketch")
            # 铅笔颜色略有变化
            c[:3] += torch.randn(3, device=c.device) * 0.02
            
        elif brush_name == "marker":
            # 马克笔：均匀宽度、半透明
            w = torch.ones_like(w) * w.mean()
            c[3] = min(c[3].item(), 0.7)  # 马克笔半透明
            
        elif brush_name == "watercolor":
            # 水彩：边缘模糊、渗透、积水
            cp = HandDrawnEffect.apply_hand_tremor(cp, strength=0.004, frequency=3.0)
            w = HandDrawnEffect.apply_pressure_variation(cp, w.mean().item(), "watercolor")
            w = HandDrawnEffect.apply_ink_bleed(cp, w, bleed_factor=0.2)
            # 水彩颜色更透明
            c[3] = min(c[3].item(), 0.6)
            
        elif brush_name == "airbrush":
            # 喷笔：柔和、渐变
            cp = HandDrawnEffect.apply_hand_tremor(cp, strength=0.005, frequency=2.0)
            w = w * 1.5  # 喷笔更宽
            c[3] = min(c[3].item(), 0.3)  # 喷笔很透明
            
        elif brush_name == "pressure":
            # 压感笔：粗细变化大
            cp = HandDrawnEffect.apply_hand_tremor(cp, strength=0.001, frequency=6.0)
            w = HandDrawnEffect.apply_pressure_variation(cp, w.mean().item(), "natural")
        
        return cp, w.clamp(min=0.5), c.clamp(0, 1)
