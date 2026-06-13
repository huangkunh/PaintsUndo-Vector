"""
语义绘画模块

将"操作语义"显式化的完整绘画系统：
1. 语义图层系统
2. 行为损失
3. 人类绘画模拟
4. 行为级回放
5. 手绘感增强
"""

from core.semantic.layers import (
    LayerType, BlendMode, OperationType,
    PaintLayer, PaintingPlan, SemanticStroke,
    PaintingPlanBuilder,
    HUMAN_LAYER_ORDER, LAYER_BLEND_MAP, LAYER_ALPHA_MAP, LAYER_BRUSH_MAP,
)
from core.semantic.behavior_loss import HumanFrameGenerator, BehaviorLoss, MultiStageLoss
from core.semantic.human_sim import (
    PaintingRhythm, RhythmEvent,
    CorrectionStrategy,
    BehaviorReplay,
    HandDrawnEffect,
)
