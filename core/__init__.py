"""
核心模块初始化

PaintsUndo-Vector 核心组件：
- renderer: 可微渲染器
- losses: 损失函数（纯 PyTorch VGG 感知损失，无需 lpips）
- optimizer: 多阶段优化器
- scheduler: 阶段调度器
- attention: 注意力引导（纯 PyTorch，无需 cv2）
- painting_sim: 人类绘画模拟
- cli: 命令行入口
"""

from core.renderer import DifferentiableRenderer
from core.losses import CombinedLoss, PixelLoss, VGGPerceptualLoss
from core.optimizer import MultiStageOptimizer, StageConfig, DEFAULT_STAGE_CONFIGS
from core.scheduler import StageScheduler
from core.attention import AttentionMap
from core.painting_sim import (
    sort_strokes_human_like,
    simulate_painting_rhythm,
    add_hand_tremor,
    mix_colors,
    generate_painting_narrative,
)
