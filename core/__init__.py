"""核心模块"""

from core.renderer import DifferentiableRenderer
from core.losses import CombinedLoss, PixelLoss, VGGPerceptualLoss
from core.optimizer import MultiStageOptimizer, StageConfig, DEFAULT_STAGE_CONFIGS
from core.attention import AttentionMap
from core.painting_sim import sort_strokes_human_like, add_hand_tremor, generate_painting_narrative
from core.direct_painter import multiscale_paint, compute_ssim, run_full_pipeline
from core.human_painter import HumanPainter, run_human_painting
