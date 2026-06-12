"""
阶段调度器 - 管理多阶段优化的执行流程

支持自定义阶段配置、动态调整参数、早停策略等。
"""

from typing import List, Optional, Tuple, Dict

from core.optimizer import StageConfig, DEFAULT_STAGE_CONFIGS


class StageScheduler:
    """
    阶段调度器
    
    管理多阶段优化的执行流程，支持：
    - 自定义阶段配置
    - 动态调整参数
    - 早停策略
    - 阶段间参数传递
    """
    
    def __init__(
        self,
        stage_configs: Optional[List[StageConfig]] = None,
        early_stopping_patience: int = 50,
        early_stopping_threshold: float = 0.001,
    ):
        self.stage_configs = stage_configs or DEFAULT_STAGE_CONFIGS
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
    
    def should_stop_early(self, loss_history: List[float]) -> bool:
        """
        判断是否应该早停。
        
        如果连续 patience 次迭代损失下降不超过 threshold，则早停。
        """
        if len(loss_history) < self.early_stopping_patience:
            return False
        
        recent_losses = loss_history[-self.early_stopping_patience:]
        improvement = recent_losses[0] - recent_losses[-1]
        
        return improvement < self.early_stopping_threshold
    
    def get_stage_config(self, stage_idx: int) -> StageConfig:
        """获取指定阶段的配置"""
        if stage_idx < len(self.stage_configs):
            return self.stage_configs[stage_idx]
        # 如果超出预设阶段，返回最后一个阶段的配置（降低学习率）
        last_config = self.stage_configs[-1]
        return StageConfig(
            name=f"stage_{stage_idx + 1}_extra",
            num_strokes=last_config.num_strokes,
            max_width=last_config.max_width * 0.5,
            min_width=last_config.min_width * 0.5,
            num_control_points=last_config.num_control_points,
            resolution=last_config.resolution,
            num_iterations=last_config.num_iterations,
            lr=last_config.lr * 0.5,
            lr_width=last_config.lr_width * 0.5,
            lr_color=last_config.lr_color * 0.5,
            brush_names=last_config.brush_names,
        )
    
    def adapt_config_from_loss(
        self,
        config: StageConfig,
        loss_dict: Dict[str, float],
    ) -> StageConfig:
        """
        根据损失情况动态调整配置。
        
        如果感知损失较高，增加笔画数；如果像素损失较高，增加学习率。
        """
        # 简单的自适应策略
        if loss_dict.get("perceptual", 0) > 0.5:
            config.num_strokes = min(config.num_strokes + 10, 500)
        
        if loss_dict.get("pixel", 0) > 0.1:
            config.lr = min(config.lr * 1.1, 2.0)
        
        return config
