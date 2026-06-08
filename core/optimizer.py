"""
多阶段优化器 - 核心优化管线

实现方案B的渐进式优化策略：
- Stage 1: 铺底色（低分辨率，粗线条，少量笔画）
- Stage 2: 形体刻画（中分辨率，中等线条，较多笔画）
- Stage 3: 细节线稿（高分辨率，细线条，大量笔画）

每个阶段锁定前一阶段的参数，只优化当前阶段的笔画。
"""

import os
import time
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from brushes.base import BrushStroke
from core.renderer import DifferentiableRenderer
from core.losses import CombinedLoss
from utils.image import load_image, save_image, resize_image
from utils.init import (
    initialize_strokes_stage1,
    initialize_strokes_stage2,
    initialize_strokes_stage3,
)


class StageConfig:
    """单个阶段的配置"""
    
    def __init__(
        self,
        name: str,
        num_strokes: int = 20,
        max_width: float = 50.0,
        min_width: float = 20.0,
        num_control_points: int = 5,
        resolution: Tuple[int, int] = (256, 256),
        num_iterations: int = 500,
        lr: float = 1.0,
        lr_width: float = 0.1,
        lr_color: float = 0.01,
        brush_names: List[str] = None,
        lambda_pixel: float = 1.0,
        lambda_perceptual: float = 10.0,
        lambda_length: float = 0.01,
        lambda_smooth: float = 0.005,
    ):
        self.name = name
        self.num_strokes = num_strokes
        self.max_width = max_width
        self.min_width = min_width
        self.num_control_points = num_control_points
        self.resolution = resolution
        self.num_iterations = num_iterations
        self.lr = lr
        self.lr_width = lr_width
        self.lr_color = lr_color
        self.brush_names = brush_names or ["marker"]
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_length = lambda_length
        self.lambda_smooth = lambda_smooth


# 默认三阶段配置
DEFAULT_STAGE_CONFIGS = [
    StageConfig(
        name="stage1_base_color",
        num_strokes=20,
        max_width=50.0,
        min_width=20.0,
        num_control_points=5,
        resolution=(256, 256),
        num_iterations=500,
        lr=1.0,
        lr_width=0.1,
        lr_color=0.01,
        brush_names=["marker", "airbrush"],
        lambda_pixel=1.0,
        lambda_perceptual=10.0,
        lambda_length=0.01,
        lambda_smooth=0.005,
    ),
    StageConfig(
        name="stage2_shape",
        num_strokes=100,
        max_width=15.0,
        min_width=5.0,
        num_control_points=7,
        resolution=(512, 512),
        num_iterations=800,
        lr=0.5,
        lr_width=0.05,
        lr_color=0.005,
        brush_names=["pressure", "watercolor"],
        lambda_pixel=1.0,
        lambda_perceptual=10.0,
        lambda_length=0.02,
        lambda_smooth=0.01,
    ),
    StageConfig(
        name="stage3_detail",
        num_strokes=300,
        max_width=3.0,
        min_width=1.0,
        num_control_points=5,
        resolution=(1024, 1024),
        num_iterations=1000,
        lr=0.2,
        lr_width=0.02,
        lr_color=0.002,
        brush_names=["pencil", "hatching"],
        lambda_pixel=1.0,
        lambda_perceptual=10.0,
        lambda_length=0.03,
        lambda_smooth=0.015,
    ),
]


class MultiStageOptimizer:
    """
    多阶段渐进式优化器
    
    按照方案B的三阶段策略，逐步优化笔画参数。
    每个阶段完成后锁定参数，记录到历史中。
    """
    
    def __init__(
        self,
        target_image_path: str,
        output_dir: str = "./output",
        canvas_size: Tuple[int, int] = (640, 480),
        device: str = "cpu",
        stage_configs: Optional[List[StageConfig]] = None,
    ):
        self.target_image_path = target_image_path
        self.output_dir = output_dir
        self.canvas_size = canvas_size
        self.device = device
        
        self.stage_configs = stage_configs or DEFAULT_STAGE_CONFIGS
        
        # 加载目标图像
        self.target_image_full = load_image(target_image_path, canvas_size, device)
        
        # 绘画历史：记录每个阶段的笔画
        self.stages_history: List[Dict] = []
        
        # 所有已锁定的笔画（不参与梯度更新）
        self.locked_strokes: List[BrushStroke] = []
        self.locked_brush_names: List[str] = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def optimize(self) -> List[Dict]:
        """
        执行完整的多阶段优化。
        
        Returns:
            stages_history: 每个阶段的笔画历史
        """
        print(f"Starting multi-stage optimization for: {self.target_image_path}")
        print(f"Canvas size: {self.canvas_size}")
        print(f"Device: {self.device}")
        print(f"Number of stages: {len(self.stage_configs)}")
        print("=" * 60)
        
        for stage_idx, config in enumerate(self.stage_configs):
            print(f"\n{'=' * 60}")
            print(f"Stage {stage_idx + 1}: {config.name}")
            print(f"  Strokes: {config.num_strokes}")
            print(f"  Resolution: {config.resolution}")
            print(f"  Iterations: {config.num_iterations}")
            print(f"  Brushes: {config.brush_names}")
            print(f"{'=' * 60}")
            
            stage_result = self._optimize_stage(stage_idx, config)
            self.stages_history.append(stage_result)
            
            # 锁定当前阶段的笔画
            for stroke in stage_result["strokes"]:
                stroke.requires_grad_(False)
                self.locked_strokes.append(stroke)
            self.locked_brush_names.extend(stage_result["brush_names"])
            
            # 保存中间结果
            self._save_stage_result(stage_idx, stage_result)
        
        print("\n" + "=" * 60)
        print("Optimization complete!")
        print(f"Total strokes: {len(self.locked_strokes)}")
        print(f"Results saved to: {self.output_dir}")
        
        return self.stages_history
    
    def _optimize_stage(self, stage_idx: int, config: StageConfig) -> Dict:
        """
        优化单个阶段。
        """
        # 调整目标图像到当前阶段分辨率
        target_resized = resize_image(self.target_image_full, config.resolution)
        
        # 初始化当前阶段的笔画
        init_func = [initialize_strokes_stage1, initialize_strokes_stage2, initialize_strokes_stage3]
        strokes, brush_names = init_func[stage_idx](
            target_image=target_resized,
            num_strokes=config.num_strokes,
            max_width=config.max_width,
            min_width=config.min_width,
            num_control_points=config.num_control_points,
            canvas_size=config.resolution,
            device=self.device,
        )
        
        # 创建渲染器
        renderer = DifferentiableRenderer(
            canvas_size=config.resolution,
            device=self.device,
        )
        
        # 创建损失函数
        loss_fn = CombinedLoss(
            lambda_pixel=config.lambda_pixel,
            lambda_perceptual=config.lambda_perceptual,
            lambda_length=config.lambda_length,
            lambda_smooth=config.lambda_smooth,
            device=self.device,
        )
        
        # 创建优化器 - 分组学习率
        params_groups = self._create_param_groups(strokes, config)
        optimizer = optim.Adam(params_groups)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_iterations, eta_min=config.lr * 0.01
        )
        
        # 训练循环
        loss_history = []
        best_loss = float('inf')
        best_strokes_state = None
        
        pbar = tqdm(range(config.num_iterations), desc=f"Stage {stage_idx + 1}")
        for iteration in pbar:
            optimizer.zero_grad()
            
            # 渲染当前阶段的笔画
            # 将已锁定笔画和当前阶段笔画一起渲染
            all_strokes = self.locked_strokes + strokes
            all_brush_names = self.locked_brush_names + brush_names
            
            # 按笔刷分组渲染
            stroke_groups = self._group_strokes_by_brush(all_strokes, all_brush_names)
            rendered = renderer.render_multi_brush(stroke_groups)
            
            # 如果分辨率不同，调整渲染结果
            if rendered.shape[1:] != target_resized.shape[1:]:
                rendered = resize_image(rendered, config.resolution)
            
            # 计算损失
            total_loss, loss_dict = loss_fn(rendered, target_resized, strokes)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for s in strokes for p in s.parameters()], max_norm=1.0
            )
            
            optimizer.step()
            scheduler.step()
            
            # 记录损失
            loss_history.append(loss_dict)
            
            # 保存最佳状态
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_strokes_state = {
                    i: {k: v.clone() for k, v in s.state_dict().items()}
                    for i, s in enumerate(strokes)
                }
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.4f}",
                "pixel": f"{loss_dict['pixel']:.4f}",
                "perc": f"{loss_dict['perceptual']:.4f}",
            })
            
            # 定期保存中间结果
            if (iteration + 1) % 100 == 0:
                self._save_iteration_result(
                    stage_idx, iteration, rendered, target_resized
                )
        
        # 恢复最佳状态
        if best_strokes_state is not None:
            for i, stroke in enumerate(strokes):
                if i in best_strokes_state:
                    stroke.load_state_dict(best_strokes_state[i])
        
        return {
            "strokes": strokes,
            "brush_names": brush_names,
            "loss_history": loss_history,
            "best_loss": best_loss,
            "config": config,
        }
    
    def _create_param_groups(
        self, strokes: List[BrushStroke], config: StageConfig
    ) -> List[Dict]:
        """创建分组学习率的参数组"""
        point_params = []
        width_params = []
        color_params = []
        opacity_params = []
        
        for stroke in strokes:
            point_params.append(stroke.raw_control_points)
            width_params.append(stroke.raw_width)
            color_params.append(stroke.raw_color)
            opacity_params.append(stroke.raw_opacity)
        
        return [
            {"params": point_params, "lr": config.lr},
            {"params": width_params, "lr": config.lr_width},
            {"params": color_params, "lr": config.lr_color},
            {"params": opacity_params, "lr": config.lr_color},
        ]
    
    def _group_strokes_by_brush(
        self, strokes: List[BrushStroke], brush_names: List[str]
    ) -> List[Tuple[str, List[BrushStroke]]]:
        """按笔刷类型分组笔画"""
        groups: Dict[str, List[BrushStroke]] = {}
        for stroke, name in zip(strokes, brush_names):
            if name not in groups:
                groups[name] = []
            groups[name].append(stroke)
        return list(groups.items())
    
    def _save_stage_result(self, stage_idx: int, result: Dict):
        """保存阶段结果"""
        stage_dir = os.path.join(self.output_dir, f"stage_{stage_idx + 1}")
        os.makedirs(stage_dir, exist_ok=True)
        
        # 保存最终渲染结果
        renderer = DifferentiableRenderer(
            canvas_size=result["config"].resolution,
            device=self.device,
        )
        
        all_strokes = self.locked_strokes + result["strokes"]
        all_brush_names = self.locked_brush_names + result["brush_names"]
        stroke_groups = self._group_strokes_by_brush(all_strokes, all_brush_names)
        rendered = renderer.render_multi_brush(stroke_groups)
        
        save_image(rendered, os.path.join(stage_dir, "final_render.png"))
        
        # 保存笔画参数
        strokes_data = []
        for stroke, brush_name in zip(result["strokes"], result["brush_names"]):
            strokes_data.append({
                "control_points": stroke.control_points.detach().cpu().numpy().tolist(),
                "width": stroke.width.detach().cpu().item(),
                "color": stroke.color.detach().cpu().numpy().tolist(),
                "opacity": stroke.opacity.detach().cpu().item(),
                "brush_name": brush_name,
            })
        
        import json
        with open(os.path.join(stage_dir, "strokes.json"), "w") as f:
            json.dump(strokes_data, f, indent=2)
        
        print(f"  Stage {stage_idx + 1} results saved to: {stage_dir}")
    
    def _save_iteration_result(
        self,
        stage_idx: int,
        iteration: int,
        rendered: torch.Tensor,
        target: torch.Tensor,
    ):
        """保存迭代中间结果"""
        iter_dir = os.path.join(self.output_dir, f"stage_{stage_idx + 1}", "iterations")
        os.makedirs(iter_dir, exist_ok=True)
        
        from utils.vis import create_comparison_image
        comparison = create_comparison_image(target, rendered)
        comparison.save(os.path.join(iter_dir, f"iter_{iteration:04d}.png"))
