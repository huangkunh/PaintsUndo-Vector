"""
多阶段优化器 - 高性能版本

优化：
1. 统一分辨率：所有阶段在同一分辨率优化，避免缩放开销
2. 简化管线：去掉重初始化开销
3. 使用高性能渲染器的 render_stroke_with_texture
4. 支持轻量模式（无 VGG）
5. 进度回调支持
"""

import os, time, logging
from typing import List, Optional, Tuple, Dict, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from brushes.base import BrushStroke
from core.renderer import DifferentiableRenderer
from core.losses import CombinedLoss
from utils.image import load_image, save_image
from utils.init import initialize_strokes_stage1, initialize_strokes_stage2, initialize_strokes_stage3

logger = logging.getLogger(__name__)


class StageConfig:
    def __init__(self, name="stage", num_strokes=20, max_width=50.0, min_width=20.0,
                 num_control_points=5, num_iterations=500, lr=1.0, lr_width=0.1,
                 lr_color=0.01, brush_names=None, lambda_pixel=1.0, lambda_perceptual=10.0,
                 lambda_ssim=2.0, lambda_color=1.0, lambda_length=0.01, lambda_smooth=0.005):
        self.name = name
        self.num_strokes = num_strokes
        self.max_width = max_width
        self.min_width = min_width
        self.num_control_points = num_control_points
        self.num_iterations = num_iterations
        self.lr = lr
        self.lr_width = lr_width
        self.lr_color = lr_color
        self.brush_names = brush_names or ["marker"]
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color
        self.lambda_length = lambda_length
        self.lambda_smooth = lambda_smooth


DEFAULT_STAGE_CONFIGS = [
    StageConfig("铺底色", num_strokes=20, max_width=50, min_width=20, num_control_points=5,
                num_iterations=500, lr=1.0, brush_names=["marker", "watercolor", "airbrush"]),
    StageConfig("形体刻画", num_strokes=100, max_width=15, min_width=5, num_control_points=7,
                num_iterations=800, lr=1.0, brush_names=["pressure", "marker"]),
    StageConfig("细节线稿", num_strokes=300, max_width=3, min_width=1, num_control_points=9,
                num_iterations=1000, lr=0.5, brush_names=["pencil", "pressure_sharp"]),
]


class MultiStageOptimizer:
    """高性能多阶段优化器"""
    
    def __init__(self, canvas_size=(640, 480), device="cpu", output_dir="output",
                 lightweight=False, progress_callback: Optional[Callable] = None):
        self.canvas_size = canvas_size
        self.device = device
        self.output_dir = output_dir
        self.lightweight = lightweight
        self.progress_callback = progress_callback
        os.makedirs(output_dir, exist_ok=True)
    
    def optimize(self, target_image: torch.Tensor,
                 stage_configs: Optional[List[StageConfig]] = None) -> Dict:
        if stage_configs is None:
            stage_configs = DEFAULT_STAGE_CONFIGS
        
        renderer = DifferentiableRenderer(canvas_size=self.canvas_size, device=self.device)
        loss_fn = CombinedLoss(device=self.device, lightweight=self.lightweight)
        loss_fn.cache_target(target_image)
        
        all_strokes = []
        all_brush_names = []
        stages_history = []
        prev_canvas = None
        
        for stage_idx, config in enumerate(stage_configs):
            print(f"\n{'='*50}")
            print(f"Stage {stage_idx+1}: {config.name} ({config.num_strokes} 条笔画)")
            print(f"{'='*50}")
            
            # 初始化笔画
            strokes, brush_names = self._init_strokes(target_image, config, stage_idx)
            
            # 锁定前一阶段参数
            for s in all_strokes:
                for p in s.parameters():
                    p.requires_grad = False
            
            # 优化
            best_canvas, best_loss = self._optimize_stage(
                strokes, brush_names, all_strokes, all_brush_names,
                target_image, renderer, loss_fn, config, stage_idx
            )
            
            all_strokes.extend(strokes)
            all_brush_names.extend(brush_names)
            prev_canvas = best_canvas
            
            # 保存
            save_image(best_canvas, os.path.join(self.output_dir, f"stage_{stage_idx+1}.png"))
            
            stages_history.append({
                "strokes": strokes,
                "brush_names": brush_names,
                "best_loss": best_loss,
            })
            
            print(f"✓ Stage {stage_idx+1} 完成, loss={best_loss:.4f}")
        
        # 最终渲染
        final_canvas = renderer.render_strokes(all_strokes, brush_name=all_brush_names[0])
        save_image(final_canvas, os.path.join(self.output_dir, "final_result.png"))
        
        return {
            "strokes": all_strokes,
            "brush_names": all_brush_names,
            "stages_history": stages_history,
            "final_canvas": final_canvas,
        }
    
    def _init_strokes(self, target, config, stage_idx):
        cs = self.canvas_size
        dev = self.device
        if stage_idx == 0:
            return initialize_strokes_stage1(target, config.num_strokes, config.max_width,
                                             config.min_width, config.num_control_points, cs, dev)
        elif stage_idx == 1:
            return initialize_strokes_stage2(target, config.num_strokes, config.max_width,
                                             config.min_width, config.num_control_points, cs, dev)
        else:
            return initialize_strokes_stage3(target, config.num_strokes, config.max_width,
                                             config.min_width, config.num_control_points, cs, dev)
    
    def _optimize_stage(self, strokes, brush_names, prev_strokes, prev_brushes,
                        target, renderer, loss_fn, config, stage_idx):
        # 收集可优化参数
        params = [p for s in strokes for p in s.parameters()]
        optimizer = optim.Adam(params, lr=config.lr)
        
        best_loss = float('inf')
        best_canvas = None
        n_iter = config.num_iterations
        
        for it in range(n_iter):
            optimizer.zero_grad()
            
            # 渲染所有笔画
            all_s = prev_strokes + strokes
            all_b = prev_brushes + brush_names
            
            # 使用高性能渲染
            canvas = renderer._create_canvas()
            for s, bn in zip(all_s, all_b):
                canvas = renderer.render_stroke_with_texture(s, canvas, bn)
            rendered = canvas[:3]
            
            # 损失
            total_loss, loss_dict = loss_fn(rendered, target, strokes, stage=stage_idx)
            
            if torch.isnan(total_loss):
                continue
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            # 钳制参数范围
            with torch.no_grad():
                for s in strokes:
                    s.raw_control_points.clamp_(-5, 5)
                    s.raw_width.clamp_(min=-2)
                    s.raw_opacity.clamp_(min=-3, max=2)
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_canvas = rendered.detach().clone()
            
            if it % max(1, n_iter // 10) == 0:
                print(f"  iter {it:4d}/{n_iter}: loss={total_loss.item():.4f} "
                      f"(pixel={loss_dict['pixel']:.4f})")
            
            if self.progress_callback:
                self.progress_callback(stage_idx, it, n_iter, loss_dict)
        
        return best_canvas, best_loss
