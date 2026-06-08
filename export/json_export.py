"""
JSON 导出模块

将优化后的笔画参数导出为 JSON 文件，
方便后续回放和二次开发。
"""

import os
import json
from typing import List, Optional, Tuple, Dict

import numpy as np

from brushes.base import BrushStroke


class JSONExporter:
    """
    JSON 导出器
    
    将笔画列表导出为 JSON 文件，包含：
    - 笔画参数（控制点、宽度、颜色、透明度）
    - 笔刷类型
    - 阶段信息
    - 画布设置
    """
    
    def export(
        self,
        strokes: List[BrushStroke],
        brush_names: List[str],
        output_path: str,
        canvas_size: Tuple[int, int] = (640, 480),
        stage_info: Optional[Dict] = None,
    ):
        """导出笔画为 JSON 文件"""
        data = {
            "canvas_size": list(canvas_size),
            "num_strokes": len(strokes),
            "strokes": [],
        }
        
        if stage_info:
            data["stage_info"] = stage_info
        
        for stroke, brush_name in zip(strokes, brush_names):
            stroke_data = {
                "brush_name": brush_name,
                "control_points": stroke.control_points.detach().cpu().numpy().tolist(),
                "width": stroke.width.detach().cpu().item(),
                "color": stroke.color.detach().cpu().numpy().tolist(),
                "opacity": stroke.opacity.detach().cpu().item(),
                "num_control_points": stroke.num_control_points,
            }
            data["strokes"].append(stroke_data)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON exported to: {output_path}")
    
    def export_stages(
        self,
        stages_history: List[Dict],
        output_dir: str,
        canvas_size: Tuple[int, int] = (640, 480),
    ):
        """分阶段导出 JSON 文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        accumulated_strokes = []
        accumulated_brushes = []
        
        for stage_idx, stage_data in enumerate(stages_history):
            accumulated_strokes.extend(stage_data["strokes"])
            accumulated_brushes.extend(stage_data["brush_names"])
            
            output_path = os.path.join(output_dir, f"step_{stage_idx + 1}.json")
            self.export(
                accumulated_strokes,
                accumulated_brushes,
                output_path,
                canvas_size=canvas_size,
                stage_info={"stage": stage_idx + 1, "total_stages": len(stages_history)},
            )
        
        # 最终完整版
        final_path = os.path.join(output_dir, "step_final.json")
        self.export(
            accumulated_strokes,
            accumulated_brushes,
            final_path,
            canvas_size=canvas_size,
            stage_info={"stage": "final", "total_stages": len(stages_history)},
        )
    
    @staticmethod
    def load(path: str) -> Dict:
        """加载 JSON 笔画文件"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
