"""
验证脚本 - 将生成的 JSON 日志输入引擎渲染器，验证是否能正确还原图像
"""

import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_reverser.constants import CANVAS_WIDTH, CANVAS_HEIGHT, BACKGROUND_COLORS, DEFAULT_COLORS
from engine_reverser.renderer import DifferentiableCanvasRenderer
from utils.image import load_image, save_image


def validate_log(log_path: str, target_path: str, output_dir: str = "validation_output"):
    """
    验证 JSON 日志是否能正确还原图像
    
    Args:
        log_path: JSON 日志文件路径
        target_path: 原始目标图像路径
        output_dir: 验证结果输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    
    # 加载日志
    with open(log_path, 'r', encoding='utf-8') as f:
        actions = json.load(f)
    print(f"加载日志: {len(actions)} 个 Action")
    
    # 加载目标图像
    target = load_image(target_path, (CANVAS_WIDTH, CANVAS_HEIGHT), device)
    save_image(target, os.path.join(output_dir, "target.png"))
    
    # 用渲染器回放
    renderer = DifferentiableCanvasRenderer(CANVAS_WIDTH, CANVAS_HEIGHT, device)
    
    # 逐步回放，保存关键帧
    keyframes = []
    line_count = 0
    
    for i, action in enumerate(actions):
        renderer.execute_action(action)
        
        if action[0] == "line":
            line_count += 1
    
    canvas = renderer.canvas
    
    # 计算相似度
    ssim = _compute_ssim(canvas, target)
    l1 = (canvas - target).abs().mean().item()
    
    print(f"\n验证结果:")
    print(f"  笔画数: {line_count}")
    print(f"  SSIM: {ssim*100:.1f}%")
    print(f"  PixelSim: {(1-l1)*100:.1f}%")
    
    # 保存结果
    save_image(canvas, os.path.join(output_dir, "reconstructed.png"))
    print(f"✓ 重建图像已保存")
    
    # 验证 JSON 格式
    _validate_json_format(actions)
    
    return ssim, l1


def _compute_ssim(img1, img2, ws=7):
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = torch.nn.functional.avg_pool2d(img1, ws, stride=1, padding=ws // 2)
    mu2 = torch.nn.functional.avg_pool2d(img2, ws, stride=1, padding=ws // 2)
    s1 = torch.nn.functional.avg_pool2d(img1 ** 2, ws, stride=1, padding=ws // 2) - mu1 ** 2
    s2 = torch.nn.functional.avg_pool2d(img2 ** 2, ws, stride=1, padding=ws // 2) - mu2 ** 2
    s12 = torch.nn.functional.avg_pool2d(img1 * img2, ws, stride=1, padding=ws // 2) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return ssim.mean().item()


def _validate_json_format(actions: list):
    """验证 JSON 日志格式是否符合引擎规范"""
    print("\n格式验证:")
    
    valid_action_types = {"background", "colour", "color", "width", "radius", "alpha", "blend", "line"}
    
    errors = []
    warnings = []
    
    for i, action in enumerate(actions):
        if not isinstance(action, list) or len(action) < 2:
            errors.append(f"  Action {i}: 不是有效的 [type, payload] 格式")
            continue
        
        action_type = action[0]
        
        if action_type not in valid_action_types:
            errors.append(f"  Action {i}: 未知动作类型 '{action_type}'")
            continue
        
        payload = action[1]
        
        if action_type == "background":
            if not isinstance(payload, int) or payload < 0 or payload >= len(BACKGROUND_COLORS):
                warnings.append(f"  Action {i}: background 索引 {payload} 可能越界")
        
        elif action_type in ("colour", "color"):
            if isinstance(payload, str):
                if not payload.startswith("#") or len(payload) != 7:
                    warnings.append(f"  Action {i}: 颜色格式 '{payload}' 不是 #RRGGBB")
            elif isinstance(payload, int):
                if payload < 0 or payload >= len(DEFAULT_COLORS):
                    warnings.append(f"  Action {i}: 颜色索引 {payload} 可能越界")
        
        elif action_type == "width":
            if not isinstance(payload, (int, float)) or payload <= 0:
                errors.append(f"  Action {i}: width 值 {payload} 无效")
        
        elif action_type == "alpha":
            if not isinstance(payload, (int, float)) or payload < 0 or payload > 1:
                errors.append(f"  Action {i}: alpha 值 {payload} 不在 [0,1] 范围")
        
        elif action_type == "blend":
            valid_blends = {"source-over", "destination-over", "destination-out", "multiply", "screen", "darken"}
            if payload not in valid_blends:
                warnings.append(f"  Action {i}: blend 模式 '{payload}' 可能不被支持")
        
        elif action_type == "line":
            if not isinstance(payload, list) or len(payload) < 3:
                errors.append(f"  Action {i}: line 数据格式无效")
            else:
                brush_id = payload[0]
                if not isinstance(brush_id, int) or brush_id < 0 or brush_id > 72:
                    warnings.append(f"  Action {i}: 笔刷 ID {brush_id} 可能无效")
                
                coords = payload[1:]
                # 检查步长
                pressure_brushes = {5, 6, 9, 14, 16, 28, 34, 36, 54, 66, 69, 70}
                if brush_id in pressure_brushes:
                    if len(coords) % 3 != 0:
                        errors.append(f"  Action {i}: 压感笔刷 {brush_id} 坐标步长应为3，实际 {len(coords)} 个值")
                else:
                    if len(coords) % 2 != 0:
                        errors.append(f"  Action {i}: 无压感笔刷 {brush_id} 坐标步长应为2，实际 {len(coords)} 个值")
    
    if errors:
        print(f"  ❌ {len(errors)} 个错误:")
        for e in errors[:10]:
            print(e)
    else:
        print("  ✓ 无格式错误")
    
    if warnings:
        print(f"  ⚠ {len(warnings)} 个警告:")
        for w in warnings[:10]:
            print(w)
    else:
        print("  ✓ 无格式警告")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "reverser_output/painting_log.json"
    target_path = sys.argv[2] if len(sys.argv) > 2 else "/home/z/my-project/upload/6a042d82867b10d77f923081_mao_low.png"
    
    validate_log(log_path, target_path)
