"""
人类画师模拟绘画 - 参考 PaintsUndo 绘画过程

PaintsUndo 核心原理：
1. 单帧模型：输入最终图像 + 操作步骤(0-999)，输出该步骤时的"模拟截图"
2. 多帧模型：输入两张关键帧，输出16帧中间过渡帧
3. 默认流程：先用单帧模型生成5-7个关键帧(粗略→精细)，再用多帧模型插值
4. 关键：每个关键帧是完整的画面状态，从粗略线稿到上色完成

本项目实现（不依赖深度学习模型）：
- 用多尺度色块模拟 PaintsUndo 的关键帧生成
- 每个阶段生成覆盖整个画布的色块，形成完整画面
- 阶段1: 极粗色块(32px) → 模糊的整体色调
- 阶段2: 中等色块(16px) → 可辨认的形体
- 阶段3: 细色块(8px) → 清晰的细节
- 阶段4: 精细色块(4px) → 接近完成
- 阶段5: 像素精修(2px) → 高相似度

每个阶段都是完整的画面，不是局部绘制。
色块按人类绘画顺序排列：暗色→亮色，背景→前景，大→小。
"""

import os, time, json, math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

from brushes.base import BrushStroke
from core.renderer import DifferentiableRenderer
from export.svg_export import SVGExporter
from export.json_export import JSONExporter
from utils.image import load_image, save_image


def compute_ssim(img1, img2, ws=7):
    """计算 SSIM"""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(img1, ws, stride=1, padding=ws // 2)
    mu2 = F.avg_pool2d(img2, ws, stride=1, padding=ws // 2)
    s1 = F.avg_pool2d(img1 ** 2, ws, stride=1, padding=ws // 2) - mu1 ** 2
    s2 = F.avg_pool2d(img2 ** 2, ws, stride=1, padding=ws // 2) - mu2 ** 2
    s12 = F.avg_pool2d(img1 * img2, ws, stride=1, padding=ws // 2) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return ssim.mean().item()


def _avg_pool_color(target: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    将目标图像下采样到 patch_size 网格，再上采样回原尺寸。
    这模拟了"用大色块铺色"的效果——每个色块取该区域的平均颜色。
    
    Args:
        target: [3, H, W]
        patch_size: 色块大小（像素）
    Returns:
        [3, H, W] 色块化后的图像
    """
    C, H, W = target.shape
    # 下采样
    h_blocks = (H + patch_size - 1) // patch_size
    w_blocks = (W + patch_size - 1) // patch_size
    # 用 avg_pool2d 实现精确下采样
    pooled = F.adaptive_avg_pool2d(target.unsqueeze(0), (h_blocks, w_blocks)).squeeze(0)
    # 上采样回原尺寸（最近邻，产生色块效果）
    blockified = F.interpolate(pooled.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
    return blockified


def _sort_patches_human_like(
    target: torch.Tensor, patch_size: int, canvas: torch.Tensor
) -> List[Tuple[int, int, torch.Tensor, float]]:
    """
    按人类绘画顺序排列色块。
    
    排序规则（参考 PaintsUndo 的绘画行为）：
    1. 暗色优先（先画暗部，再画亮部——画家通常先铺暗色底）
    2. 残差大的优先（差异大的区域更需要绘制）
    3. 从上到下，从左到右（模拟画家的视线移动）
    
    Returns:
        List of (row, col, color, priority) 按优先级排序
    """
    C, H, W = target.shape
    h_blocks = H // patch_size
    w_blocks = W // patch_size
    
    patches = []
    for r in range(h_blocks):
        for c in range(w_blocks):
            y1, y2 = r * patch_size, (r + 1) * patch_size
            x1, x2 = c * patch_size, (c + 1) * patch_size
            
            target_patch = target[:, y1:y2, x1:x2]
            canvas_patch = canvas[:, y1:y2, x1:x2]
            
            # 平均颜色
            avg_color = target_patch.mean(dim=(1, 2))
            
            # 残差（该色块需要绘制的程度）
            residual = (target_patch - canvas_patch).abs().mean().item()
            
            # 亮度（暗色优先）
            brightness = avg_color.mean().item()
            
            # 优先级：残差大 + 暗色 + 位置靠上靠左
            # 残差权重最高，亮度次之，位置最低
            priority = (
                residual * 10.0          # 残差越大越优先
                + (1.0 - brightness) * 3.0  # 暗色优先
                + (1.0 - r / max(h_blocks, 1)) * 1.0  # 上方优先
                + (1.0 - c / max(w_blocks, 1)) * 0.5   # 左方优先
            )
            
            patches.append((r, c, avg_color, priority))
    
    # 按优先级降序排列
    patches.sort(key=lambda x: -x[3])
    return patches


def _paint_stage(
    canvas: torch.Tensor,
    target: torch.Tensor,
    patch_size: int,
    diff_threshold: float = 0.005,
    opacity: float = 1.0,
) -> torch.Tensor:
    """
    执行一个绘画阶段：用指定大小的色块覆盖整个画布。
    
    参考 PaintsUndo 的关键帧概念：
    - 每个阶段结束后，画面是一个完整的"关键帧"
    - 大色块阶段 = 粗略线稿/铺色
    - 小色块阶段 = 精细刻画
    
    Args:
        canvas: 当前画布 [3, H, W]
        target: 目标图像 [3, H, W]
        patch_size: 色块大小
        diff_threshold: 只绘制残差超过此阈值的色块
        opacity: 色块透明度（模拟颜料的覆盖力）
    Returns:
        更新后的画布 [3, H, W]
    """
    C, H, W = target.shape
    
    # 获取排序后的色块列表
    patches = _sort_patches_human_like(target, patch_size, canvas)
    
    for r, c, avg_color, priority in patches:
        y1, y2 = r * patch_size, (r + 1) * patch_size
        x1, x2 = c * patch_size, (c + 1) * patch_size
        
        # 计算该色块与当前画布的残差
        canvas_patch = canvas[:, y1:y2, x1:x2]
        residual = (target[:, y1:y2, x1:x2] - canvas_patch).abs().mean().item()
        
        if residual < diff_threshold:
            continue
        
        # 用目标颜色覆盖该色块
        # 模拟人类画家的笔触：不是完全覆盖，而是有一定透明度
        target_color = target[:, y1:y2, x1:x2].mean(dim=(1, 2), keepdim=True)
        new_patch = canvas_patch * (1 - opacity) + target_color.expand_as(canvas_patch) * opacity
        canvas[:, y1:y2, x1:x2] = new_patch
    
    return canvas


def _add_stroke_texture(canvas: torch.Tensor, target: torch.Tensor, patch_size: int, strength: float = 0.3):
    """
    添加笔画纹理效果，让色块看起来更像手绘。
    
    在色块边界添加轻微的颜色变化，模拟：
    - 笔触重叠的痕迹
    - 颜料厚薄不均
    - 手部颤抖导致的颜色偏移
    """
    C, H, W = canvas.shape
    h_blocks = H // patch_size
    w_blocks = W // patch_size
    
    for r in range(h_blocks):
        for c in range(w_blocks):
            y1, y2 = r * patch_size, (r + 1) * patch_size
            x1, x2 = c * patch_size, (c + 1) * patch_size
            
            patch = canvas[:, y1:y2, x1:x2]
            
            # 添加轻微的随机颜色偏移（模拟手部颤抖）
            noise = torch.randn_like(patch) * strength * 0.02
            canvas[:, y1:y2, x1:x2] = (patch + noise).clamp(0, 1)
    
    return canvas


def human_painting_process(
    target: torch.Tensor,
    canvas_size: Tuple[int, int],
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[Dict], List[torch.Tensor]]:
    """
    人类画师绘画过程（参考 PaintsUndo）
    
    模拟 PaintsUndo 的关键帧生成：
    - 每个阶段生成一个完整的"关键帧"
    - 从粗略到精细，逐步构建画面
    - 每个关键帧都是完整的画面状态
    
    Args:
        target: 目标图像 [3, H, W]
        canvas_size: 画布尺寸 (W, H)
        device: 设备
    Returns:
        (最终画布, 笔画数据列表, 关键帧列表)
    """
    C, H, W = target.shape
    
    # 定义绘画阶段（参考 PaintsUndo 的关键帧）
    # 每个阶段对应 PaintsUndo 的一个 "operation step" 范围
    stages = [
        {
            "name": "粗略铺色",
            "patch_size": 32,
            "opacity": 0.95,
            "diff_threshold": 0.005,
            "narrative": "用大号画笔快速铺底色，确定整体色调...",
            "operation_step": "0-200",  # PaintsUndo 操作步骤
        },
        {
            "name": "形体刻画",
            "patch_size": 16,
            "opacity": 0.95,
            "diff_threshold": 0.005,
            "narrative": "用中等画笔刻画形体轮廓，区分明暗...",
            "operation_step": "200-400",
        },
        {
            "name": "细节描绘",
            "patch_size": 8,
            "opacity": 0.95,
            "diff_threshold": 0.003,
            "narrative": "用细画笔描绘细节，添加纹理...",
            "operation_step": "400-600",
        },
        {
            "name": "精细调整",
            "patch_size": 4,
            "opacity": 0.95,
            "diff_threshold": 0.002,
            "narrative": "精细调整颜色和细节...",
            "operation_step": "600-800",
        },
        {
            "name": "像素精修",
            "patch_size": 2,
            "opacity": 1.0,
            "diff_threshold": 0.001,
            "narrative": "最后精修，达到高相似度...",
            "operation_step": "800-950",
        },
        {
            "name": "最终完善",
            "patch_size": 1,
            "opacity": 1.0,
            "diff_threshold": 0.005,
            "narrative": "最终完善细节...",
            "operation_step": "950-999",
        },
    ]
    
    # 初始化画布（白色）
    canvas = torch.ones_like(target)
    
    # 收集关键帧（参考 PaintsUndo 的关键帧概念）
    keyframes = [canvas.clone()]  # 初始空白画布
    keyframe_labels = ["空白画布"]
    
    strokes_data = []
    total_strokes = 0
    
    for stage_idx, stage in enumerate(stages):
        ps = stage["patch_size"]
        print(f"\n=== Stage {stage_idx + 1}: {stage['name']} (patch={ps}px) ===")
        print(f"  {stage['narrative']}")
        
        # 执行绘画阶段
        canvas = _paint_stage(
            canvas, target,
            patch_size=ps,
            diff_threshold=stage["diff_threshold"],
            opacity=stage["opacity"],
        )
        
        # 添加笔画纹理（仅大色块阶段）
        if ps >= 8:
            canvas = _add_stroke_texture(canvas, target, ps, strength=0.3)
        
        # 计算当前 SSIM
        ssim = compute_ssim(canvas, target)
        l1 = (canvas - target).abs().mean().item()
        
        # 统计该阶段绘制的色块数
        h_blocks = H // ps
        w_blocks = W // ps
        n_patches = h_blocks * w_blocks
        total_strokes += n_patches
        
        print(f"  {n_patches} patches, SSIM={ssim*100:.1f}%, PixelSim={(1-l1)*100:.1f}%")
        
        # 保存关键帧
        keyframes.append(canvas.clone())
        keyframe_labels.append(stage['name'])
        
        # 记录笔画数据
        for r in range(h_blocks):
            for c in range(w_blocks):
                y1, y2 = r * ps, (r + 1) * ps
                x1, x2 = c * ps, (c + 1) * ps
                color = canvas[:, y1:y2, x1:x2].mean(dim=(1, 2))
                strokes_data.append({
                    "stage": stage_idx,
                    "stage_name": stage["name"],
                    "patch_size": ps,
                    "row": r, "col": c,
                    "x": x1, "y": y1,
                    "width": ps, "height": ps,
                    "color": color.tolist(),
                    "operation_step": stage["operation_step"],
                })
    
    # 最终 SSIM
    final_ssim = compute_ssim(canvas, target)
    final_l1 = (canvas - target).abs().mean().item()
    print(f"\n最终结果: {total_strokes} patches, SSIM={final_ssim*100:.1f}%, PixelSim={(1-final_l1)*100:.1f}%")
    
    return canvas, strokes_data, keyframes, keyframe_labels


def run_human_painting(input_path: str, output_dir: str = "cat_output",
                       target_ssim: float = 0.98):
    """完整的人类画师绘画管线"""
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    cs = (256, 256)
    
    target = load_image(input_path, cs, device)
    save_image(target, os.path.join(output_dir, "target.png"))
    print(f"目标图像: {target.shape}")
    
    t0 = time.time()
    canvas, strokes_data, keyframes, keyframe_labels = human_painting_process(
        target, cs, device
    )
    elapsed = time.time() - t0
    
    # 保存所有关键帧截图
    print("\n保存绘画过程截图...")
    for i, (kf, label) in enumerate(zip(keyframes, keyframe_labels)):
        path = os.path.join(output_dir, f"process_{i:02d}_{label}.png")
        save_image(kf, path)
        ssim = compute_ssim(kf, target)
        print(f"  关键帧 {i}: {label}, SSIM={ssim*100:.1f}%")
    
    # 保存最终结果
    save_image(canvas, os.path.join(output_dir, "final_result.png"))
    
    # 导出 SVG（用色块数据生成矩形笔画）
    svg_strokes = []
    svg_brushes = []
    for sd in strokes_data:
        s = BrushStroke(num_control_points=2, canvas_size=cs, device=device)
        x_n = sd["x"] / cs[0]
        y_n = sd["y"] / cs[1]
        w_n = sd["width"] / cs[0]
        h_n = sd["height"] / cs[1]
        cp = torch.tensor([[x_n, y_n], [x_n + w_n, y_n + h_n]], device=device)
        s.raw_control_points.data = torch.logit(torch.clamp(cp, 0.01, 0.99))
        s.raw_width.data = torch.logit(torch.clamp(torch.tensor(w_n * 0.8), 0.01, 0.99))
        color_t = torch.tensor(sd["color"] + [1.0], device=device)
        s.raw_color.data = torch.logit(torch.clamp(color_t, 0.01, 0.99))
        s.raw_opacity.data = torch.logit(torch.clamp(torch.tensor(0.95), 0.01, 0.99))
        svg_strokes.append(s)
        svg_brushes.append(sd.get("brush", "marker"))
    SVGExporter(cs).export(svg_strokes, svg_brushes, os.path.join(output_dir, "result.svg"))
    print("✓ SVG 已导出")
    
    # 导出 JSON
    json_data = {
        "canvas_size": list(cs),
        "num_strokes": len(strokes_data),
        "stages": keyframe_labels,
        "strokes": strokes_data,
    }
    with open(os.path.join(output_dir, "strokes.json"), "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print("✓ JSON 已导出")
    
    # 生成绘画过程图
    _generate_process_chart(target, keyframes, keyframe_labels, output_dir, elapsed, len(strokes_data))
    
    # 生成对比图
    _generate_comparison(target, canvas, output_dir, len(strokes_data), elapsed)
    
    # 生成 HTML 回放器
    _generate_html_player(keyframes, keyframe_labels, cs, output_dir)
    
    return canvas, strokes_data, keyframes, keyframe_labels, elapsed


def _generate_process_chart(target, keyframes, labels, output_dir, elapsed, num_strokes):
    """生成绘画过程对比图（参考 PaintsUndo 的关键帧展示）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        matplotlib.font_manager.fontManager.addfont(
            "/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf"
        )
        plt.rcParams["font.sans-serif"] = ["Noto Serif SC", "DejaVu Sans"]
    except:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    
    n = len(keyframes)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(rows * cols):
        r, c_idx = i // cols, i % cols
        ax = axes[r][c_idx]
        if i < n:
            arr = keyframes[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(arr)
            ssim = compute_ssim(keyframes[i], target)
            ax.set_title(f"{labels[i]}\nSSIM={ssim*100:.1f}%", fontsize=10)
        ax.axis("off")
    
    plt.suptitle(
        f"PaintsUndo-Vector 人类画师绘画过程\n"
        f"参考 PaintsUndo 关键帧生成 | {num_strokes} patches | {elapsed:.1f}s",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "painting_process.png"), dpi=200, bbox_inches="tight")
    print("✓ 绘画过程图已保存")


def _generate_comparison(target, canvas, output_dir, num_strokes, elapsed):
    """生成原图 vs 生成图对比"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        matplotlib.font_manager.fontManager.addfont(
            "/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf"
        )
        plt.rcParams["font.sans-serif"] = ["Noto Serif SC", "DejaVu Sans"]
    except:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    
    ssim = compute_ssim(canvas, target)
    l1 = (canvas - target).abs().mean().item()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(target.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    ax1.set_title("Original", fontsize=16)
    ax1.axis("off")
    
    ax2.imshow(canvas.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    ax2.set_title(f"Generated (SSIM={ssim*100:.1f}%)", fontsize=16)
    ax2.axis("off")
    
    # 差异图
    diff = (canvas - target).abs().mean(dim=0).detach().cpu().numpy()
    ax3.imshow(diff, cmap='hot', vmin=0, vmax=0.1)
    ax3.set_title(f"Difference (L1={l1:.4f})", fontsize=16)
    ax3.axis("off")
    
    plt.suptitle(f"PaintsUndo-Vector ({num_strokes} patches, {elapsed:.1f}s)", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=200, bbox_inches="tight")
    print("✓ 对比图已保存")


def _generate_html_player(keyframes, labels, canvas_size, output_dir):
    """生成 HTML 绘画回放器（参考 PaintsUndo 的视频回放）"""
    W, H = canvas_size
    
    # 将关键帧转为 base64
    import io, base64
    from PIL import Image
    
    frames_b64 = []
    for kf in keyframes:
        arr = (kf.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        frames_b64.append(base64.b64encode(buf.getvalue()).decode())
    
    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PaintsUndo-Vector 绘画回放</title>
<style>
body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }}
h1 {{ margin: 20px 0 10px; font-size: 24px; }}
.canvas-container {{ position: relative; margin: 10px; }}
canvas {{ border: 2px solid #333; border-radius: 4px; image-rendering: pixelated; }}
.controls {{ display: flex; gap: 10px; margin: 10px; align-items: center; }}
button {{ background: #16213e; color: #eee; border: 1px solid #0f3460; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; }}
button:hover {{ background: #0f3460; }}
button.active {{ background: #e94560; }}
.info {{ margin: 10px; text-align: center; }}
.stage-name {{ font-size: 18px; color: #e94560; font-weight: bold; }}
.progress-bar {{ width: 512px; height: 8px; background: #333; border-radius: 4px; margin: 10px; }}
.progress-fill {{ height: 100%; background: #e94560; border-radius: 4px; transition: width 0.3s; }}
.timeline {{ display: flex; gap: 4px; margin: 10px; }}
.timeline-dot {{ width: 12px; height: 12px; border-radius: 50%; background: #333; cursor: pointer; transition: background 0.2s; }}
.timeline-dot.active {{ background: #e94560; }}
.timeline-dot.completed {{ background: #0f3460; }}
</style>
</head>
<body>
<h1>PaintsUndo-Vector 绘画过程回放</h1>
<div class="canvas-container">
<canvas id="canvas" width="{W}" height="{H}"></canvas>
</div>
<div class="stage-name" id="stageName">空白画布</div>
<div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
<div class="timeline" id="timeline"></div>
<div class="controls">
<button id="prevBtn">◀ 上一步</button>
<button id="playBtn">▶ 自动播放</button>
<button id="nextBtn">下一步 ▶</button>
<button id="resetBtn">⟲ 重置</button>
<span style="margin-left: 20px;">速度:</span>
<input type="range" id="speed" min="0.5" max="5" step="0.5" value="1">
<span id="speedLabel">1x</span>
</div>
<div class="info" id="info">0 / {len(keyframes) - 1}</div>

<script>
const frames = {json.dumps(frames_b64)};
const labels = {json.dumps(labels)};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let currentFrame = 0;
let playing = false;
let playInterval = null;

// 创建时间线
const timeline = document.getElementById('timeline');
for (let i = 0; i < frames.length; i++) {{
    const dot = document.createElement('div');
    dot.className = 'timeline-dot' + (i === 0 ? ' active' : '');
    dot.onclick = () => showFrame(i);
    timeline.appendChild(dot);
}}

function showFrame(idx) {{
    currentFrame = idx;
    const img = new Image();
    img.onload = () => {{
        ctx.clearRect(0, 0, {W}, {H});
        ctx.drawImage(img, 0, 0, {W}, {H});
    }};
    img.src = 'data:image/png;base64,' + frames[idx];
    document.getElementById('stageName').textContent = labels[idx];
    document.getElementById('progress').style.width = (idx / (frames.length - 1) * 100) + '%';
    document.getElementById('info').textContent = idx + ' / ' + (frames.length - 1);
    
    // 更新时间线
    const dots = timeline.children;
    for (let i = 0; i < dots.length; i++) {{
        dots[i].className = 'timeline-dot';
        if (i < idx) dots[i].classList.add('completed');
        else if (i === idx) dots[i].classList.add('active');
    }}
}}

document.getElementById('prevBtn').onclick = () => showFrame(Math.max(0, currentFrame - 1));
document.getElementById('nextBtn').onclick = () => showFrame(Math.min(frames.length - 1, currentFrame + 1));
document.getElementById('resetBtn').onclick = () => {{ showFrame(0); if (playing) togglePlay(); }};

function togglePlay() {{
    playing = !playing;
    const btn = document.getElementById('playBtn');
    if (playing) {{
        btn.textContent = '⏸ 暂停';
        btn.classList.add('active');
        const speed = parseFloat(document.getElementById('speed').value);
        playInterval = setInterval(() => {{
            if (currentFrame < frames.length - 1) showFrame(currentFrame + 1);
            else {{ togglePlay(); showFrame(0); }}
        }}, 1000 / speed);
    }} else {{
        btn.textContent = '▶ 自动播放';
        btn.classList.remove('active');
        clearInterval(playInterval);
    }}
}}
document.getElementById('playBtn').onclick = togglePlay;
document.getElementById('speed').addEventListener('input', e => {{
    document.getElementById('speedLabel').textContent = e.target.value + 'x';
    if (playing) {{ clearInterval(playInterval); togglePlay(); togglePlay(); }}
}});

document.addEventListener('keydown', e => {{
    switch(e.key) {{
        case ' ': e.preventDefault(); togglePlay(); break;
        case 'ArrowRight': showFrame(Math.min(frames.length - 1, currentFrame + 1)); break;
        case 'ArrowLeft': showFrame(Math.max(0, currentFrame - 1)); break;
    }}
}});

showFrame(0);
</script>
</body>
</html>'''
    
    with open(os.path.join(output_dir, "painting_player.html"), "w") as f:
        f.write(html)
    print(f"✓ HTML 回放器已保存: {os.path.join(output_dir, 'painting_player.html')}")


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/upload/6a042d82867b10d77f923081_mao_low.png"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "cat_output"
    run_human_painting(input_path, output_dir)
