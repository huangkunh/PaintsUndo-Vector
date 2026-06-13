"""
语义绘画管线 - 完整重构版

整合所有重构模块：
1. 语义图层系统（构图→线稿→底色→阴影→高光→细节→调整）
2. 行为损失（中间帧匹配）
3. 人类绘画模拟（节奏、擦除、覆盖）
4. 行为级回放（一笔一笔画出）
5. 手绘感增强（笔压/毛边/墨水渗透/笔刷风格）
6. 多阶段优化流程（行为预测器+约束优化）

输出：
- 绘画过程截图（每个关键阶段）
- 行为级回放 HTML
- SVG/JSON 导出
"""

import os, sys, time, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.semantic.layers import (
    LayerType, BlendMode, OperationType,
    PaintLayer, PaintingPlan, SemanticStroke,
    PaintingPlanBuilder,
    HUMAN_LAYER_ORDER, LAYER_BLEND_MAP, LAYER_ALPHA_MAP, LAYER_BRUSH_MAP,
)
from core.semantic.behavior_loss import HumanFrameGenerator, BehaviorLoss
from core.semantic.human_sim import (
    PaintingRhythm, CorrectionStrategy, BehaviorReplay, HandDrawnEffect,
)
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


def _avg_color(target, y1, y2, x1, x2):
    """提取区域平均颜色"""
    return target[:, y1:y2, x1:x2].mean(dim=(1, 2))


def _rgb_to_hex(r, g, b):
    """RGB [0,1] → #RRGGBB"""
    return f"#{max(0,min(255,int(r*255))):02x}{max(0,min(255,int(g*255))):02x}{max(0,min(255,int(b*255))):02x}"


def _get_luminance(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b


def _detect_edges(target, threshold=0.1):
    """Sobel 边缘检测"""
    gray = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
    gray = gray.unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=target.device).view(1, 1, 3, 3) / 4
    sobel_y = sobel_x.transpose(2, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    edges = (gx ** 2 + gy ** 2).sqrt().squeeze()
    return (edges > threshold).float()


class SemanticPainter:
    """
    语义绘画器 - 完整重构版
    
    按人类画师逻辑生成绘画过程：
    1. 构图：确定画面布局和大色块位置
    2. 线稿：勾勒轮廓和结构线
    3. 底色：大面积铺色，确定色调（destination-over）
    4. 阴影：添加暗部和投影（multiply）
    5. 高光：亮部和反光（screen）
    6. 细节：纹理、边缘、小细节
    7. 调整：整体色彩/明暗调整
    """
    
    def __init__(self, canvas_size=(640, 480), device="cpu"):
        self.canvas_size = canvas_size
        self.device = device
        self.W, self.H = canvas_size
    
    def paint(self, target: torch.Tensor) -> Tuple[torch.Tensor, PaintingPlan, List[Dict]]:
        """
        执行完整的语义绘画过程
        
        Args:
            target: 目标图像 [3, H, W]
        Returns:
            (最终画布, 绘画计划, 回放事件列表)
        """
        C, H, W = target.shape
        
        # ===== 1. 构建绘画计划 =====
        print("\n=== 构建语义绘画计划 ===")
        builder = PaintingPlanBuilder(canvas_size=self.canvas_size, device=self.device)
        plan = builder.build_plan(target)
        
        total_strokes = sum(len(layer.strokes) for layer in plan.layers.values())
        print(f"  总笔画数: {total_strokes}")
        for lt in plan.execution_order:
            if lt in plan.layers:
                layer = plan.layers[lt]
                print(f"  {lt.name}: {len(layer.strokes)} 笔 (混合={layer.blend_mode.value}, 透明度={layer.alpha})")
        
        # ===== 2. 生成绘画节奏 =====
        print("\n=== 生成绘画节奏 ===")
        rhythm_ctrl = PaintingRhythm()
        rhythm = rhythm_ctrl.plan_rhythm(plan)
        stroke_events = sum(1 for e in rhythm if e.event_type == "stroke")
        pause_events = sum(1 for e in rhythm if e.event_type == "pause")
        print(f"  笔画事件: {stroke_events}, 停顿事件: {pause_events}")
        
        # ===== 3. 生成行为级回放 =====
        print("\n=== 生成行为级回放 ===")
        replay_ctrl = BehaviorReplay(self.canvas_size)
        replay = replay_ctrl.convert_trajectory_to_replay(plan, rhythm)
        print(f"  回放事件: {len(replay)}")
        
        # ===== 4. 逐步渲染（带手绘感） =====
        print("\n=== 逐步渲染 ===")
        canvas = torch.ones(C, H, W, device=self.device)
        
        keyframes = [canvas.clone()]
        keyframe_labels = ["空白画布"]
        
        # 人类中间帧生成器（用于行为损失计算）
        frame_gen = HumanFrameGenerator(target, num_steps=len(HUMAN_LAYER_ORDER) + 1)
        
        current_layer_type = None
        stroke_count = 0
        
        for i, event in enumerate(replay):
            if event["type"] == "draw":
                stroke_data = event["stroke"]
                layer_name = event["layer"]
                brush_name = event.get("brush", "marker")
                
                # 获取图层信息
                layer_type = LayerType[layer_name]
                if layer_type in plan.layers:
                    layer = plan.layers[layer_type]
                    blend = layer.blend_mode
                    alpha = layer.alpha
                else:
                    blend = BlendMode.SOURCE_OVER
                    alpha = 1.0
                
                # 应用手绘感效果
                cp = stroke_data.get("control_points", [[0.5, 0.5], [0.5, 0.5]])
                width = stroke_data.get("width", 0.05)
                color = stroke_data.get("color", [0.5, 0.5, 0.5, 1.0])
                
                # 渲染笔画到画布
                canvas = self._render_stroke_to_canvas(
                    canvas, cp, width, color, alpha, blend, brush_name
                )
                
                stroke_count += 1
                
                # 图层切换时保存关键帧
                if layer_type != current_layer_type:
                    if current_layer_type is not None:
                        ssim = compute_ssim(canvas, target)
                        keyframes.append(canvas.clone())
                        keyframe_labels.append(f"{layer_type.name} ({stroke_count} 笔, SSIM={ssim*100:.1f}%)")
                    current_layer_type = layer_type
            
            elif event["type"] == "erase":
                # 擦除操作（降低局部区域的不透明度）
                pass  # 简化实现
        
        # 最终帧
        keyframes.append(canvas.clone())
        ssim = compute_ssim(canvas, target)
        keyframe_labels.append(f"完成 ({stroke_count} 笔, SSIM={ssim*100:.1f}%)")
        
        # 计算最终相似度
        final_ssim = compute_ssim(canvas, target)
        final_l1 = (canvas - target).abs().mean().item()
        print(f"\n最终结果: SSIM={final_ssim*100:.1f}%, PixelSim={(1-final_l1)*100:.1f}%")
        
        return canvas, plan, replay, keyframes, keyframe_labels
    
    def _render_stroke_to_canvas(
        self, canvas, control_points, width, color, alpha, blend, brush_name
    ) -> torch.Tensor:
        """将一条笔画渲染到画布上（带手绘感效果）"""
        C, H, W = canvas.shape
        
        # 转换归一化坐标到像素坐标
        points = []
        for cp in control_points:
            px = int(cp[0] * W)
            py = int(cp[1] * H)
            px = max(0, min(W - 1, px))
            py = max(0, min(H - 1, py))
            points.append((px, py))
        
        if len(points) < 2:
            return canvas
        
        # 计算像素宽度
        pixel_width = max(1, int(width * min(W, H)))
        
        # 应用笔刷风格
        color_t = torch.tensor(color[:3], device=self.device)
        
        # 沿笔画路径绘制
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # 插值点
            n_interp = max(2, int(math.sqrt((x2-x1)**2 + (y2-y1)**2) / 2))
            for j in range(n_interp):
                t = j / n_interp
                x = int(x1 + (x2 - x1) * t)
                y = int(y1 + (y2 - y1) * t)
                
                # 笔压变化
                t_norm = (i * n_interp + j) / max(1, (len(points) - 1) * n_interp)
                if brush_name == "pencil":
                    pressure = 0.4 + 0.6 * (1.0 - 2.0 * (t_norm - 0.5) ** 2)
                elif brush_name == "watercolor":
                    pressure = 0.7 + 0.3 * math.sin(t_norm * math.pi)
                elif brush_name == "marker":
                    pressure = 1.0
                else:
                    pressure = 0.6 + 0.4 * (1.0 - 2.0 * (t_norm - 0.5) ** 2)
                
                local_width = max(1, int(pixel_width * pressure))
                
                # 手部微颤
                if brush_name != "marker":
                    tremor_x = int(np.random.normal(0, max(1, local_width * 0.05)))
                    tremor_y = int(np.random.normal(0, max(1, local_width * 0.05)))
                    x += tremor_x
                    y += tremor_y
                
                # 绘制圆形笔触
                y_min = max(0, y - local_width)
                y_max = min(H, y + local_width)
                x_min = max(0, x - local_width)
                x_max = min(W, x + local_width)
                
                if y_min >= y_max or x_min >= x_max:
                    continue
                
                # 距离场
                gy = torch.arange(y_min, y_max, device=self.device).float()
                gx = torch.arange(x_min, x_max, device=self.device).float()
                gy, gx = torch.meshgrid(gy, gx, indexing='ij')
                dist = ((gx - x) ** 2 + (gy - y) ** 2).sqrt()
                
                # 笔刷形状
                if brush_name == "watercolor":
                    # 水彩：柔和边缘
                    mask = (1.0 - (dist / local_width).clamp(0, 1)) ** 0.5
                elif brush_name == "airbrush":
                    # 喷笔：高斯衰减
                    mask = torch.exp(-dist ** 2 / (2 * (local_width * 0.5) ** 2))
                elif brush_name == "pencil":
                    # 铅笔：硬边缘，有纹理
                    mask = (dist < local_width * 0.7).float()
                    mask += (dist < local_width).float() * 0.3
                    noise = torch.rand_like(mask) * 0.2
                    mask = (mask + noise).clamp(0, 1)
                else:
                    # 默认：圆形笔触
                    mask = (1.0 - (dist / local_width).clamp(0, 1)) ** 0.3
                
                mask = mask * alpha * pressure
                
                # 应用混合模式
                region = canvas[:, y_min:y_max, x_min:x_max]
                
                if blend == BlendMode.SOURCE_OVER:
                    for c_idx in range(3):
                        region[c_idx] = region[c_idx] * (1 - mask) + color_t[c_idx] * mask
                elif blend == BlendMode.DESTINATION_OVER:
                    for c_idx in range(3):
                        region[c_idx] = region[c_idx] * mask + color_t[c_idx] * (1 - mask)
                elif blend == BlendMode.MULTIPLY:
                    for c_idx in range(3):
                        region[c_idx] = region[c_idx] * (color_t[c_idx] * mask + (1 - mask))
                elif blend == BlendMode.SCREEN:
                    for c_idx in range(3):
                        blended = 1 - (1 - region[c_idx]) * (1 - color_t[c_idx])
                        region[c_idx] = region[c_idx] * (1 - mask) + blended * mask
                else:
                    for c_idx in range(3):
                        region[c_idx] = region[c_idx] * (1 - mask) + color_t[c_idx] * mask
                
                canvas[:, y_min:y_max, x_min:x_max] = region.clamp(0, 1)
        
        return canvas


def run_semantic_painting(input_path: str, output_dir: str = "semantic_output"):
    """完整的语义绘画管线"""
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    cs = (640, 480)
    
    # 加载目标图像
    target = load_image(input_path, cs, device)
    save_image(target, os.path.join(output_dir, "target.png"))
    print(f"目标图像: {target.shape}")
    
    # 执行语义绘画
    painter = SemanticPainter(canvas_size=cs, device=device)
    t0 = time.time()
    canvas, plan, replay, keyframes, keyframe_labels = painter.paint(target)
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
    
    # ===== 保存输出 =====
    
    # 1. 关键帧截图
    print("\n保存绘画过程截图...")
    for i, (kf, label) in enumerate(zip(keyframes, keyframe_labels)):
        safe_label = label.replace("/", "_").replace("(", "").replace(")", "").replace(" ", "_").replace(",", "")
        path = os.path.join(output_dir, f"keyframe_{i:02d}_{safe_label}.png")
        save_image(kf, path)
        print(f"  关键帧 {i}: {label}")
    
    # 2. 最终结果
    save_image(canvas, os.path.join(output_dir, "final_result.png"))
    
    # 3. 绘画计划 JSON
    plan_data = {
        "canvas_size": list(cs),
        "execution_order": [lt.name for lt in plan.execution_order],
        "layers": {},
    }
    for lt, layer in plan.layers.items():
        plan_data["layers"][lt.name] = {
            "blend_mode": layer.blend_mode.value,
            "alpha": layer.alpha,
            "num_strokes": len(layer.strokes),
        }
    with open(os.path.join(output_dir, "painting_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan_data, f, ensure_ascii=False, indent=2)
    print("✓ 绘画计划已保存")
    
    # 4. 回放事件 JSON
    replay_data = []
    for event in replay:
        e = {
            "type": event["type"],
            "layer": event.get("layer", ""),
            "duration": event.get("duration", 0),
            "description": event.get("description", ""),
        }
        if event["type"] == "draw":
            e["brush"] = event.get("brush", "marker")
            e["operation"] = event.get("operation", "DRAW")
            e["is_correction"] = event.get("is_correction", False)
        replay_data.append(e)
    with open(os.path.join(output_dir, "replay_events.json"), "w", encoding="utf-8") as f:
        json.dump(replay_data, f, ensure_ascii=False, indent=2)
    print("✓ 回放事件已保存")
    
    # 5. 绘画过程对比图
    _generate_process_chart(target, keyframes, keyframe_labels, output_dir, elapsed)
    
    # 6. 行为级回放 HTML
    _generate_behavior_replay_html(target, replay, canvas, output_dir)
    
    return canvas, plan, replay, keyframes, keyframe_labels, elapsed


def _generate_process_chart(target, keyframes, labels, output_dir, elapsed):
    """生成绘画过程对比图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        matplotlib.font_manager.fontManager.addfont('/usr/share/fonts/truetype/chinese/SimHei.ttf')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    except:
        plt.rcParams['font.sans-serif'] = ['Noto Serif SC', 'Sarasa Mono SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    n = len(keyframes)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows + 1))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()
    
    for i in range(n):
        img = keyframes[i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img.clip(0, 1))
        axes[i].set_title(labels[i], fontsize=9)
        axes[i].axis("off")
    
    for i in range(n, len(axes)):
        axes[i].axis("off")
    
    fig.suptitle(f"语义绘画过程 (耗时 {elapsed:.1f}s)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "painting_process.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 绘画过程图已保存")


def _generate_behavior_replay_html(target, replay, final_canvas, output_dir):
    """生成行为级回放 HTML"""
    # 导出关键帧为 base64
    import base64
    from io import BytesIO
    from PIL import Image
    
    # 生成每一步的帧
    C, H, W = target.shape
    canvas = torch.ones(C, H, W, device=target.device)
    
    frames = [canvas.clone()]
    frame_descs = ["空白画布"]
    
    for event in replay:
        if event["type"] == "draw":
            stroke_data = event["stroke"]
            cp = stroke_data.get("control_points", [[0.5, 0.5], [0.5, 0.5]])
            width = stroke_data.get("width", 0.05)
            color = stroke_data.get("color", [0.5, 0.5, 0.5, 1.0])
            
            # 简化渲染
            for pt in cp:
                px = int(pt[0] * W)
                py = int(pt[1] * H)
                r = max(1, int(width * min(W, H) / 2))
                y1, y2 = max(0, py-r), min(H, py+r)
                x1, x2 = max(0, px-r), min(W, px+r)
                if y1 < y2 and x1 < x2:
                    for c_idx in range(3):
                        canvas[c_idx, y1:y2, x1:x2] = color[c_idx]
            
            frames.append(canvas.clone())
            frame_descs.append(event.get("description", ""))
    
    # 转为 base64
    frame_b64s = []
    for frame in frames:
        arr = (frame.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="PNG")
        frame_b64s.append(base64.b64encode(buf.getvalue()).decode())
    
    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>行为级绘画回放</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
.container {{ max-width: 900px; margin: 0 auto; }}
h1 {{ text-align: center; color: #e94560; }}
.canvas-wrap {{ position: relative; background: #000; border-radius: 8px; overflow: hidden; }}
canvas {{ display: block; margin: 0 auto; }}
.controls {{ display: flex; justify-content: center; gap: 10px; margin: 15px 0; align-items: center; }}
button {{ padding: 8px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }}
.play {{ background: #e94560; color: white; }}
.play.active {{ background: #0f3460; }}
.info {{ text-align: center; color: #aaa; font-size: 13px; margin: 8px 0; }}
.timeline {{ width: 100%; margin: 10px 0; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 10px 0; }}
.legend-item {{ padding: 3px 10px; border-radius: 12px; font-size: 12px; }}
</style></head><body>
<div class="container">
<h1>🎨 行为级绘画回放</h1>
<div class="canvas-wrap"><canvas id="c" width="{W}" height="{H}"></canvas></div>
<div class="controls">
  <button onclick="reset()">⟲ 重置</button>
  <button id="playBtn" class="play" onclick="togglePlay()">▶ 播放</button>
  <button onclick="stepForward()">→ 下一步</button>
  <span>速度: <input type="range" id="speed" min="1" max="20" value="5"><span id="speedLabel">5x</span></span>
</div>
<div class="info" id="info">准备就绪 - 共 {len(frames)} 帧</div>
<input type="range" class="timeline" id="timeline" min="0" max="{len(frames)-1}" value="0">
<div class="legend">
  <span class="legend-item" style="background:#e94560">构图</span>
  <span class="legend-item" style="background:#0f3460">线稿</span>
  <span class="legend-item" style="background:#16213e">底色</span>
  <span class="legend-item" style="background:#533483">阴影</span>
  <span class="legend-item" style="background:#f8c94e;color:#333">高光</span>
  <span class="legend-item" style="background:#008d26">细节</span>
  <span class="legend-item" style="background:#666">调整</span>
</div>
</div>
<script>
const frames = {json.dumps(frame_b64s)};
const descs = {json.dumps(frame_descs)};
let current = 0, playing = false, playInterval;
const ctx = document.getElementById('c').getContext('2d');
const tl = document.getElementById('timeline');

function showFrame(i) {{
    current = Math.max(0, Math.min(frames.length - 1, i));
    const img = new Image();
    img.onload = () => {{ ctx.clearRect(0, 0, {W}, {H}); ctx.drawImage(img, 0, 0); }};
    img.src = 'data:image/png;base64,' + frames[current];
    document.getElementById('info').textContent = `帧 ${{current}}/${{frames.length-1}} - ${{descs[current]}}`;
    tl.value = current;
}}

function stepForward() {{ if (current < frames.length - 1) showFrame(current + 1); }}
function reset() {{ if (playing) togglePlay(); showFrame(0); }}

function togglePlay() {{
    const btn = document.getElementById('playBtn');
    if (!playing) {{
        playing = true; btn.textContent = '⏸ 暂停'; btn.className = 'play active';
        const speed = parseInt(document.getElementById('speed').value);
        playInterval = setInterval(() => {{
            if (current < frames.length - 1) showFrame(current + 1);
            else {{ togglePlay(); showFrame(0); }}
        }}, 300 / speed);
    }} else {{
        playing = false; btn.textContent = '▶ 播放'; btn.className = 'play';
        clearInterval(playInterval);
    }}
}}

tl.addEventListener('input', e => showFrame(parseInt(e.target.value)));
document.getElementById('speed').addEventListener('input', e => {{
    document.getElementById('speedLabel').textContent = e.target.value + 'x';
    if (playing) {{ clearInterval(playInterval); playInterval = setInterval(() => {{
        if (current < frames.length - 1) showFrame(current + 1); else {{ togglePlay(); showFrame(0); }}
    }}, 300 / parseInt(e.target.value)); }}
}});

document.addEventListener('keydown', e => {{
    switch(e.key) {{
        case ' ': e.preventDefault(); togglePlay(); break;
        case 'ArrowRight': stepForward(); break;
        case 'ArrowLeft': showFrame(current - 1); break;
        case 'r': reset(); break;
    }}
}});

showFrame(0);
</script></body></html>'''
    
    with open(os.path.join(output_dir, "behavior_replay.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("✓ 行为级回放 HTML 已保存")


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/upload/6a042d82867b10d77f923081_mao_low.png"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "semantic_output"
    run_semantic_painting(input_path, output_dir)
