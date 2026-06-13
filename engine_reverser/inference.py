"""
推理脚本 - 输入图像，自回归生成完整的 JSON 日志数组
"""

import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_reverser.constants import CANVAS_WIDTH, CANVAS_HEIGHT
from engine_reverser.reverser import StrokeReverser
from engine_reverser.renderer import DifferentiableCanvasRenderer
from engine_reverser.tokenizer import ActionTokenizer
from utils.image import load_image, save_image


def compute_ssim(img1, img2, ws=7):
    """计算 SSIM"""
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


def run_inference(input_path: str, output_dir: str = "reverser_output"):
    """
    完整推理管线
    
    1. 加载目标图像
    2. 逆向重建 Action 序列
    3. 用可微渲染器回放验证
    4. 生成绘画过程截图
    5. 输出 JSON 日志
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    
    # 加载目标图像 (640x480)
    target = load_image(input_path, (CANVAS_WIDTH, CANVAS_HEIGHT), device)
    save_image(target, os.path.join(output_dir, "target.png"))
    print(f"目标图像: {target.shape}")
    
    # ===== 逆向重建 Action 序列 =====
    print("\n=== 逆向笔画重建 ===")
    reverser = StrokeReverser(device=device)
    
    t0 = time.time()
    actions = reverser.reconstruct(target)
    elapsed = time.time() - t0
    
    print(f"生成 {len(actions)} 个 Action, 耗时 {elapsed:.1f}s")
    
    # 统计 Action 类型
    action_types = {}
    for a in actions:
        at = a[0]
        action_types[at] = action_types.get(at, 0) + 1
    print(f"Action 类型统计: {action_types}")
    
    # ===== 用可微渲染器回放验证 =====
    print("\n=== 回放验证 ===")
    renderer = DifferentiableCanvasRenderer(CANVAS_WIDTH, CANVAS_HEIGHT, device)
    
    # 逐步回放，保存关键帧
    keyframes = []
    keyframe_labels = []
    
    # 初始空白画布
    initial_canvas = torch.ones(3, CANVAS_HEIGHT, CANVAS_WIDTH, device=device)
    keyframes.append(initial_canvas)
    keyframe_labels.append("空白画布")
    
    line_count = 0
    current_stage = "初始化"
    last_blend = None
    blend_change_count = 0
    
    def on_action(idx, action, canvas_state):
        nonlocal line_count, current_stage, last_blend, blend_change_count
        
        if action[0] == "line":
            line_count += 1
        
        # 在 blend 模式切换时保存关键帧
        if action[0] == "blend":
            new_blend = action[1]
            if new_blend != last_blend:
                blend_change_count += 1
                if last_blend is not None:
                    stage_map = {
                        "destination-over": "铺底色",
                        "source-over": "刻画/线稿",
                    }
                    current_stage = stage_map.get(new_blend, new_blend)
                    keyframes.append(canvas_state.clone())
                    keyframe_labels.append(f"{current_stage} ({line_count} 笔)")
                last_blend = new_blend
        
        # 每200条笔画保存一帧
        if action[0] == "line" and line_count % 200 == 0:
            keyframes.append(canvas_state.clone())
            keyframe_labels.append(f"{current_stage} ({line_count} 笔)")
    
    canvas = renderer.render_with_keyframes(actions, keyframe_callback=on_action)
    
    # 最终帧
    keyframes.append(canvas.clone())
    keyframe_labels.append(f"完成 ({line_count} 笔)")
    
    # 计算相似度
    ssim = compute_ssim(canvas, target)
    l1 = (canvas - target).abs().mean().item()
    print(f"\n回放结果: SSIM={ssim*100:.1f}%, PixelSim={(1-l1)*100:.1f}%")
    
    # ===== 保存输出 =====
    
    # 1. JSON 日志
    json_path = os.path.join(output_dir, "painting_log.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(actions, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON 日志已保存: {json_path}")
    
    # 2. 关键帧截图
    print("\n保存绘画过程截图...")
    for i, (kf, label) in enumerate(zip(keyframes, keyframe_labels)):
        safe_label = label.replace("/", "_").replace("(", "").replace(")", "").replace(" ", "_")
        path = os.path.join(output_dir, f"keyframe_{i:02d}_{safe_label}.png")
        save_image(kf, path)
        kf_ssim = compute_ssim(kf, target)
        print(f"  关键帧 {i}: {label}, SSIM={kf_ssim*100:.1f}%")
    
    # 3. 最终结果
    save_image(canvas, os.path.join(output_dir, "final_result.png"))
    
    # 4. 绘画过程图
    _generate_process_chart(target, keyframes, keyframe_labels, output_dir, elapsed, line_count, ssim)
    
    # 5. 对比图
    _generate_comparison(target, canvas, output_dir, line_count, elapsed, ssim)
    
    # 6. HTML 回放器
    _generate_html_player(actions, output_dir)
    
    # 7. Token 序列 (用于 Transformer 训练)
    tokenizer = ActionTokenizer()
    tokens = tokenizer.encode_sequence(actions)
    token_path = os.path.join(output_dir, "tokens.json")
    with open(token_path, "w") as f:
        json.dump(tokens, f)
    print(f"✓ Token 序列已保存: {token_path} ({len(tokens)} tokens)")
    
    return canvas, actions, keyframes, keyframe_labels, elapsed


def _generate_process_chart(target, keyframes, labels, output_dir, elapsed, num_strokes, ssim):
    """生成绘画过程对比图"""
    try:
        matplotlib.font_manager.fontManager.addfont(
            "/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf"
        )
        plt.rcParams["font.sans-serif"] = ["Noto Serif SC", "DejaVu Sans"]
    except:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    
    n = len(keyframes)
    cols = min(n, 7)
    fig, axes = plt.subplots(1, cols, figsize=(3.5 * cols, 4))
    if cols == 1:
        axes = [axes]
    
    # 第一张: 原图
    if n > 0:
        axes[0].imshow(keyframes[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
        axes[0].set_title("空白画布", fontsize=11)
        axes[0].axis("off")
    
    step = max(1, (n - 1) // (cols - 1)) if cols > 1 else 1
    for i in range(1, cols):
        idx = min(i * step, n - 1)
        arr = keyframes[idx].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(arr)
        label = labels[idx] if idx < len(labels) else f"Step {idx}"
        axes[i].set_title(label, fontsize=10)
        axes[i].axis("off")
    
    plt.suptitle(
        f"逆向笔画重建 - 绘画过程\n{num_strokes} 笔画, SSIM={ssim*100:.1f}%, {elapsed:.1f}s",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "painting_process.png"), dpi=200, bbox_inches="tight")
    print("✓ 绘画过程图已保存")


def _generate_comparison(target, canvas, output_dir, num_strokes, elapsed, ssim):
    """生成对比图"""
    try:
        matplotlib.font_manager.fontManager.addfont(
            "/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf"
        )
        plt.rcParams["font.sans-serif"] = ["Noto Serif SC", "DejaVu Sans"]
    except:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(target.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    ax1.set_title("Original", fontsize=16)
    ax1.axis("off")
    ax2.imshow(canvas.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    ax2.set_title(f"Reconstructed (SSIM={ssim*100:.1f}%)", fontsize=16)
    ax2.axis("off")
    plt.suptitle(f"Stroke Reverser ({num_strokes} strokes, {elapsed:.1f}s)", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=200, bbox_inches="tight")
    print("✓ 对比图已保存")


def _generate_html_player(actions: list, output_dir: str):
    """生成 HTML Canvas 回放器（使用引擎格式）"""
    
    # 将 actions 转为 JS 数组字符串
    actions_json = json.dumps(actions, ensure_ascii=False)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PaintsUndo-Vector 引擎回放器</title>
<style>
body {{ margin: 0; background: #1a1a2e; color: #eee; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; padding: 20px; }}
canvas {{ border: 2px solid #444; border-radius: 4px; background: white; }}
.controls {{ margin: 15px 0; display: flex; gap: 10px; align-items: center; }}
button {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }}
.play {{ background: #4CAF50; color: white; }}
.pause {{ background: #f44336; color: white; }}
.reset {{ background: #2196F3; color: white; }}
.info {{ margin: 10px 0; font-size: 14px; color: #aaa; }}
.speed {{ width: 80px; }}
</style>
</head>
<body>
<h2>PaintsUndo-Vector 引擎回放器</h2>
<canvas id="canvas" width="640" height="480"></canvas>
<div class="controls">
  <button class="play" id="playBtn" onclick="togglePlay()">▶ 播放</button>
  <button class="reset" onclick="reset()">⟲ 重置</button>
  <span>速度: <input type="range" class="speed" id="speed" min="1" max="50" value="10"><span id="speedLabel">10x</span></span>
</div>
<div class="info" id="info">准备就绪</div>

<script>
const logs = {actions_json};

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let currentAction = 0;
let playing = false;
let playInterval = null;

// 状态机
let state = {{
    color: "#222222",
    width: 4,
    alpha: 1.0,
    blend: "source-over"
}};

// 背景色
const backgrounds = ["#f8ecdb", "#861010", "#080808", "#956327", "#93aeda"];

// Catmull-Rom 插值 (无压感)
function catmullRomNoPressure(points, tension, subdivisions) {{
    if (points.length < 4) return points;
    let result = [];
    let M = points.slice();
    M.unshift(points[0], points[1]);
    M.push(points[points.length-2], points[points.length-1]);
    for (let p = 2; p < M.length - 4; p += 2) {{
        for (let b = 0; b < subdivisions; b++) {{
            let t = b / subdivisions;
            let c0 = 2*t*t*t - 3*t*t + 1;
            let c1 = -2*t*t*t + 3*t*t;
            let c2 = t*t*t - 2*t*t + t;
            let c3 = t*t*t - t*t;
            let t1 = (M[p+2]-M[p-2])*tension;
            let t2 = (M[p+4]-M[p-0])*tension;
            let t3 = (M[p+3]-M[p-1])*tension;
            let t4 = (M[p+5]-M[p+1])*tension;
            result.push(c0*M[p]+c1*M[p+2]+c2*t1+c3*t2);
            result.push(c0*M[p+1]+c1*M[p+3]+c2*t3+c3*t4);
        }}
    }}
    return result;
}}

// Catmull-Rom 插值 (压感)
function catmullRomPressure(points, tension, subdivisions) {{
    if (points.length < 6) return points;
    let result = [];
    let M = points.slice();
    M.unshift(points[0], points[1], points[2]);
    M.push(points[points.length-3], points[points.length-2], points[points.length-1]);
    for (let p = 3; p < M.length - 6; p += 3) {{
        for (let b = 0; b < subdivisions; b++) {{
            let t = b / subdivisions;
            let c0 = 2*t*t*t - 3*t*t + 1;
            let c1 = -2*t*t*t + 3*t*t;
            let c2 = t*t*t - 2*t*t + t;
            let c3 = t*t*t - t*t;
            let t1 = (M[p+3]-M[p-3])*tension;
            let t2 = (M[p+6]-M[p-0])*tension;
            let t3 = (M[p+4]-M[p-2])*tension;
            let t4 = (M[p+7]-M[p+1])*tension;
            result.push(c0*M[p]+c1*M[p+3]+c2*t1+c3*t2);
            result.push(c0*M[p+1]+c1*M[p+4]+c2*t3+c3*t4);
            result.push(c0*M[p+2]+c1*M[p+5]+c2*(M[p+2+1]-M[p+2-3])*tension+c3*(M[p+2+4]-M[p+2])*tension);
        }}
    }}
    return result;
}}

function executeAction(action) {{
    const type = action[0];
    const payload = action[1];
    
    if (type === "background") {{
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = backgrounds[payload] || backgrounds[0];
        ctx.fillRect(0, 0, 640, 480);
    }} else if (type === "colour" || type === "color") {{
        state.color = payload;
        ctx.strokeStyle = payload;
        ctx.fillStyle = payload;
    }} else if (type === "width") {{
        state.width = payload;
        ctx.lineWidth = payload;
    }} else if (type === "radius") {{
        const rl = [2,4,10,20];
        state.width = rl[Math.min(payload, rl.length-1)];
        ctx.lineWidth = state.width;
    }} else if (type === "alpha") {{
        state.alpha = payload;
        ctx.globalAlpha = payload;
    }} else if (type === "blend") {{
        state.blend = payload;
        ctx.globalCompositeOperation = payload;
    }} else if (type === "line") {{
        const data = payload;
        const brushId = data[0];
        const coords = data.slice(1);
        
        ctx.save();
        ctx.strokeStyle = state.color;
        ctx.fillStyle = state.color;
        ctx.globalAlpha = state.alpha;
        ctx.globalCompositeOperation = state.blend;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        
        const isPressure = [5,6,9,14,16,28,34,36,54,66,69,70].includes(brushId);
        const step = isPressure ? 3 : 2;
        
        if (brushId === 18 || brushId === 19) {{
            // 区域笔刷: 填充多边形
            ctx.beginPath();
            ctx.moveTo(coords[0], coords[1]);
            for (let i = 2; i < coords.length; i += 2) {{
                ctx.lineTo(coords[i], coords[i+1]);
            }}
            ctx.closePath();
            ctx.fillStyle = state.color;
            ctx.strokeStyle = null;
            ctx.fill();
        }} else if (isPressure) {{
            // 压感笔刷
            if (coords.length === 3) {{
                // 单点
                const r = coords[2] * state.width * 0.1;
                ctx.beginPath();
                ctx.arc(coords[0], coords[1], Math.max(1, r), 0, 2*Math.PI);
                ctx.fillStyle = state.color;
                ctx.fill();
            }} else {{
                const interpolated = catmullRomPressure(coords, 0.5, 6);
                const pressureMin = 0.2;
                const pressurePower = 0.2;
                
                for (let i = 3; i < interpolated.length; i += 3) {{
                    const p = interpolated[i+2];
                    const lw = state.width * (pressureMin + p * pressurePower);
                    const alpha = Math.pow(p / 8, 3) * state.alpha;
                    
                    ctx.beginPath();
                    ctx.lineWidth = Math.max(0.5, lw);
                    ctx.globalAlpha = Math.max(0.01, alpha);
                    ctx.moveTo(interpolated[i-3], interpolated[i-2]);
                    ctx.lineTo(interpolated[i], interpolated[i+1]);
                    ctx.stroke();
                }}
            }}
        }} else {{
            // 无压感笔刷 (马克笔等)
            ctx.lineWidth = state.width;
            if (coords.length === 2) {{
                ctx.beginPath();
                ctx.arc(coords[0], coords[1], state.width/2, 0, 2*Math.PI);
                ctx.fill();
            }} else {{
                const interpolated = catmullRomNoPressure(coords, 0.5, 8);
                ctx.beginPath();
                ctx.moveTo(interpolated[0]+0.01, interpolated[1]);
                for (let i = 2; i < interpolated.length - 2; i += 2) {{
                    const mx = (interpolated[i] + interpolated[i+2]) / 2;
                    const my = (interpolated[i+1] + interpolated[i+3]) / 2;
                    ctx.quadraticCurveTo(interpolated[i], interpolated[i+1], mx, my);
                }}
                ctx.stroke();
            }}
        }}
        ctx.restore();
    }}
}}

function step() {{
    if (currentAction < logs.length) {{
        executeAction(logs[currentAction]);
        currentAction++;
        document.getElementById('info').textContent = 
            `Action ${{currentAction}}/${{logs.length}} | 笔画: ${{logs.slice(0,currentAction).filter(a=>a[0]==="line").length}}`;
    }} else {{
        togglePlay();
    }}
}}

function togglePlay() {{
    const btn = document.getElementById('playBtn');
    if (!playing) {{
        playing = true;
        btn.textContent = '⏸ 暂停';
        btn.className = 'pause';
        const speed = parseInt(document.getElementById('speed').value);
        playInterval = setInterval(step, 1000 / speed);
    }} else {{
        playing = false;
        btn.textContent = '▶ 播放';
        btn.className = 'play';
        clearInterval(playInterval);
    }}
}}

function reset() {{
    if (playing) togglePlay();
    currentAction = 0;
    ctx.clearRect(0, 0, 640, 480);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 640, 480);
    document.getElementById('info').textContent = '准备就绪';
}}

document.getElementById('speed').addEventListener('input', e => {{
    document.getElementById('speedLabel').textContent = e.target.value + 'x';
    if (playing) {{ clearInterval(playInterval); playInterval = setInterval(step, 1000 / parseInt(e.target.value)); }}
}});

document.addEventListener('keydown', e => {{
    switch(e.key) {{
        case ' ': e.preventDefault(); togglePlay(); break;
        case 'ArrowRight': step(); break;
        case 'r': reset(); break;
    }}
}});

// 初始化
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, 640, 480);
</script>
</body>
</html>'''
    
    html_path = os.path.join(output_dir, "engine_player.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ HTML 引擎回放器已保存: {html_path}")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/upload/6a042d82867b10d77f923081_mao_low.png"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "reverser_output"
    
    canvas, actions, keyframes, labels, elapsed = run_inference(input_path, output_dir)
