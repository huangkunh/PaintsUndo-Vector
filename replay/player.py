"""
笔画回放器 - 高质量绘画过程回放

将优化后的笔画参数动态回放，模拟绘画过程。
支持：
- 逐步绘制（笔画生长动画）
- Undo 模拟（从最终状态退回到前一阶段）
- HTML/JS 回放（生成可在浏览器中播放的 HTML 文件）
- 绘画叙述（画家的内心独白）
- 速度曲线（模拟人类绘画节奏）

对应方案B阶段六：制作动态回放
"""

import os
import json
from typing import List, Optional, Tuple, Dict

import numpy as np

from brushes.base import BrushStroke
from core.painting_sim import sort_strokes_human_like, simulate_painting_rhythm, generate_painting_narrative


class StrokePlayer:
    """
    笔画回放器
    
    支持多种回放方式：
    - HTML/JS 回放：生成可在浏览器中播放的 HTML 文件
    - 逐帧 PNG：生成每一帧的 PNG 图片
    - GIF 动画：生成 GIF 动画文件
    """
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (640, 480),
        background_color: str = "#FFFFFF",
    ):
        self.canvas_width, self.canvas_height = canvas_size
        self.background_color = background_color
    
    def export_html_player(
        self,
        stages_history: List[Dict],
        output_path: str,
    ):
        """
        导出 HTML 回放器。
        
        生成一个包含 Canvas 的 HTML 文件，
        可以在浏览器中播放绘画过程，支持：
        - 播放/暂停
        - 逐步前进/后退
        - Undo（退回上一阶段）
        - 速度调节
        - 笔画生长动画
        - 绘画叙述文字
        """
        # 收集所有笔画数据
        all_strokes = []
        all_brushes = []
        stage_boundaries = [0]
        
        for stage_data in stages_history:
            # 对每个阶段的笔画进行人类绘画排序
            sorted_strokes, sorted_brushes = sort_strokes_human_like(
                stage_data["strokes"],
                stage_data["brush_names"],
            )
            
            for stroke, brush_name in zip(sorted_strokes, sorted_brushes):
                stroke_dict = {
                    "brush": brush_name,
                    "points": stroke.control_points.detach().cpu().numpy().tolist(),
                    "width": stroke.width.detach().cpu().item(),
                    "color": stroke.color.detach().cpu().numpy().tolist(),
                    "opacity": stroke.opacity.detach().cpu().item(),
                }
                all_strokes.append(stroke_dict)
                all_brushes.append(brush_name)
            
            stage_boundaries.append(len(all_strokes))
        
        # 生成绘画节奏
        timeline = simulate_painting_rhythm(
            stages_history[-1]["strokes"] if stages_history else [],
            stages_history[-1]["brush_names"] if stages_history else [],
        )
        
        # 生成绘画叙述
        narratives = generate_painting_narrative(stages_history)
        
        # 生成 HTML
        html = self._generate_html(all_strokes, stage_boundaries, narratives)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"HTML player exported to: {output_path}")
    
    def _generate_html(
        self,
        strokes: List[Dict],
        stage_boundaries: List[int],
        narratives: List[str],
    ) -> str:
        """生成 HTML 回放器代码"""
        strokes_json = json.dumps(strokes)
        boundaries_json = json.dumps(stage_boundaries)
        narratives_json = json.dumps(narratives)
        
        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PaintsUndo-Vector 绘画回放</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }}
        h1 {{
            font-size: 24px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .canvas-container {{
            position: relative;
            border: 2px solid #333;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        canvas {{
            display: block;
            background: {self.background_color};
        }}
        .narrative {{
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            color: #fff;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            transition: opacity 0.3s;
            pointer-events: none;
            white-space: nowrap;
        }}
        .controls {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        button {{
            background: #333;
            color: #eee;
            border: 1px solid #555;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        button:hover {{ background: #444; border-color: #667eea; }}
        button:active {{ transform: scale(0.95); }}
        button.active {{ background: #667eea; border-color: #667eea; }}
        .progress-container {{
            width: 100%;
            max-width: {self.canvas_width}px;
            height: 4px;
            background: #333;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .progress {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.1s;
            width: 0%;
        }}
        .info {{
            margin-top: 8px;
            font-size: 13px;
            color: #888;
        }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .speed-control input[type="range"] {{
            width: 80px;
            accent-color: #667eea;
        }}
        .stage-indicators {{
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }}
        .stage-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #333;
            transition: background 0.3s;
        }}
        .stage-dot.active {{ background: #667eea; }}
        .stage-dot.completed {{ background: #764ba2; }}
    </style>
</head>
<body>
    <h1>PaintsUndo-Vector 绘画回放</h1>
    <div class="canvas-container">
        <canvas id="canvas" width="{self.canvas_width}" height="{self.canvas_height}"></canvas>
        <div class="narrative" id="narrative"></div>
    </div>
    <div class="progress-container">
        <div class="progress" id="progress"></div>
    </div>
    <div class="info" id="info">准备就绪</div>
    <div class="stage-indicators" id="stageIndicators"></div>
    <div class="controls">
        <button onclick="reset()">⏮ 重置</button>
        <button onclick="undoStage()">⏪ 上一阶段</button>
        <button onclick="prevStroke()">◀ 上一笔</button>
        <button onclick="togglePlay()" id="playBtn">▶ 播放</button>
        <button onclick="nextStroke()">下一笔 ▶</button>
        <button onclick="redoStage()">下一阶段 ⏩</button>
        <button onclick="skipToEnd()">⏭ 跳到结尾</button>
        <div class="speed-control">
            <span>速度:</span>
            <input type="range" id="speed" min="1" max="20" value="5">
            <span id="speedLabel">5x</span>
        </div>
    </div>
    
    <script>
        const strokes = {strokes_json};
        const stageBoundaries = {boundaries_json};
        const narratives = {narratives_json};
        
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let currentStroke = 0;
        let isPlaying = false;
        let playTimer = null;
        let growProgress = 1.0; // 笔画生长进度 0-1
        
        // 初始化阶段指示器
        const stageInd = document.getElementById('stageIndicators');
        for (let i = 0; i < stageBoundaries.length - 1; i++) {{
            const dot = document.createElement('div');
            dot.className = 'stage-dot';
            dot.id = 'stageDot' + i;
            stageInd.appendChild(dot);
        }}
        
        function clearCanvas() {{
            ctx.fillStyle = '{self.background_color}';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }}
        
        function drawStroke(stroke, progress = 1.0) {{
            const points = stroke.points;
            if (points.length < 2) return;
            
            // 转换归一化坐标到像素坐标
            const pixelPoints = points.map(p => [p[0] * canvas.width, p[1] * canvas.height]);
            
            // 根据进度截取笔画
            const numPoints = Math.max(2, Math.ceil(pixelPoints.length * progress));
            const visiblePoints = pixelPoints.slice(0, numPoints);
            
            ctx.save();
            ctx.globalAlpha = stroke.opacity;
            ctx.strokeStyle = `rgb(${{Math.round(stroke.color[0]*255)}},${{Math.round(stroke.color[1]*255)}},${{Math.round(stroke.color[2]*255)}})`;
            ctx.lineWidth = stroke.width;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            
            // 根据笔刷类型选择绘制方式
            switch (stroke.brush) {{
                case 'watercolor':
                    // 水彩：添加模糊效果
                    ctx.filter = 'blur(2px)';
                    ctx.globalAlpha = stroke.opacity * 0.7;
                    break;
                case 'airbrush':
                    // 喷笔：更大的模糊
                    ctx.filter = 'blur(3px)';
                    ctx.globalAlpha = stroke.opacity * 0.5;
                    break;
                case 'pencil':
                    // 铅笔：细线条
                    ctx.lineWidth = Math.max(1, stroke.width * 0.8);
                    break;
                case 'pressure':
                case 'pressure_sharp':
                    // 压感笔：变宽效果
                    drawVariableWidthStroke(visiblePoints, stroke, progress);
                    ctx.restore();
                    return;
                case 'hatching':
                    // 排线：平行线
                    drawHatchingStroke(visiblePoints, stroke, progress);
                    ctx.restore();
                    return;
            }}
            
            // 默认绘制：二次贝塞尔曲线
            ctx.beginPath();
            ctx.moveTo(visiblePoints[0][0], visiblePoints[0][1]);
            
            if (visiblePoints.length === 2) {{
                ctx.lineTo(visiblePoints[1][0], visiblePoints[1][1]);
            }} else {{
                for (let i = 1; i < visiblePoints.length - 1; i += 2) {{
                    if (i + 1 < visiblePoints.length) {{
                        ctx.quadraticCurveTo(
                            visiblePoints[i][0], visiblePoints[i][1],
                            visiblePoints[i+1][0], visiblePoints[i+1][1]
                        );
                    }} else {{
                        ctx.lineTo(visiblePoints[i][0], visiblePoints[i][1]);
                    }}
                }}
            }}
            
            ctx.stroke();
            ctx.filter = 'none';
            ctx.restore();
        }}
        
        function drawVariableWidthStroke(points, stroke, progress) {{
            if (points.length < 2) return;
            
            const n = points.length;
            const baseWidth = stroke.width;
            
            // 计算每个点的宽度和法线
            const leftSide = [];
            const rightSide = [];
            
            for (let i = 0; i < n; i++) {{
                const t = i / (n - 1);
                let pressure;
                if (stroke.brush === 'pressure_sharp') {{
                    pressure = 1 - Math.exp(-3 * t);
                }} else {{
                    pressure = Math.sin(Math.PI * t);
                }}
                const w = baseWidth * pressure / 2;
                
                // 计算切线
                let tx, ty;
                if (i === 0) {{ tx = points[1][0] - points[0][0]; ty = points[1][1] - points[0][1]; }}
                else if (i === n - 1) {{ tx = points[n-1][0] - points[n-2][0]; ty = points[n-1][1] - points[n-2][1]; }}
                else {{ tx = points[i+1][0] - points[i-1][0]; ty = points[i+1][1] - points[i-1][1]; }}
                
                const len = Math.sqrt(tx*tx + ty*ty) + 0.001;
                const nx = -ty / len;
                const ny = tx / len;
                
                leftSide.push([points[i][0] + nx * w, points[i][1] + ny * w]);
                rightSide.push([points[i][0] - nx * w, points[i][1] - ny * w]);
            }}
            
            // 绘制多边形
            ctx.save();
            ctx.globalAlpha = stroke.opacity;
            ctx.fillStyle = `rgb(${{Math.round(stroke.color[0]*255)}},${{Math.round(stroke.color[1]*255)}},${{Math.round(stroke.color[2]*255)}})`;
            
            ctx.beginPath();
            ctx.moveTo(leftSide[0][0], leftSide[0][1]);
            for (let i = 1; i < leftSide.length; i++) {{
                ctx.lineTo(leftSide[i][0], leftSide[i][1]);
            }}
            for (let i = rightSide.length - 1; i >= 0; i--) {{
                ctx.lineTo(rightSide[i][0], rightSide[i][1]);
            }}
            ctx.closePath();
            ctx.fill();
            ctx.restore();
        }}
        
        function drawHatchingStroke(points, stroke, progress) {{
            // 简化的排线效果
            ctx.save();
            ctx.globalAlpha = stroke.opacity * 0.6;
            ctx.strokeStyle = `rgb(${{Math.round(stroke.color[0]*255)}},${{Math.round(stroke.color[1]*255)}},${{Math.round(stroke.color[2]*255)}})`;
            ctx.lineWidth = Math.max(1, stroke.width * 0.3);
            
            const spacing = stroke.width * 1.5;
            for (let offset = -spacing * 3; offset <= spacing * 3; offset += spacing) {{
                ctx.beginPath();
                for (let i = 0; i < points.length; i++) {{
                    const x = points[i][0] + offset;
                    const y = points[i][1];
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();
            }}
            ctx.restore();
        }}
        
        function redrawUpTo(n) {{
            clearCanvas();
            for (let i = 0; i < Math.min(n, strokes.length); i++) {{
                drawStroke(strokes[i], 1.0);
            }}
            currentStroke = Math.min(n, strokes.length);
            updateInfo();
        }}
        
        function togglePlay() {{
            if (isPlaying) {{
                stopPlay();
            }} else {{
                startPlay();
            }}
        }}
        
        function startPlay() {{
            if (currentStroke >= strokes.length) {{
                currentStroke = 0;
            }}
            isPlaying = true;
            document.getElementById('playBtn').textContent = '⏸ 暂停';
            document.getElementById('playBtn').classList.add('active');
            playNext();
        }}
        
        function stopPlay() {{
            isPlaying = false;
            document.getElementById('playBtn').textContent = '▶ 播放';
            document.getElementById('playBtn').classList.remove('active');
            if (playTimer) clearTimeout(playTimer);
        }}
        
        function playNext() {{
            if (!isPlaying || currentStroke >= strokes.length) {{
                stopPlay();
                return;
            }}
            
            // 绘制当前笔画（带生长动画）
            animateStrokeGrowth(currentStroke, () => {{
                currentStroke++;
                updateInfo();
                
                const speed = parseInt(document.getElementById('speed').value);
                const delay = Math.max(10, 200 / speed);
                playTimer = setTimeout(playNext, delay);
            }});
        }}
        
        function animateStrokeGrowth(index, callback) {{
            const stroke = strokes[index];
            const speed = parseInt(document.getElementById('speed').value);
            const duration = Math.max(50, 300 / speed);
            const startTime = performance.now();
            
            function step(time) {{
                const elapsed = time - startTime;
                const progress = Math.min(1, elapsed / duration);
                
                // 重绘到当前笔画（带进度）
                redrawUpTo(index);
                drawStroke(stroke, progress);
                
                if (progress < 1) {{
                    requestAnimationFrame(step);
                }} else {{
                    callback();
                }}
            }}
            
            requestAnimationFrame(step);
        }}
        
        function nextStroke() {{
            if (currentStroke < strokes.length) {{
                drawStroke(strokes[currentStroke], 1.0);
                currentStroke++;
                updateInfo();
            }}
        }}
        
        function prevStroke() {{
            if (currentStroke > 0) {{
                currentStroke--;
                redrawUpTo(currentStroke);
            }}
        }}
        
        function undoStage() {{
            for (let i = stageBoundaries.length - 1; i >= 0; i--) {{
                if (currentStroke > stageBoundaries[i]) {{
                    currentStroke = stageBoundaries[i];
                    redrawUpTo(currentStroke);
                    break;
                }}
            }}
        }}
        
        function redoStage() {{
            for (let i = 0; i < stageBoundaries.length; i++) {{
                if (currentStroke < stageBoundaries[i]) {{
                    redrawUpTo(stageBoundaries[i]);
                    break;
                }}
            }}
        }}
        
        function reset() {{
            stopPlay();
            currentStroke = 0;
            clearCanvas();
            updateInfo();
        }}
        
        function skipToEnd() {{
            stopPlay();
            redrawUpTo(strokes.length);
        }}
        
        function updateInfo() {{
            const stage = stageBoundaries.filter(b => b <= currentStroke).length - 1;
            const stageNames = ['铺底色', '形体刻画', '细节线稿'];
            const stageName = stageNames[stage] || '完成';
            
            document.getElementById('info').textContent = 
                `笔画: ${{currentStroke}} / ${{strokes.length}} | 阶段: ${{stageName}} (${{stage + 1}} / ${{stageBoundaries.length - 1}})`;
            
            const progress = strokes.length > 0 ? (currentStroke / strokes.length * 100) : 0;
            document.getElementById('progress').style.width = progress + '%';
            
            // 更新阶段指示器
            for (let i = 0; i < stageBoundaries.length - 1; i++) {{
                const dot = document.getElementById('stageDot' + i);
                dot.className = 'stage-dot';
                if (i < stage) dot.classList.add('completed');
                else if (i === stage) dot.classList.add('active');
            }}
            
            // 更新叙述文字
            const narrativeEl = document.getElementById('narrative');
            if (currentStroke > 0 && currentStroke <= narratives.length && narratives[currentStroke - 1]) {{
                narrativeEl.textContent = narratives[currentStroke - 1];
                narrativeEl.style.opacity = '1';
            }} else {{
                narrativeEl.style.opacity = '0';
            }}
        }}
        
        document.getElementById('speed').addEventListener('input', (e) => {{
            document.getElementById('speedLabel').textContent = e.target.value + 'x';
        }});
        
        // 键盘快捷键
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case ' ': e.preventDefault(); togglePlay(); break;
                case 'ArrowRight': nextStroke(); break;
                case 'ArrowLeft': prevStroke(); break;
                case 'Home': reset(); break;
                case 'End': skipToEnd(); break;
            }}
        }});
        
        // 初始化
        clearCanvas();
        updateInfo();
    </script>
</body>
</html>'''
