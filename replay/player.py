"""
笔画回放器

将优化后的笔画参数动态回放，模拟绘画过程。
支持：
- 逐步绘制（像打字机一样逐步生长）
- Undo 模拟（从最终状态退回到前一阶段）
- HTML/JS 回放（生成可在浏览器中播放的 HTML 文件）

对应方案B阶段六：制作动态回放
"""

import os
import json
from typing import List, Optional, Tuple, Dict

import numpy as np

from brushes.base import BrushStroke


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
        """
        # 收集所有笔画数据
        all_strokes = []
        stage_boundaries = [0]
        
        for stage_data in stages_history:
            for stroke, brush_name in zip(stage_data["strokes"], stage_data["brush_names"]):
                stroke_dict = {
                    "brush": brush_name,
                    "points": stroke.control_points.detach().cpu().numpy().tolist(),
                    "width": stroke.width.detach().cpu().item(),
                    "color": stroke.color.detach().cpu().numpy().tolist(),
                    "opacity": stroke.opacity.detach().cpu().item(),
                }
                all_strokes.append(stroke_dict)
            stage_boundaries.append(len(all_strokes))
        
        html_content = self._generate_html(all_strokes, stage_boundaries)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"HTML player exported to: {output_path}")
    
    def _generate_html(
        self,
        strokes: List[Dict],
        stage_boundaries: List[int],
    ) -> str:
        """生成 HTML 回放器代码"""
        strokes_json = json.dumps(strokes, ensure_ascii=False)
        boundaries_json = json.dumps(stage_boundaries)
        
        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PaintsUndo-Vector 回放器</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Microsoft YaHei', sans-serif;
            background: #1a1a2e;
            color: #eee;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }}
        h1 {{ margin-bottom: 20px; font-size: 24px; color: #e94560; }}
        .canvas-container {{
            position: relative;
            border: 2px solid #16213e;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(233, 69, 96, 0.3);
        }}
        canvas {{
            display: block;
            background: {self.background_color};
        }}
        .controls {{
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            background: #16213e;
            color: #eee;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        button:hover {{ background: #0f3460; }}
        button.active {{ background: #e94560; }}
        .info {{
            margin-top: 15px;
            font-size: 14px;
            color: #aaa;
        }}
        .progress-bar {{
            width: 100%;
            max-width: {self.canvas_width}px;
            height: 6px;
            background: #16213e;
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: #e94560;
            transition: width 0.1s;
        }}
        .speed-control {{
            margin-top: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        input[type="range"] {{
            width: 150px;
        }}
    </style>
</head>
<body>
    <h1>PaintsUndo-Vector 回放器</h1>
    <div class="canvas-container">
        <canvas id="canvas" width="{self.canvas_width}" height="{self.canvas_height}"></canvas>
    </div>
    <div class="progress-bar">
        <div class="progress-fill" id="progress"></div>
    </div>
    <div class="controls">
        <button id="btnPlay" onclick="togglePlay()">▶ 播放</button>
        <button onclick="stepForward()">→ 下一步</button>
        <button onclick="stepBackward()">← 上一步</button>
        <button onclick="undoStage()">↩ Undo 阶段</button>
        <button onclick="redoStage()">↪ Redo 阶段</button>
        <button onclick="reset()">⟲ 重置</button>
    </div>
    <div class="speed-control">
        <span>速度:</span>
        <input type="range" id="speed" min="1" max="20" value="5">
        <span id="speedLabel">5x</span>
    </div>
    <div class="info" id="info">笔画: 0 / {len(strokes)} | 阶段: 0 / {len(stage_boundaries) - 1}</div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const strokes = {strokes_json};
        const stageBoundaries = {boundaries_json};
        
        let currentStroke = 0;
        let isPlaying = false;
        let playInterval = null;
        let currentStage = 0;
        
        function clearCanvas() {{
            ctx.fillStyle = '{self.background_color}';
            ctx.fillRect(0, 0, {self.canvas_width}, {self.canvas_height});
        }}
        
        function drawStroke(index) {{
            const stroke = strokes[index];
            if (!stroke) return;
            
            const points = stroke.points;
            const w = {self.canvas_width};
            const h = {self.canvas_height};
            
            // 转换归一化坐标为像素坐标
            const pixelPoints = points.map(p => [p[0] * w, p[1] * h]);
            
            ctx.save();
            ctx.globalAlpha = stroke.opacity;
            ctx.strokeStyle = `rgb(${{Math.round(stroke.color[0]*255)}},${{Math.round(stroke.color[1]*255)}},${{Math.round(stroke.color[2]*255)}})`;
            ctx.lineWidth = stroke.width;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            
            ctx.beginPath();
            if (pixelPoints.length >= 2) {{
                ctx.moveTo(pixelPoints[0][0], pixelPoints[0][1]);
                for (let i = 1; i < pixelPoints.length; i++) {{
                    ctx.lineTo(pixelPoints[i][0], pixelPoints[i][1]);
                }}
            }}
            ctx.stroke();
            ctx.restore();
        }}
        
        function redrawUpTo(n) {{
            clearCanvas();
            for (let i = 0; i < n; i++) {{
                drawStroke(i);
            }}
            currentStroke = n;
            updateInfo();
        }}
        
        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('btnPlay');
            if (isPlaying) {{
                btn.textContent = '⏸ 暂停';
                btn.classList.add('active');
                play();
            }} else {{
                btn.textContent = '▶ 播放';
                btn.classList.remove('active');
                if (playInterval) clearInterval(playInterval);
            }}
        }}
        
        function play() {{
            if (playInterval) clearInterval(playInterval);
            const speed = parseInt(document.getElementById('speed').value);
            playInterval = setInterval(() => {{
                if (currentStroke >= strokes.length) {{
                    togglePlay();
                    return;
                }}
                drawStroke(currentStroke);
                currentStroke++;
                updateInfo();
            }}, 200 / speed);
        }}
        
        function stepForward() {{
            if (currentStroke < strokes.length) {{
                drawStroke(currentStroke);
                currentStroke++;
                updateInfo();
            }}
        }}
        
        function stepBackward() {{
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
            currentStroke = 0;
            clearCanvas();
            updateInfo();
        }}
        
        function updateInfo() {{
            const stage = stageBoundaries.filter(b => b <= currentStroke).length - 1;
            document.getElementById('info').textContent = 
                `笔画: ${{currentStroke}} / ${{strokes.length}} | 阶段: ${{stage}} / ${{stageBoundaries.length - 1}}`;
            const progress = strokes.length > 0 ? (currentStroke / strokes.length * 100) : 0;
            document.getElementById('progress').style.width = progress + '%';
        }}
        
        document.getElementById('speed').addEventListener('input', (e) => {{
            document.getElementById('speedLabel').textContent = e.target.value + 'x';
            if (isPlaying) play();
        }});
        
        // 初始化
        clearCanvas();
        updateInfo();
    </script>
</body>
</html>'''
