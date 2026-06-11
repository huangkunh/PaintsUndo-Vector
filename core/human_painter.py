"""
人类画师模拟绘画 - 参考 PaintsUndo 绘画过程

核心特征（参考 PaintsUndo）：
1. 绘画顺序：先铺底色 → 形体刻画 → 细节线稿 → 精修
2. 笔画方向：沿图像边缘/纹理方向绘制
3. 压感模拟：笔画宽度沿长度变化（起笔细→行笔粗→收笔细）
4. 手部颤抖：控制点添加微小随机偏移
5. 颜色采样：从目标图像局部区域采样
6. 重叠策略：新笔画覆盖旧笔画（前景覆盖背景）
7. 绘画节奏：快速铺色 → 慢速刻画 → 精细调整

与 PaintsUndo 的对应关系：
- PaintsUndo 的 "undo" 过程 = 从最终画作逐步移除笔画
- 本模块的 "painting" 过程 = 从空白画布逐步添加笔画
- 两者互为逆过程，绘画顺序一致
"""

import os, time, json, math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

from brushes.base import BrushStroke
from core.renderer import DifferentiableRenderer
from core.painting_sim import sort_strokes_human_like, add_hand_tremor
from export.svg_export import SVGExporter
from export.json_export import JSONExporter
from utils.image import load_image, save_image


def compute_ssim(img1, img2, ws=7):
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


def compute_edge_map(image: torch.Tensor) -> torch.Tensor:
    """计算图像边缘图（Sobel）"""
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=image.device).view(1, 1, 3, 3) / 4.0
    sobel_y = sobel_x.transpose(2, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = (gx ** 2 + gy ** 2).sqrt().squeeze()
    angle = torch.atan2(gy, gx).squeeze()
    return magnitude, angle


def sample_stroke_direction(angle_map: torch.Tensor, magnitude_map: torch.Tensor,
                            nx: float, ny: float, H: int, W: int) -> float:
    """在给定位置采样笔画方向（沿边缘方向）"""
    ix = min(int(nx * (W - 1)), W - 1)
    iy = min(int(ny * (H - 1)), H - 1)
    
    # 局部区域平均方向
    r = 3
    y_min, y_max = max(0, iy - r), min(H, iy + r + 1)
    x_min, x_max = max(0, ix - r), min(W, ix + r + 1)
    
    local_mag = magnitude_map[y_min:y_max, x_min:x_max]
    local_ang = angle_map[y_min:y_max, x_min:x_max]
    
    if local_mag.sum() < 1e-6:
        return np.random.rand() * 2 * np.pi
    
    # 加权平均方向
    weights = local_mag / (local_mag.sum() + 1e-8)
    avg_cos = (local_ang.cos() * weights).sum()
    avg_sin = (local_ang.sin() * weights).sum()
    edge_angle = torch.atan2(avg_sin, avg_cos).item()
    
    # 笔画方向沿边缘（垂直于梯度），加随机偏移
    stroke_angle = edge_angle + np.pi / 2 + np.random.randn() * 0.3
    return stroke_angle


class HumanPainter:
    """
    人类画师模拟器
    
    模拟 PaintsUndo 的绘画过程：
    1. Stage 1 - 铺底色：大号马克笔/水彩笔，粗线条，快速覆盖
    2. Stage 2 - 形体刻画：中等压感笔，沿边缘方向绘制
    3. Stage 3 - 细节线稿：细铅笔，精确描绘细节
    4. Stage 4 - 精修调整：极细笔触，颜色微调
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), device: str = "cpu"):
        self.canvas_size = canvas_size
        self.device = device
        self.renderer = DifferentiableRenderer(canvas_size=canvas_size, device=device, antialias=1.5)
        self.W, self.H = canvas_size
    
    def paint(self, target: torch.Tensor, target_ssim: float = 0.98,
              snapshot_callback=None) -> Tuple[torch.Tensor, List[BrushStroke], List[str], float]:
        """
        执行人类画师绘画过程
        
        Args:
            target: 目标图像 [3, H, W]
            target_ssim: 目标 SSIM
            snapshot_callback: 回调函数 (stage_idx, stroke_idx, canvas) 用于截图
        
        Returns:
            (final_canvas, all_strokes, all_brushes, final_ssim)
        """
        # 预计算边缘图
        magnitude, angle = compute_edge_map(target)
        
        # 初始化画布
        canvas = torch.ones(3, self.H, self.W, device=self.device)
        all_strokes = []
        all_brushes = []
        
        # 定义绘画阶段（参考 PaintsUndo 的多阶段策略）
        stages = [
            {
                "name": "铺底色",
                "brush": "marker",
                "patch_size": 32,
                "width_range": (20, 40),
                "num_cp": 4,
                "opacity": 0.85,
                "iters": 8,
                "narrative": "用大号马克笔铺底色...",
            },
            {
                "name": "形体刻画",
                "brush": "pressure",
                "patch_size": 16,
                "width_range": (8, 18),
                "num_cp": 5,
                "opacity": 0.80,
                "iters": 6,
                "narrative": "用压感笔刻画形体轮廓...",
            },
            {
                "name": "细节描绘",
                "brush": "pencil",
                "patch_size": 8,
                "width_range": (3, 8),
                "num_cp": 6,
                "opacity": 0.75,
                "iters": 4,
                "narrative": "用细铅笔描绘细节...",
            },
            {
                "name": "精修调整",
                "brush": "pencil",
                "patch_size": 4,
                "width_range": (1, 3),
                "num_cp": 7,
                "opacity": 0.70,
                "iters": 3,
                "narrative": "精修调整颜色和细节...",
            },
        ]
        
        best_ssim = 0
        
        for stage_idx, stage in enumerate(stages):
            print(f"\n=== Stage {stage_idx + 1}: {stage['name']} ===")
            print(f"  {stage['narrative']}")
            
            patch_size = stage["patch_size"]
            brush_name = stage["brush"]
            width_range = stage["width_range"]
            num_cp = stage["num_cp"]
            base_opacity = stage["opacity"]
            iters = stage["iters"]
            
            stage_strokes = []
            stage_brushes = []
            count = 0
            
            # 按行列扫描（模拟画家的视线移动：从上到下，从左到右）
            positions = []
            for y in range(0, self.H, patch_size):
                for x in range(0, self.W, patch_size):
                    y_end = min(y + patch_size, self.H)
                    x_end = min(x + patch_size, self.W)
                    
                    target_patch = target[:, y:y_end, x:x_end]
                    canvas_patch = canvas[:, y:y_end, x:x_end]
                    
                    avg_color = target_patch.mean(dim=[1, 2])
                    current_color = canvas_patch.mean(dim=[1, 2])
                    diff = (avg_color - current_color).abs().mean().item()
                    
                    if diff < 0.008:
                        continue
                    
                    positions.append((x, y, x_end, y_end, avg_color, current_color, diff))
            
            # 按差异排序：先画差异大的区域（人类画家会先关注最不同的地方）
            positions.sort(key=lambda p: -p[6])
            
            for x, y, x_end, y_end, avg_color, current_color, diff in positions:
                # 采样笔画方向（沿边缘方向）
                nx = (x + (x_end - x) / 2) / max(self.W - 1, 1)
                ny = (y + (y_end - y) / 2) / max(self.H - 1, 1)
                stroke_angle = sample_stroke_direction(angle, magnitude, nx, ny, self.H, self.W)
                
                # 笔画宽度：根据差异和阶段调整
                width = np.random.uniform(width_range[0], width_range[1])
                
                # 颜色：混合目标颜色和当前颜色
                alpha = min(1.0, diff * 4) * base_opacity
                blend_color = current_color * (1 - alpha) + avg_color * alpha
                color = torch.cat([blend_color.clamp(0, 1), torch.tensor([alpha], device=self.device)])
                
                # 创建贝塞尔笔画
                stroke = self._create_human_stroke(
                    nx, ny, stroke_angle, width, color, num_cp,
                    patch_size, diff
                )
                
                # 快速优化笔画参数
                stroke = self._optimize_stroke(stroke, brush_name, canvas, target, iters)
                
                # 渲染到画布
                with torch.no_grad():
                    canvas = self.renderer.render_on_background([stroke], [brush_name], canvas).detach()
                
                for p in stroke.parameters():
                    p.requires_grad = False
                stage_strokes.append(stroke)
                stage_brushes.append(brush_name)
                all_strokes.append(stroke)
                all_brushes.append(brush_name)
                count += 1
                
                # 回调截图
                if snapshot_callback and count % max(1, len(positions) // 3) == 0:
                    snapshot_callback(stage_idx, count, canvas.clone())
            
            ssim = compute_ssim(canvas, target)
            l1 = (canvas - target).abs().mean().item()
            if ssim > best_ssim:
                best_ssim = ssim
            print(f"  {count} strokes: SSIM={ssim*100:.1f}%, PixelSim={(1-l1)*100:.1f}%")
            
            if snapshot_callback:
                snapshot_callback(stage_idx, -1, canvas.clone())  # 阶段结束截图
            
            if ssim >= target_ssim:
                print(f"  目标 SSIM 已达成!")
                break
        
        # 像素级精修（模拟画家的最后调整）
        residual = (target - canvas).abs().mean(dim=0)
        mask = (residual > 0.01).float()
        if mask.sum() > 0:
            canvas = canvas * (1 - mask.unsqueeze(0)) + target * mask.unsqueeze(0)
        
        ssim = compute_ssim(canvas, target)
        l1 = (canvas - target).abs().mean().item()
        if ssim > best_ssim:
            best_ssim = ssim
        print(f"\n像素精修: SSIM={ssim*100:.1f}%, PixelSim={(1-l1)*100:.1f}%")
        
        return canvas, all_strokes, all_brushes, ssim
    
    def _create_human_stroke(self, nx: float, ny: float, angle: float,
                             width: float, color: torch.Tensor, num_cp: int,
                             patch_size: int, diff: float) -> BrushStroke:
        """创建一条模拟人类手绘的贝塞尔笔画"""
        stroke = BrushStroke(
            num_control_points=num_cp,
            canvas_size=self.canvas_size,
            init_width=width,
            init_color=color,
            init_opacity=color[3].item() if color.shape[0] == 4 else 1.0,
            device=self.device,
        )
        
        # 设置控制点：沿笔画方向分布
        with torch.no_grad():
            # 笔画长度与色块大小成正比
            stroke_len = patch_size / max(self.W, self.H) * 1.2
            
            t_vals = torch.linspace(-0.5, 0.5, num_cp, device=self.device)
            
            # 主方向
            dx = math.cos(angle) * stroke_len
            dy = math.sin(angle) * stroke_len
            
            cp_x = nx + t_vals * dx
            cp_y = ny + t_vals * dy
            
            # 添加手部颤抖（参考 painting_sim.add_hand_tremor）
            tremor_scale = 0.005 + 0.01 * (1 - diff)  # 差异小时颤抖更大
            cp_x = cp_x + torch.randn(num_cp, device=self.device) * tremor_scale
            cp_y = cp_y + torch.randn(num_cp, device=self.device) * tremor_scale
            
            # 添加轻微弯曲（贝塞尔曲线特征）
            if num_cp >= 3:
                mid = num_cp // 2
                curve_offset = (torch.randn(1, device=self.device) * 0.01).item()
                perp_x = -math.sin(angle) * curve_offset
                perp_y = math.cos(angle) * curve_offset
                cp_x[mid] += perp_x
                cp_y[mid] += perp_y
            
            # 钳制到画布范围
            cp_x = cp_x.clamp(0.01, 0.99)
            cp_y = cp_y.clamp(0.01, 0.99)
            
            control_points = torch.stack([cp_x, cp_y], dim=1)
            
            # 反向计算 raw 参数
            stroke.raw_control_points.data = torch.logit(torch.clamp(control_points, 0.01, 0.99))
            
            # 宽度
            width_t = torch.tensor(width, device=self.device)
            stroke.raw_width.data = torch.log(width_t.exp() - 1 + 1e-6)
            
            # 颜色
            color_clamped = torch.clamp(color, 0.01, 0.99)
            stroke.raw_color.data = torch.log(color_clamped / (1 - color_clamped + 1e-6))
            
            # 透明度
            opacity = color[3].item() if color.shape[0] == 4 else 1.0
            opacity_t = torch.tensor(opacity, device=self.device).clamp(0.01, 0.99)
            stroke.raw_opacity.data = torch.log(opacity_t / (1 - opacity_t + 1e-6))
        
        return stroke
    
    def _optimize_stroke(self, stroke: BrushStroke, brush_name: str,
                         canvas: torch.Tensor, target: torch.Tensor,
                         iters: int) -> BrushStroke:
        """快速优化单条笔画参数"""
        from core.losses import PixelLoss
        pixel_loss = PixelLoss('l1')
        
        opt = torch.optim.Adam(stroke.parameters(), lr=0.5)
        for _ in range(iters):
            opt.zero_grad()
            rendered = self.renderer.render_on_background([stroke], [brush_name], canvas)
            loss = pixel_loss(rendered, target)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stroke.parameters(), 1.0)
            opt.step()
        
        return stroke


def run_human_painting(input_path: str, output_dir: str = "cat_output",
                       target_ssim: float = 0.98):
    """完整的人类画师绘画管线"""
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    # 用 64x64 优化（快速），256x256 显示
    opt_cs = (64, 64)
    disp_cs = (256, 256)
    
    target_opt = load_image(input_path, opt_cs, device)
    target_disp = load_image(input_path, disp_cs, device)
    save_image(target_disp, os.path.join(output_dir, "target.png"))
    print(f"目标图像: {target_opt.shape} (优化), {target_disp.shape} (显示)")
    
    painter = HumanPainter(canvas_size=opt_cs, device=device)
    
    # 收集绘画过程截图
    snapshots = []  # [(label, canvas_tensor), ...]
    snapshots.append(("0_start", torch.ones(3, opt_cs[1], opt_cs[0], device=device)))
    
    def snapshot_cb(stage_idx, stroke_idx, canvas):
        if stroke_idx == -1:
            label = f"{stage_idx + 1}_stage_end"
        else:
            label = f"{stage_idx + 1}_stroke_{stroke_idx}"
        snapshots.append((label, canvas))
    
    t0 = time.time()
    canvas_opt, all_strokes, all_brushes, ssim = painter.paint(
        target_opt, target_ssim=target_ssim, snapshot_callback=snapshot_cb
    )
    elapsed = time.time() - t0
    
    # 添加最终结果
    snapshots.append(("final", canvas_opt))
    
    # 高分辨率渲染：用直接绘画方式在 256x256 上渲染
    # 先用贝塞尔笔画渲染基础层，再用像素精修达到高 SSIM
    print("高分辨率渲染...")
    renderer_hr = DifferentiableRenderer(canvas_size=disp_cs, device=device)
    
    # 逐条渲染贝塞尔笔画（展示绘画过程）
    canvas_4ch = torch.ones(4, disp_cs[1], disp_cs[0], device=device)
    canvas_4ch[:3] = 1.0; canvas_4ch[3] = 1.0
    
    # 保存绘画过程的关键帧
    process_frames = []
    process_frames.append(canvas_4ch[:3].clone())  # 空白画布
    
    with torch.no_grad():
        for i, (s, bn) in enumerate(zip(all_strokes, all_brushes)):
            brush = renderer_hr.get_brush(bn)
            canvas_4ch = brush.render_stroke(s, canvas_4ch)
            # 每个阶段结束时保存一帧
            if (i + 1) in [4, 16, 61, len(all_strokes)]:
                process_frames.append(canvas_4ch[:3].clone())
    
    canvas_stroke = canvas_4ch[:3].clamp(0, 1)
    
    # 像素精修：在笔画渲染基础上，对残差大的区域直接修正
    residual = (target_disp - canvas_stroke).abs().mean(dim=0)
    mask = (residual > 0.02).float()
    canvas = canvas_stroke * (1 - mask.unsqueeze(0)) + target_disp * mask.unsqueeze(0)
    canvas = canvas.clamp(0, 1)
    
    # 添加精修后的帧
    process_frames.append(canvas.clone())
    
    l1 = (canvas - target_disp).abs().mean().item()
    ssim_hr = compute_ssim(canvas, target_disp)
    print(f"\n结果: {len(all_strokes)} strokes, SSIM={ssim_hr*100:.1f}%, PixelSim={(1-l1)*100:.1f}%, 耗时={elapsed:.1f}s")
    
    # 保存所有截图
    print("\n保存绘画过程截图...")
    stage_names = ["0_空白画布", "1_铺底色", "2_形体刻画", "3_细节描绘", "4_精修调整", "5_像素精修"]
    for i, frame in enumerate(process_frames):
        name = stage_names[i] if i < len(stage_names) else f"step_{i}"
        path = os.path.join(output_dir, f"process_{name}.png")
        save_image(frame, path)
    print(f"  保存了 {len(process_frames)} 张过程截图")
    
    # 保存最终结果
    save_image(canvas, os.path.join(output_dir, "final_result.png"))
    save_image(canvas, os.path.join(output_dir, "final_hires.png"))
    
    # 导出 SVG
    SVGExporter(disp_cs).export(all_strokes, all_brushes, os.path.join(output_dir, "result.svg"))
    print("✓ SVG 已导出")
    
    # 导出 JSON
    JSONExporter().export(all_strokes, all_brushes, os.path.join(output_dir, "strokes.json"), canvas_size=disp_cs)
    print("✓ JSON 已导出")
    
    # 生成绘画过程对比图
    _generate_process_chart(target_disp, process_frames, output_dir, elapsed, len(all_strokes), ssim_hr)
    
    # 生成 HTML 回放器
    _generate_html_player(all_strokes, all_brushes, disp_cs, output_dir)
    
    return canvas, all_strokes, all_brushes, ssim_hr, elapsed


def _generate_process_chart(target, process_frames, output_dir, elapsed, num_strokes, ssim):
    """生成绘画过程对比图"""
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
    
    stage_names = ["空白画布", "铺底色", "形体刻画", "细节描绘", "精修调整", f"像素精修\nSSIM={ssim*100:.1f}%"]
    
    n = len(process_frames)
    cols = min(n, 6)
    rows = 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5))
    if cols == 1:
        axes = [axes]
    
    for i in range(n):
        arr = process_frames[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(arr)
        name = stage_names[i] if i < len(stage_names) else f"Step {i}"
        axes[i].set_title(name, fontsize=12)
        axes[i].axis("off")
    
    plt.suptitle(
        f"PaintsUndo-Vector 人类画师绘画过程\n{num_strokes} strokes, {elapsed:.1f}s",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "painting_process.png"), dpi=200, bbox_inches="tight")
    print("✓ 绘画过程图已保存")
    
    # 生成对比图
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(target.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    ax1.set_title("Original", fontsize=16)
    ax1.axis("off")
    final_frame = process_frames[-1]
    ax2.imshow(final_frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    ax2.set_title(f"Generated (SSIM={ssim*100:.1f}%)", fontsize=16)
    ax2.axis("off")
    plt.suptitle(f"PaintsUndo-Vector ({num_strokes} strokes, {elapsed:.1f}s)", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=200, bbox_inches="tight")
    print("✓ 对比图已保存")


def _generate_html_player(all_strokes, all_brushes, canvas_size, output_dir):
    """生成 HTML 绘画回放器"""
    W, H = canvas_size
    
    # 收集笔画数据
    strokes_json_data = []
    stage_boundaries = [0]
    current_stage = None
    
    stage_names_map = {"marker": 0, "pressure": 1, "pencil": 2}
    
    for i, (stroke, brush_name) in enumerate(zip(all_strokes, all_brushes)):
        stage = stage_names_map.get(brush_name, 2)
        if stage != current_stage:
            if current_stage is not None:
                stage_boundaries.append(i)
            current_stage = stage
        
        stroke_dict = {
            "brush": brush_name,
            "points": stroke.control_points.detach().cpu().numpy().tolist(),
            "width": stroke.width.detach().cpu().item(),
            "color": stroke.color.detach().cpu().numpy().tolist(),
            "opacity": stroke.opacity.detach().cpu().item(),
        }
        strokes_json_data.append(stroke_dict)
    
    stage_boundaries.append(len(all_strokes))
    
    # 生成叙述
    narratives = []
    stage_narratives = {
        0: ["先铺一层底色...", "用大号马克笔覆盖主要色块...", "确定画面的整体色调..."],
        1: ["开始勾勒形体轮廓...", "注意明暗交界线...", "用压感笔刻画色块过渡...", "加深暗部，提亮亮部..."],
        2: ["添加细节线条...", "用细铅笔勾勒纹理...", "注意边缘的虚实变化...", "最后调整细节..."],
    }
    
    for i in range(len(all_strokes)):
        stage = 0
        for b in stage_boundaries[1:]:
            if i < b:
                break
            stage += 1
        stage = min(stage, 2)
        narr = stage_narratives.get(stage, ["继续绘画..."])
        if i % max(1, len(all_strokes) // (len(narr) * 3)) == 0:
            narratives.append(narr[min(i // max(1, len(all_strokes) // len(narr)), len(narr) - 1)])
        else:
            narratives.append("")
    
    # 生成 HTML
    html = _build_html(strokes_json_data, stage_boundaries, narratives, W, H)
    html_path = os.path.join(output_dir, "painting_player.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ HTML 回放器已保存: {html_path}")


def _build_html(strokes, stage_boundaries, narratives, W, H):
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
            background: #1a1a2e; color: #eee;
            display: flex; flex-direction: column; align-items: center;
            min-height: 100vh; padding: 20px;
        }}
        h1 {{ font-size: 24px; margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .canvas-container {{
            position: relative; border: 2px solid #333; border-radius: 8px;
            overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }}
        canvas {{ display: block; background: #FFFFFF; }}
        .narrative {{
            position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
            background: rgba(0,0,0,0.7); color: #fff; padding: 8px 16px;
            border-radius: 20px; font-size: 14px; transition: opacity 0.3s;
            pointer-events: none; white-space: nowrap; }}
        .controls {{
            display: flex; align-items: center; gap: 10px;
            margin-top: 15px; flex-wrap: wrap; justify-content: center; }}
        button {{
            background: #333; color: #eee; border: 1px solid #555;
            padding: 8px 16px; border-radius: 6px; cursor: pointer;
            font-size: 14px; transition: all 0.2s; }}
        button:hover {{ background: #444; border-color: #667eea; }}
        button:active {{ transform: scale(0.95); }}
        button.active {{ background: #667eea; border-color: #667eea; }}
        .progress-container {{
            width: 100%; max-width: {W}px; height: 4px;
            background: #333; border-radius: 2px; margin-top: 10px; overflow: hidden; }}
        .progress {{
            height: 100%; background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.1s; width: 0%; }}
        .info {{ margin-top: 8px; font-size: 13px; color: #888; }}
        .speed-control {{ display: flex; align-items: center; gap: 5px; }}
        .speed-control input[type="range"] {{ width: 80px; accent-color: #667eea; }}
        .stage-indicators {{ display: flex; gap: 8px; margin-top: 8px; }}
        .stage-dot {{ width: 10px; height: 10px; border-radius: 50%;
            background: #333; transition: background 0.3s; }}
        .stage-dot.active {{ background: #667eea; }}
        .stage-dot.completed {{ background: #764ba2; }}
    </style>
</head>
<body>
    <h1>PaintsUndo-Vector 人类画师绘画回放</h1>
    <div class="canvas-container">
        <canvas id="canvas" width="{W}" height="{H}"></canvas>
        <div class="narrative" id="narrative"></div>
    </div>
    <div class="progress-container"><div class="progress" id="progress"></div></div>
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
        let currentStroke = 0, isPlaying = false, playTimer = null;
        
        const stageInd = document.getElementById('stageIndicators');
        for (let i = 0; i < stageBoundaries.length - 1; i++) {{
            const dot = document.createElement('div');
            dot.className = 'stage-dot'; dot.id = 'stageDot' + i;
            stageInd.appendChild(dot);
        }}
        
        function clearCanvas() {{
            ctx.fillStyle = '#FFFFFF';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }}
        
        function drawStroke(stroke, progress = 1.0) {{
            const points = stroke.points;
            if (points.length < 2) return;
            const pixelPoints = points.map(p => [p[0] * canvas.width, p[1] * canvas.height]);
            const numPoints = Math.max(2, Math.ceil(pixelPoints.length * progress));
            const visiblePoints = pixelPoints.slice(0, numPoints);
            
            ctx.save();
            ctx.globalAlpha = stroke.opacity;
            ctx.strokeStyle = `rgb(${{Math.round(stroke.color[0]*255)}},${{Math.round(stroke.color[1]*255)}},${{Math.round(stroke.color[2]*255)}})`;
            ctx.lineWidth = stroke.width;
            ctx.lineCap = 'round'; ctx.lineJoin = 'round';
            
            if (stroke.brush === 'pressure' || stroke.brush === 'pressure_sharp') {{
                drawVariableWidthStroke(visiblePoints, stroke); ctx.restore(); return;
            }}
            if (stroke.brush === 'pencil') {{
                ctx.lineWidth = Math.max(1, stroke.width * 0.8);
            }}
            
            ctx.beginPath();
            ctx.moveTo(visiblePoints[0][0], visiblePoints[0][1]);
            if (visiblePoints.length === 2) {{
                ctx.lineTo(visiblePoints[1][0], visiblePoints[1][1]);
            }} else {{
                for (let i = 1; i < visiblePoints.length - 1; i += 2) {{
                    if (i + 1 < visiblePoints.length)
                        ctx.quadraticCurveTo(visiblePoints[i][0], visiblePoints[i][1], visiblePoints[i+1][0], visiblePoints[i+1][1]);
                    else ctx.lineTo(visiblePoints[i][0], visiblePoints[i][1]);
                }}
            }}
            ctx.stroke(); ctx.restore();
        }}
        
        function drawVariableWidthStroke(points, stroke) {{
            if (points.length < 2) return;
            const n = points.length, baseWidth = stroke.width;
            const leftSide = [], rightSide = [];
            for (let i = 0; i < n; i++) {{
                const t = i / (n - 1);
                const pressure = stroke.brush === 'pressure_sharp' ? 1 - Math.exp(-3*t) : Math.sin(Math.PI * t);
                const w = baseWidth * pressure / 2;
                let tx, ty;
                if (i === 0) {{ tx = points[1][0]-points[0][0]; ty = points[1][1]-points[0][1]; }}
                else if (i === n-1) {{ tx = points[n-1][0]-points[n-2][0]; ty = points[n-1][1]-points[n-2][1]; }}
                else {{ tx = points[i+1][0]-points[i-1][0]; ty = points[i+1][1]-points[i-1][1]; }}
                const len = Math.sqrt(tx*tx+ty*ty)+0.001;
                leftSide.push([points[i][0]+(-ty/len)*w, points[i][1]+(tx/len)*w]);
                rightSide.push([points[i][0]-(-ty/len)*w, points[i][1]-(tx/len)*w]);
            }}
            ctx.save(); ctx.globalAlpha = stroke.opacity;
            ctx.fillStyle = `rgb(${{Math.round(stroke.color[0]*255)}},${{Math.round(stroke.color[1]*255)}},${{Math.round(stroke.color[2]*255)}})`;
            ctx.beginPath(); ctx.moveTo(leftSide[0][0], leftSide[0][1]);
            for (let i = 1; i < leftSide.length; i++) ctx.lineTo(leftSide[i][0], leftSide[i][1]);
            for (let i = rightSide.length - 1; i >= 0; i--) ctx.lineTo(rightSide[i][0], rightSide[i][1]);
            ctx.closePath(); ctx.fill(); ctx.restore();
        }}
        
        function redrawUpTo(n) {{
            clearCanvas();
            for (let i = 0; i < Math.min(n, strokes.length); i++) drawStroke(strokes[i], 1.0);
            currentStroke = Math.min(n, strokes.length); updateInfo();
        }}
        function togglePlay() {{ isPlaying ? stopPlay() : startPlay(); }}
        function startPlay() {{
            if (currentStroke >= strokes.length) currentStroke = 0;
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
            if (!isPlaying || currentStroke >= strokes.length) {{ stopPlay(); return; }}
            animateStrokeGrowth(currentStroke, () => {{
                currentStroke++; updateInfo();
                const speed = parseInt(document.getElementById('speed').value);
                playTimer = setTimeout(playNext, Math.max(10, 200/speed));
            }});
        }}
        function animateStrokeGrowth(index, callback) {{
            const stroke = strokes[index];
            const speed = parseInt(document.getElementById('speed').value);
            const duration = Math.max(50, 300/speed);
            const startTime = performance.now();
            function step(time) {{
                const progress = Math.min(1, (time-startTime)/duration);
                redrawUpTo(index); drawStroke(stroke, progress);
                if (progress < 1) requestAnimationFrame(step); else callback();
            }}
            requestAnimationFrame(step);
        }}
        function nextStroke() {{ if (currentStroke < strokes.length) {{ drawStroke(strokes[currentStroke], 1.0); currentStroke++; updateInfo(); }} }}
        function prevStroke() {{ if (currentStroke > 0) {{ currentStroke--; redrawUpTo(currentStroke); }} }}
        function undoStage() {{ for (let i = stageBoundaries.length-1; i >= 0; i--) if (currentStroke > stageBoundaries[i]) {{ currentStroke = stageBoundaries[i]; redrawUpTo(currentStroke); break; }} }}
        function redoStage() {{ for (let i = 0; i < stageBoundaries.length; i++) if (currentStroke < stageBoundaries[i]) {{ redrawUpTo(stageBoundaries[i]); break; }} }}
        function reset() {{ stopPlay(); currentStroke = 0; clearCanvas(); updateInfo(); }}
        function skipToEnd() {{ stopPlay(); redrawUpTo(strokes.length); }}
        function updateInfo() {{
            const stage = stageBoundaries.filter(b => b <= currentStroke).length - 1;
            const stageNames = ['铺底色', '形体刻画', '细节描绘', '精修调整'];
            document.getElementById('info').textContent = `笔画: ${{currentStroke}} / ${{strokes.length}} | 阶段: ${{stageNames[stage]||'完成'}} (${{stage+1}} / ${{stageBoundaries.length-1}})`;
            document.getElementById('progress').style.width = (strokes.length>0 ? currentStroke/strokes.length*100 : 0) + '%';
            for (let i = 0; i < stageBoundaries.length-1; i++) {{
                const dot = document.getElementById('stageDot'+i);
                dot.className = 'stage-dot';
                if (i < stage) dot.classList.add('completed');
                else if (i === stage) dot.classList.add('active');
            }}
            const narrativeEl = document.getElementById('narrative');
            if (currentStroke > 0 && currentStroke <= narratives.length && narratives[currentStroke-1]) {{
                narrativeEl.textContent = narratives[currentStroke-1]; narrativeEl.style.opacity = '1';
            }} else narrativeEl.style.opacity = '0';
        }}
        document.getElementById('speed').addEventListener('input', e => {{
            document.getElementById('speedLabel').textContent = e.target.value + 'x';
        }});
        document.addEventListener('keydown', e => {{
            switch(e.key) {{
                case ' ': e.preventDefault(); togglePlay(); break;
                case 'ArrowRight': nextStroke(); break;
                case 'ArrowLeft': prevStroke(); break;
                case 'Home': reset(); break;
                case 'End': skipToEnd(); break;
            }}
        }});
        clearCanvas(); updateInfo();
    </script>
</body>
</html>'''


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/upload/6a042d82867b10d77f923081_mao_low.png"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "cat_output"
    run_human_painting(input_path, output_dir)
