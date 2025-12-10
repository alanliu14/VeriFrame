# 🚀 TrueSight MVP 实战手册：计划 + 代码

## 第一部分：执行路线图 (The Execution Plan)

我们不盲目写代码，而是按照**“数据验证 -> 阈值校准 -> 引擎封装”**的顺序推进。

### **阶段 1：环境与数据准备 (第 1-2 天)**
*   **动作**：
    1.  找一台有 NVIDIA 显卡的电脑（或租用 AutoDL/Colab）。
    2.  准备 **“校准数据集”**：
        *   文件夹 `dataset/real/`: 放入 10 个你用手机拍的真实视频。
        *   文件夹 `dataset/ai/`: 放入 10 个用 Kling/Sora/Runway 生成的视频。
        *   *关键点*：请务必包含几个经过微信传输后的视频，以模拟真实环境。
    3.  安装依赖库（见下文代码注释）。

### **阶段 2：基准测试与阈值“炼丹” (第 3-4 天)**
*   **动作**：
    1.  运行我提供的 **统一引擎代码 (`truesight_engine.py`)**。
    2.  它会批量跑完你的数据集，并输出一个 CSV 或 JSON 报告。
    3.  **核心任务**：观察 `curvature_score` (曲率) 和 `ssim_drop_score` (SSIM跌落) 这两个数值。
        *   *目标*：找到一个数字 $T$，使得 `curvature > T` 能把 90% 的 AI 视频圈出来，同时误伤率最低。这个 $T$ 就是你 App 的核心机密。

### **阶段 3：API 服务化 (第 5-7 天)**
*   **动作**：
    1.  一旦阈值确定，将脚本中的 `Threshold` 常量修改为你测试出的数值。
    2.  使用 `FastAPI` 包裹这个 Class，部署到云服务器。

---

## 第二部分：统一核心引擎代码 (The Unified Code)

这是一个**“多合一”**脚本。它既包含了针对 Kling/Sora 的核心检测算法，也包含了**批量测试工具**。

请将以下代码保存为 `truesight_engine.py`。

```python
"""
TrueSight Core Engine (MVP Version)
===================================
功能：整合 ReStraV (几何检测)、SSIM Leakage (可灵检测)、rPPG (生物信号)
用途：既可以作为 Python 模块被 FastAPI 调用，也可以直接运行进行批量测试。
"""

import os
import cv2
import torch
import numpy as np
import argparse
import json
import warnings
from scipy.signal import welch
from skimage.metrics import structural_similarity as ssim
from decord import VideoReader, cpu
import torchvision.transforms as T

# 忽略烦人的警告
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 配置区域 (根据阶段 2 的测试结果来微调这里)
# ==========================================
CONFIG = {
    # ReStraV 阈值：潜空间曲率大于此值视为 AI (针对 Sora/Veo/Gen-4)
    "THRESHOLD_CURVATURE": 0.25, 
    
    # Kling 阈值：SSIM 跌落超过此负值视为 AI (针对 Kling 图生视频)
    "THRESHOLD_SSIM_DROP": -0.05,
    
    # rPPG 阈值：心率能量占比低于此值视为 AI (针对即梦/数字人)
    "THRESHOLD_PULSE_RATIO": 0.04,
    
    # 抽帧设置
    "SAMPLE_FRAMES": 16,  # 每次分析抽多少帧
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

class TrueSightEngine:
    def __init__(self):
        print(f"🚀 Initializing TrueSight Engine on {CONFIG['DEVICE']}...")
        self.device = CONFIG['DEVICE']
        
        # 1. 加载 DINOv2 模型 (用于 ReStraV 几何分析)
        # 使用 facebookresearch/dinov2 的小模型 (vits14) 以平衡速度和精度
        try:
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.dino.to(self.device)
            self.dino.eval()
            print("✅ DINOv2 Model Loaded Successfully.")
        except Exception as e:
            print(f"❌ Failed to load DINOv2: {e}")
            print("Please run: pip install torch torchvision")
            exit(1)

        # 图像预处理标准
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_frames(self, video_path, num_frames=16, strategy="uniform"):
        """工具函数：从视频中读取帧"""
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            if total < num_frames:
                indices = range(total)
            else:
                if strategy == "uniform":
                    indices = np.linspace(0, total - 1, num_frames).astype(int)
                elif strategy == "start": # 针对 Kling 检测，只取前部
                    indices = range(min(total, 15))
            
            frames = vr.get_batch(indices).asnumpy()
            return frames
        except Exception as e:
            print(f"⚠️ Error reading video {video_path}: {e}")
            return None

    # ---------------------------------------------------------
    # 🕵️‍♂️ 核心算法 1: ReStraV (针对 Sora, Gen-4, Veo)
    # 原理：计算 DINOv2 潜空间中的轨迹曲率。AI 视频轨迹通常更弯曲/杂乱。
    # ---------------------------------------------------------
    def check_latent_geometry(self, video_path):
        frames = self._get_frames(video_path, num_frames=CONFIG["SAMPLE_FRAMES"])
        if frames is None: return 0.0

        # 提取特征
        feats = []
        with torch.no_grad():
            for frame in frames:
                input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
                # DINOv2 输出 [1, 384] 的特征向量
                feat = self.dino(input_tensor)
                feats.append(feat.cpu().numpy())
        
        feats = np.vstack(feats) # [T, D]
        
        # 计算轨迹曲率 (简化版：计算相邻速度向量的余弦距离)
        # 1. 计算位移向量 (Velocity)
        diffs = np.diff(feats, axis=0) 
        # 2. 归一化位移向量
        norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-6
        normalized_diffs = diffs / norms
        
        # 3. 计算相邻向量的夹角
        angles = []
        for i in range(len(normalized_diffs) - 1):
            # Dot product
            cos_theta = np.dot(normalized_diffs[i], normalized_diffs[i+1].T)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angles.append(np.arccos(cos_theta)) # 得到弧度
            
        avg_curvature = np.mean(angles) if angles else 0.0
        return float(avg_curvature)

    # ---------------------------------------------------------
    # 🕵️‍♂️ 核心算法 2: SSIM Leakage (专杀 Kling/可灵)
    # 原理：检测“图生视频”前几帧的硬编码特征导致的图像质量突变
    # ---------------------------------------------------------
    def check_kling_leakage(self, video_path):
        # 仅读取前 15 帧
        frames = self._get_frames(video_path, num_frames=15, strategy="start")
        if frames is None or len(frames) < 10: return 0.0

        ssim_scores = []
        for i in range(len(frames) - 1):
            # 转灰度计算 SSIM 更快
            f1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            f2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            score = ssim(f1, f2)
            ssim_scores.append(score)
        
        # 寻找最大的“跌落” (Min Diff)
        # 真实视频变化平滑，AI 在接管生成的瞬间会有突变
        diffs = np.diff(ssim_scores)
        max_drop = np.min(diffs) if len(diffs) > 0 else 0.0
        return float(max_drop)

    # ---------------------------------------------------------
    # 🕵️‍♂️ 核心算法 3: rPPG (专杀 Jimeng/即梦/数字人)
    # 原理：检测面部微弱的心跳血流信号 (FFT分析)
    # ---------------------------------------------------------
    def check_rppg(self, video_path):
        # 简化版：暂不使用人脸检测模型，直接取中心区域 (假设用户拍的人脸在中间)
        # 生产环境建议加上 MTCNN
        frames = self._get_frames(video_path, num_frames=90) # 需要约3秒数据
        if frames is None: return 1.0 # 无法检测则默认通过

        green_signals = []
        for frame in frames:
            h, w, _ = frame.shape
            # 取画面中心 30% 区域
            roi = frame[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65), :]
            g_mean = np.mean(roi[:, :, 1]) # Green channel
            green_signals.append(g_mean)
        
        # 信号处理
        signal = np.array(green_signals)
        signal = signal - np.mean(signal) # 去直流
        
        # 功率谱分析
        freqs, psd = welch(signal, fs=30.0) # 假设 30fps
        
        # 计算心率频段 (0.8Hz - 2.5Hz) 的能量占比
        valid_mask = (freqs >= 0.8) & (freqs <= 2.5)
        heart_energy = np.sum(psd[valid_mask])
        total_energy = np.sum(psd) + 1e-6
        
        return float(heart_energy / total_energy)

    # ---------------------------------------------------------
    # 🧠 主分析入口 (Orchestrator)
    # ---------------------------------------------------------
    def analyze_video(self, video_path):
        result = {
            "filename": os.path.basename(video_path),
            "is_ai": False,
            "confidence": 0.0,
            "primary_reason": "Real",
            "scores": {}
        }

        # 1. 跑 ReStraV (通用几何检测)
        curv_score = self.check_latent_geometry(video_path)
        result["scores"]["curvature"] = curv_score
        
        # 2. 跑 Kling Leakage (特定伪影)
        drop_score = self.check_kling_leakage(video_path)
        result["scores"]["ssim_drop"] = drop_score

        # 3. 判定逻辑 (Ensemble Logic)
        
        # 判定 A: 如果是 Kling 典型的 SSIM 突降
        if drop_score < CONFIG["THRESHOLD_SSIM_DROP"]:
            result["is_ai"] = True
            result["primary_reason"] = "Kling_Artifact_Detected"
            result["confidence"] = 0.95
            return result

        # 判定 B: 如果潜空间轨迹太弯曲 (DiT 通用特征)
        if curv_score > CONFIG["THRESHOLD_CURVATURE"]:
            result["is_ai"] = True
            result["primary_reason"] = "High_Latent_Curvature (DiT)"
            # 简单的置信度映射逻辑
            conf = min(0.99, 0.5 + (curv_score - CONFIG["THRESHOLD_CURVATURE"]) * 2)
            result["confidence"] = conf
            return result
            
        # 判定 C: (可选) rPPG 心率检测
        # rppg_score = self.check_rppg(video_path)
        # result["scores"]["pulse_ratio"] = rppg_score
        # if rppg_score < CONFIG["THRESHOLD_PULSE_RATIO"]: ...

        return result

# ==========================================
# 💻 命令行接口 (CLI) - 用于阶段 2 批量测试
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrueSight Engine: Batch Analyzer")
    parser.add_argument("--input", type=str, required=True, help="Path to video file or directory")
    args = parser.parse_args()

    engine = TrueSightEngine()
    
    # 扫描文件
    files = []
    if os.path.isdir(args.input):
        valid_exts = ('.mp4', '.mov', '.avi', '.webm')
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(valid_exts)]
    else:
        files = [args.input]

    print(f"\n🔍 Starting analysis on {len(files)} files...\n")
    print(f"{'FILENAME':<30} | {'RESULT':<10} | {'REASON':<25} | {'CURVATURE':<10} | {'SSIM_DROP'}")
    print("-" * 100)

    for f_path in files:
        res = engine.analyze_video(f_path)
        
        # 打印表格行
        tag = "🔴 AI" if res["is_ai"] else "🟢 REAL"
        print(f"{res['filename'][:28]:<30} | {tag:<10} | {res['primary_reason']:<25} | {res['scores']['curvature']:.4f}     | {res['scores']['ssim_drop']:.4f}")

    print("\n✅ Batch Analysis Complete.")
```

---

## 第三部分：如何开始 (Quick Start)

### 1. 安装依赖
打开你的终端，运行：
```bash
pip install torch torchvision opencv-python numpy scipy decord scikit-image
```

### 2. 准备测试文件
随便找个文件夹（比如 `test_videos`），里面放两个视频：一个可灵生成的，一个手机拍的。

### 3. 运行代码
```bash
python truesight_engine.py --input test_videos
```

### 4. 观察输出
你会看到类似这样的结果：
```text
FILENAME                       | RESULT     | REASON                    | CURVATURE  | SSIM_DROP
----------------------------------------------------------------------------------------------------
kling_demo.mp4                 | 🔴 AI      | Kling_Artifact_Detected   | 0.4215     | -0.0823
my_cat_video.mp4               | 🟢 REAL    | Real                      | 0.1532     | -0.0012
```
