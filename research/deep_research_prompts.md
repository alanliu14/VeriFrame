# Role
你是一位专门调研 AIGC 对抗技术的**技术情报官**。你的目标是搜集全球范围内最新的“AI 视频检测”开源情报。

# Task
我要开发一款名为 "TrueSight" 的 App，用于盲检（无水印/无元数据）的 AI 生成视频。请针对 2025 年底的技术环境，**搜索并整理**一份深度技术调研报告。

# 🔍 Research Constraints (核心搜索约束)
1.  **时间范围**：严格限制在 **2024年6月至2025年12月** 之间发布的论文、GitHub 项目和技术博客。
2.  **排除项**：忽略所有依赖“元数据(Metadata)”或“数字水印(Watermark)”的方案。忽略针对“换脸(FaceSwap)”的老旧方案。
3.  **目标模型**：必须包含针对 **OpenAI Sora 2, Google Veo 3, Runway Gen-4, Kling 1.6/2.6 (可灵), Minimax (海螺)** 的检测研究。

# Research Steps (请按此执行搜索)

## 1. 搜集“盲检”最新论文与代码 (Find the SOTA)
- **Action**: 在 arXiv, GitHub, Hugging Face 上搜索关键词 "Video Forgery Detection", "Diffusion Generated Video Detection", "GenAI Video Detection".
- **Target**: 寻找并在报告中列出 3-5 个**开源且有代码** (Official Implementation) 的顶级项目。
    - *关注点*：是否有项目专门提到利用 "Video-MAE", "CLIP Features" 或 "Frequency Analysis" 来检测 **Diffusion Transformer (DiT)** 架构？
- **Output**: 制作一个表格，包含：项目名称、GitHub 链接、核心算法原理、是否支持检测 Kling/Sora。

## 2. 挖掘国产模型检测方案 (Localization Search)
- **Action**: 专门搜索中文互联网（知乎、CSDN、微信公众号技术文章）和 GitHub 中文区。
- **Question**: 是否有开发者分享过针对 **“可灵 (Kling)”** 或 **“即梦 (Jimeng)”** 的检测实战经验？
- **Target**: 寻找关于“国产视频生成模型 频域特征”或“可灵视频 伪影分析”的技术文章。

## 3. 寻找落地技术栈 (Implementation Stack)
- **Action**: 搜索目前 Python 中处理视频推理最高效的库。
- **Comparison**: 查找对比 "ONNX Runtime", "TensorRT", "OpenVINO" 在移动端或云端部署视频分类模型的 Benchmark 数据。
- **Question**: 是否有现成的 "Python Video Deepfake Detection API" 开源库可以直接 pip install？

# Deliverable
请输出一份详细的**情报简报**，每一个提到的技术点或模型，都**必须附带真实的 URL 链接**供我验证。