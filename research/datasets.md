# AI Video Datasets Research

## 1. DeCoF (Detecting via Frame Consistency)
*   **Repo**: `wuwuwuyue/DeCoF` (Likely GitHub user)
*   **Source URL**: [Google Drive Folder](https://drive.google.com/drive/folders/1X4Gw4hkWfka42IaBQ6ImkDTGAeA9Wlk4)
* - [x] **DeCoF** (Deepfake Correction Framework): Contains Sora (v1), Kling (v1), Veo (v1).
- [ ] **Data Strategy (Hybrid)**:
    - **Foundation (90%)**: GenVideo-100K (Updated Sep 2025). Contains massive Sora v1 / Kling v1 data. This teaches the model "General Artifacts".
    - **Fine-tuning (10%)**: Manual collection of 2025 models (Sora 2, Veo 2). This aligns the model to the latest "Hyper-Stable" features.

## 2. GenVideo-100K
*   **Source**: [ModelScope (GenVideo-100K)](https://modelscope.cn/datasets/GenVideo/GenVideo-100K)
*   **Paper**: "DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark"
*   **Scale**: 100,000+ Videos.
*   **Generators**:
    *   **Tier 1 (SOTA)**: Sora, Kling, Gen-3 Alpha, Veo.
    *   **Tier 2 (Open Source)**: Stable Video Diffusion (SVD), ZeroScope, VideoCraft.
    *   **Tier 3 (Commercial)**: Pika, Runway Gen-2.
*   **Freshness**: **Updated Sep 23, 2025** on ModelScope.
*   **Why it works**: Even if some videos are from 2024, Deep Learning classifiers learn *intrinsic architecture artifacts* (e.g., DiT checkers) that persist from v1 to v2. We don't need 100% v2 data to catch v2 fakes.
*   **Relevance**: ⭐⭐⭐⭐⭐ (Perfect match for our target models)
*   **Status**: Check for HuggingFace link.

## 2. GenVideo / DeMamba
*   **Repo**: `chenhaoxing/GenVideo` or `DeMamba`
*   **Content**: "Million-Scale", includes `GenVideo-100K` (Light version).
*   **Relevance**: ⭐⭐⭐⭐ (Good for large scale training)
*   **Note**: Released Sep 2024.

## 3. WildDeepfake / AV-Deepfake1M
*   **Content**: Older generation (mostly face swaps, lip sync).
*   **Relevance**: ⭐⭐ (Good for rPPG baseline, but not for Sora/Kling).

## Action Plan
1.  Visit HuggingFace to search for `DeCoF` or `GenVideo`.
2.  Download a small subset (Calibration Set) to `dataset/ai`.
