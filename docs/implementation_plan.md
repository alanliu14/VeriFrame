# TrueSight Core Engine - Implementation Plan

## Goal
Implement the comprehensive video forgery detection engine (`TruesightEngine`) as a standalone Python module.
The engine will integrate three distinct detection vectors to identify AI-generated videos (Sora, Kling, Veo, Jimeng) without relying on watermarks.

**Hardware Context**: 
- **GPU**: NVIDIA RTX 5090 (Local).
- **Advantage**: Massive VRAM (32GB+) allowed. We can use larger DINOv2 backbones (e.g., `dinov2_vitl14` instead of `vits14`) for higher accuracy if needed, though we will start with the standard model for speed. Batch processing can be highly parallelized.

## User Review Required
> [!IMPORTANT]
> **Model Selection**: Defaulting to `dinov2_vitg14` (Giant) for maximum accuracy on RTX 3070 Ti. **I have configured the code to easily switch backbones.**

> [!NOTE]
> **Dependency Management**: Will use `conda` or `venv`? Assuming standard `pip` + `venv` per previous steps.

## Architecture Design

### 1. `TrueSightEngine` Class Structure
The core will be a single class in `truesight_engine.py` (or a package `truesight/core.py`) to maintain portability for the future API.

```python
class TrueSightEngine:
    def __init__(self, device="cuda", model_size="small"):
        # Load DINOv2 (Geometric Feature Extractor)
        # Load Face Detector (MTCNN or similar for rPPG) - Optional for Phase 1
        pass

    def analyze(self, video_path: str) -> dict:
        # Orchestrator: runs all checks and returns aggregated result
        pass
        
    def _check_restrav(self, frames) -> float:
        # 1. Extract Latent Features (DINOv2)
        # 2. Calculate Trajectory Curvature
        pass
        
    def _check_kling_leakage(self, frames) -> float:
        # 1. Calculate SSIM between first 10 frames
        # 2. Defect Knee Point / Sudden Drop
        pass
        
    def _check_rppg(self, frames) -> float:
        # 1. ROI Extraction (Face/Skin)
        # 2. Green Channel Signal Analysis
        # 3. FFT -> Heart Rate Energy Ratio
        pass
```

### 2. Algorithm Details

#### A. ReStraV (Representation Straightening)
- **Target**: Universal DiT models (Sora 2, Veo 3).
- **Logic**: 
    - Extract DINOv2 features for $N$ uniformly sampled frames.
    - Compute cosine similarity between velocity vectors $v_t = z_{t+1} - z_t$.
    - High curvature (low cosine similarity avg) = AI.
- **5090 Optimization**: Batch process all frames of a video at once instead of sequential loop.
- **[NEW] Jitter Analysis (Variance)**:
    - Instead of just Mean Curvature, we analyze the **Range (Max-Min)** of curvature angles.
    - **Hypothesis**: AI videos have high semantic instability (High Range), Real videos are stable (Low Range).
    - **Threshold**: Range > 0.757 = AI.

#### B. Kling Conditional Leakage
- **Target**: Kling AI (I2V mode).
- **Logic**:
    - Focus on frames 0-15.
    - Compute `SSIM(frame_t, frame_{t+1})`.
    - Look for $\min(\Delta SSIM)$ (the "Drop").
    - Negative spikes < Threshold = AI.
    > [!WARNING]
    > **Vulnerability**: If user trims the first 1-2 seconds, this method fails.
    > **Mitigation**: This is a "Fast Pass" check. If it fails (or video is trimmed), **ReStraV** (Method A) takes over as the primary detector, as Curvature artifacts persist throughout the video.

#### C. rPPG (Remote Photoplethysmography)
- **Target**: Jimeng / Virtual Humans / Face-swap.
- **Logic**:
    - Simple center-crop heuristic (MVP) or MTCNN face detection (Better).
    - Extract Green channel mean signal over time.
    - `scipy.signal.welch` to get Power Spectral Density (PSD).
    - If Energy in [0.8Hz, 2.5Hz] is too low (< 5% of total), flag as AI.

## Proposed Changes

### `core/`
#### [NEW] [truesight_engine.py](file:///c:/ai/fake_vids/VeriFrame/core/truesight_engine.py)
The main engine implementation.

#### [NEW] [utils.py](file:///c:/ai/fake_vids/VeriFrame/core/utils.py)
Helper functions for video loading (`decord`), tensor transformations, and FFT signal processing.

### `tests/`
#### [NEW] [test_engine.py](file:///c:/ai/fake_vids/VeriFrame/tests/test_engine.py)
Unit tests to verify the engine loads on 5090 and returns valid JSON structure.

## Verification Plan

### Automated Tests
1. **Unit Test**: Run `pytest` to ensure models load onto CUDA device (5090).
2. **Sanity Check**: Run engine on 1 dummy "Real" video and 1 dummy "Black Screen" (simulated AI) to check output format.

### Manual Verification
1. You will need to put 1 test video in `dataset/test/` and run the script.
2. Verify GPU usage via Task Manager to ensure 5090 is actually being used (CUDA load).

## Phase 6: Deep Research (Variance Analysis)
- **Goal**: Address low separability (0.06 gap) of Mean Curvature.
- **Method**: Implemented `debug_variance.py` to analyze frame-level jitter.
- **Result**: Found `Curvature Range` to be a superior metric (Gap > 0.25).
- **Integration**: Updated `truesight_engine.py` to use Range metric.

- **Objective**: Validate engine against high-consistency models (Sora 2, Kling 2.5, Veo 2).
- **Finding**: Simple Jitter metrics **failed**. Real videos (tripod) overlap with Veo 3 (0.37 vs 0.44).
- **New Strategy**: **Deep Learning Classifier**.
    - We cannot reliably hand-craft a threshold.
    - We must train a lightweight **MLP / Transformer Head** on top of DINOv2 features.
    - **Input**: Sequence of DINOv2 embeddings ($T \times D$).
    - **Output**: Probability of Real vs AI.
- **Data Requirement**: **GenVideo-100K** (from "DeMamba" paper).
    - **Scale**: ~100k videos.
    - **Generators**: Sora, Kling, Gen-3, Pika, etc.
    - **Source**: ModelScope.cn.

## Phase 7: API & Deployment
- **Goal**: Productize the engine.
- **Stack**: FastAPI + Uvicorn + Docker.
- **Status**:
    - `api_server.py`: Created.
    - `Dockerfile`: Created.
    - **Endpoints**: `/health`, `/analyze`.
3. **Robustness Test**: Manually trim a Kling video (remove first 3s) using ffmpeg, and verify `ReStraV` still flags it as AI.
