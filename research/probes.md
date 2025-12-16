# TrueSight Evaluation Probes (2025)

## Objective
To generate "Torture Test" videos using **Sora 2**, **Kling 2.5**, and **Veo 2** that specifically challenge the **Latent Curvature Range (Jitter)** metric.

## 1. The "Golden" Prompt (Rigid Geometry)
*Current Finding*: 2025 Models pass this (Range ~0.44). They are "Hyper-stable".
> "A continuous, slow forward camera push through an infinitely long, ornate marble hallway..."

## 2. The "Diamond" Prompt (High Complexity / Non-Rigid)
*Hypothesis*: AI handles rigid objects well, but struggles with **temporal consistency of crowds, fluids, and biological motion**.

### 🎥 Prompt: The Tokyo Crossing
> **English**:
> "Extremely crowded Shibuya Crossing at night in rain. Hundreds of pedestrians with umbrellas walking in different directions. Neon lights reflecting on wet pavement. Complex chaotic motion. 4k, 60fps, sharp focus."

> **中文**:
> "雨夜的东京涩谷十字路口，极度拥挤。数百名打着伞的行人向不同方向行走。霓虹灯在湿润路面上反射。复杂的混乱运动。4K，60fps，对焦清晰。"

### 🎥 Prompt: The Dancing Fabric
> **English**:
> "A dancer wearing a dress made of thousands of tiny LED lights spinning rapidly in a dark room. The fabric flows like liquid. Motion blur, long exposure traces. High dynamic range."

> **中文**:
> "黑暗房间里，一位舞者穿着由数千个微型LED灯组成的裙子快速旋转。布料像液体一样流动。动态模糊，长曝光轨迹。"
