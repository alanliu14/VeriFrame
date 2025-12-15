"""
Variance Analysis Tool
======================
Deep dive into the 'texture' of the trajectory.
Hypothesis: Real handheld video has high-frequency jitter (high variance). AI is smooth (low variance).
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.truesight_engine import TrueSightEngine, EngineConfig

def compute_curvature_sequence(features):
    """Return the sequence of angles (curvatures) instead of just the mean."""
    velocities = np.diff(features, axis=0) # (T-1, D)
    norms = np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-8
    normalized_velocities = velocities / norms
    
    angles = []
    for i in range(len(normalized_velocities) - 1):
        v1 = normalized_velocities[i]
        v2 = normalized_velocities[i + 1]
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot)
        angles.append(angle)
    return np.array(angles)

def main():
    # 1. Setup
    real_dir = "dataset/real"
    ai_dirpath = "dataset/ai" # Need to scan subdirs
    
    engine = TrueSightEngine() 
    # Ensure we use the loader from engine
    
    # 2. Pick 1 Real and 1 AI video (Randomly)
    real_vids = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')]
    
    ai_vids = []
    for root, _, files in os.walk(ai_dirpath):
        for f in files:
            if f.endswith('.mp4'):
                ai_vids.append(os.path.join(root, f))
                
    if not real_vids or not ai_vids:
        print("Missing dataset files.")
        return

    r_vid = random.choice(real_vids)
    a_vid = random.choice(ai_vids)
    
    print(f"🕵️ Comparing:\n  Real: {os.path.basename(r_vid)}\n  AI:   {os.path.basename(a_vid)}")

    # 3. Extract Features (Heavy work)
    # Get MORE frames to see the jitter (e.g., 60 frames)
    # We need to bypass the engine's default logic to get raw sequences
    
    def get_seq(path):
        frames = engine.video_loader.get_frames(path, num_frames=60, strategy="uniform")
        if frames is None: return None
        
        feats = []
        with torch.no_grad():
            for frame in frames:
                t = engine.transform(frame).unsqueeze(0).to(engine.device)
                f = engine.dino(t)
                feats.append(f.cpu().numpy())
        return compute_curvature_sequence(np.vstack(feats))

    r_seq = get_seq(r_vid)
    a_seq = get_seq(a_vid)
    
    # 4. Analysis
    print("\n[Analysis Results]")
    print(f"{'Metric':<20} {'Real':<10} {'AI':<10} {'Gap'}")
    print("-" * 50)
    
    # Mean (Old Metric)
    r_mean = np.mean(r_seq)
    a_mean = np.mean(a_seq)
    print(f"{'Mean Curvature':<20} {r_mean:.4f}     {a_mean:.4f}     {r_mean-a_mean:.4f}")
    
    # Variance / Std Dev (New Metric?)
    r_std = np.std(r_seq)
    a_std = np.std(a_seq)
    print(f"{'Std Dev (Jitter)':<20} {r_std:.4f}     {a_std:.4f}     {r_std-a_std:.4f}")
    
    # Min/Max Range
    r_range = np.max(r_seq) - np.min(r_seq)
    a_range = np.max(a_seq) - np.min(a_seq)
    print(f"{'Range (Max-Min)':<20} {r_range:.4f}     {a_range:.4f}     {r_range-a_range:.4f}")
    
    print("\nIf 'Std Dev' or 'Range' shows a larger GAP than 'Mean', we found our silver bullet.")

if __name__ == "__main__":
    main()
