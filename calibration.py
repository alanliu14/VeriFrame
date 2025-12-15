"""
TrueSight Calibration Tool
==========================
Run this script to determine optimal thresholds for your specific hardware and dataset.
"""

import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
from core.truesight_engine import TrueSightEngine, EngineConfig

def get_video_files(directory):
    valid_exts = ('.mp4', '.mov', '.avi', '.webm', '.mkv')
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(valid_exts):
                files.append(os.path.join(root, f))
    return files

def benchmark(engine, video_files, label):
    scores = {"curvature": [], "ssim_drop": [], "pulse_ratio": [], "range": [], "std": []}
    
    print(f"📊 Benchmarking {len(video_files)} {label} videos...")
    for f in tqdm(video_files):
        try:
            res = engine.analyze(f, run_rppg=False) 
            # New metrics
            scores["curvature"].append(res.scores.get("curvature_mean", 0))
            scores["range"].append(res.scores.get("curvature_range", 0))
            scores["std"].append(res.scores.get("curvature_std", 0))
            
            scores["ssim_drop"].append(res.scores.get("ssim_drop", 0))
            # scores["pulse_ratio"].append(res.scores.get("pulse_ratio", 0))
        except Exception as e:
            print(f"⚠️ Failed to analyze {f}: {e}")
            
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="dataset/real", help="Path to real videos")
    parser.add_argument("--ai", default="dataset/ai", help="Path to AI videos")
    parser.add_argument("--full", action="store_true", help="Run on full dataset (no sampling)")
    args = parser.parse_args()

    # 1. Load Engine (Uses default config -> Likely Giant now)
    engine = TrueSightEngine()

    # 2. Find Files
    real_files = get_video_files(args.real)
    ai_files = get_video_files(args.ai)
    
    if not real_files:
        print(f"❌ No videos found in {args.real}")
        return
    if not ai_files:
        print(f"❌ No videos found in {args.ai}")
        return

    # 3. Validation & Sampling
    if not real_files:
        print(f"❌ No videos found in {args.real}")
        return
    if not ai_files:
        print(f"❌ No videos found in {args.ai}")
        return

    # Sample AI files if there are too many (limit to 20 for speed) unless --full is set
    import random
    if not args.full and len(ai_files) > 20:
        print(f"⚠️ Too many AI videos ({len(ai_files)}). Sampling 20 for calibration. Use --full to run all.")
        random.shuffle(ai_files)
        ai_files = ai_files[:20]
    elif args.full:
        print(f"🚀 Running on FULL dataset ({len(ai_files)} videos). This may take a while...")

    # 3. Run Benchmark
    real_scores = benchmark(engine, real_files, "REAL")
    ai_scores = benchmark(engine, ai_files, "AI")

    # 4. Analysis & Recommendation
    print("\n" + "="*50)
    print("📈 CALIBRATION RESULTS (With Jitter Analysis)")
    print("="*50)
    
    # Range / Jitter (New Primary Metric)
    r_range = np.mean(real_scores["range"])
    a_range = np.mean(ai_scores["range"])
    
    print(f"\n[ReStraV / Curvature Range (Stability)]")
    print(f"  Real Mean: {r_range:.4f} (Lower is better)")
    print(f"  AI Mean:   {a_range:.4f}")
    print(f"  Gap:       {a_range - r_range:.4f} (Ideally > 0.5)")
    print(f"  👉 Recommended Threshold: {(r_range+a_range)/2:.4f}")

    # Standard Deviation 
    r_std = np.mean(real_scores["std"])
    a_std = np.mean(ai_scores["std"])
    print(f"\n[ReStraV / Curvature Std Dev]")
    print(f"  Real Mean: {r_std:.4f}")
    print(f"  AI Mean:   {a_std:.4f}")
    
    # SSIM Drop (Should be Near 0 for Real, Negative for Kling AI)
    r_drop = np.mean(real_scores["ssim_drop"])
    a_drop = np.mean(ai_scores["ssim_drop"])
    # Recommend threshold slightly below real min to avoid false positives
    # But for Kling, AI drops are usually -0.1 to -0.3. Real is > -0.01.
    rec_thresh_drop = -0.02 # Conservative default
    if len(ai_scores["ssim_drop"]) > 0:
         # Try to find a separation point
         rec_thresh_drop = (r_drop + a_drop) / 2

    print(f"\n[Kling / SSIM Drop]")
    print(f"  Real Mean: {r_drop:.4f}")
    print(f"  AI Mean:   {a_drop:.4f}")
    print(f"  👉 Recommended Threshold: {rec_thresh_drop:.4f}")

    print("\n✅ Done. Update EngineConfig with these values if satisfied.")

if __name__ == "__main__":
    main()
