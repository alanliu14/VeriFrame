"""
TrueSight Core Engine
=====================
AI-Generated Video Detection via Multi-Vector Analysis

Detection Methods:
1. ReStraV (Latent Geometry) - Universal DiT detection via DINOv2 trajectory curvature
2. Kling Leakage - SSIM-based detection of I2V conditional image artifacts
3. rPPG Analysis - Heart rate signal detection for face authenticity

Author: TrueSight Team
License: MIT
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import torch
import torchvision.transforms as T

from .utils import VideoLoader, SignalProcessor

warnings.filterwarnings("ignore")


@dataclass
class EngineConfig:
    """Configuration for TrueSight detection thresholds and parameters."""

    # ReStraV: Latent curvature threshold (higher = more likely AI)
    threshold_curvature: float = 0.25

    # Kling: SSIM drop threshold (more negative = more likely AI)
    threshold_ssim_drop: float = -0.05

    # rPPG: Heart rate energy ratio threshold (lower = more likely AI)
    threshold_pulse_ratio: float = 0.04

    # Frame sampling settings
    sample_frames_geometry: int = 16  # Frames for ReStraV analysis
    sample_frames_leakage: int = 15   # Frames for Kling detection
    sample_frames_rppg: int = 90      # Frames for rPPG (~3 seconds at 30fps)

    # DINOv2 model variant: "small", "base", "large", "giant"
    dino_model_size: Literal["small", "base", "large", "giant"] = "small"

    # Device selection
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class AnalysisResult:
    """Result of video analysis."""

    filename: str
    is_ai: bool
    confidence: float
    primary_reason: str
    scores: dict
    video_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "is_ai": self.is_ai,
            "confidence": self.confidence,
            "primary_reason": self.primary_reason,
            "scores": self.scores,
            "video_info": self.video_info
        }


class TrueSightEngine:
    """
    Multi-vector AI video detection engine.

    Combines multiple detection strategies:
    - ReStraV: Latent space geometry analysis using DINOv2
    - Kling Leakage: SSIM-based I2V artifact detection
    - rPPG: Remote photoplethysmography for face authentication
    """

    DINO_MODELS = {
        "small": "dinov2_vits14",
        "base": "dinov2_vitb14",
        "large": "dinov2_vitl14",
        "giant": "dinov2_vitg14"
    }

    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Initialize the TrueSight engine.

        Args:
            config: Engine configuration. Uses defaults if not provided.
        """
        self.config = config or EngineConfig()
        self.device = self.config.device

        print(f"Initializing TrueSight Engine on {self.device}...")

        # Initialize utilities
        self.video_loader = VideoLoader()
        self.signal_processor = SignalProcessor()

        # Load DINOv2 model
        self._load_dino_model()

        # Image preprocessing for DINOv2
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("TrueSight Engine initialized successfully.")

    def _load_dino_model(self):
        """Load DINOv2 model from torch hub."""
        model_name = self.DINO_MODELS.get(
            self.config.dino_model_size,
            "dinov2_vits14"
        )

        print(f"Loading DINOv2 model: {model_name}...")

        try:
            self.dino = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                trust_repo=True
            )
            self.dino.to(self.device)
            self.dino.eval()
            print(f"DINOv2 ({model_name}) loaded successfully.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 model: {e}\n"
                "Please ensure PyTorch is installed: pip install torch torchvision"
            )

    def analyze(self, video_path: str, run_rppg: bool = False) -> AnalysisResult:
        """
        Analyze a video for AI-generation artifacts.

        Args:
            video_path: Path to video file
            run_rppg: Whether to run rPPG analysis (slower, for face videos)

        Returns:
            AnalysisResult with detection results
        """
        filename = os.path.basename(video_path)

        # Get video info
        video_info = self.video_loader.get_video_info(video_path)

        result = AnalysisResult(
            filename=filename,
            is_ai=False,
            confidence=0.0,
            primary_reason="Real",
            scores={},
            video_info=video_info
        )

        # 1. ReStraV: Latent geometry analysis
        curvature = self._check_restrav(video_path)
        result.scores["curvature"] = curvature

        # 2. Kling Leakage: SSIM drop detection
        ssim_drop, drop_frame = self._check_kling_leakage(video_path)
        result.scores["ssim_drop"] = ssim_drop
        result.scores["ssim_drop_frame"] = drop_frame

        # 3. Optional: rPPG analysis
        if run_rppg:
            pulse_ratio = self._check_rppg(video_path)
            result.scores["pulse_ratio"] = pulse_ratio

        # Decision logic (ensemble)
        result = self._make_decision(result, run_rppg)

        return result

    def _check_restrav(self, video_path: str) -> float:
        """
        ReStraV: Detect AI via latent space trajectory curvature.

        AI-generated videos have more curved/erratic trajectories in
        DINOv2's representation space compared to natural videos.

        Returns:
            Average curvature score (higher = more likely AI)
        """
        frames = self.video_loader.get_frames(
            video_path,
            num_frames=self.config.sample_frames_geometry,
            strategy="uniform"
        )

        if frames is None or len(frames) < 3:
            return 0.0

        # Extract DINOv2 features for each frame
        features = []

        with torch.no_grad():
            for frame in frames:
                tensor = self.transform(frame).unsqueeze(0).to(self.device)
                feat = self.dino(tensor)
                features.append(feat.cpu().numpy())

        features = np.vstack(features)  # [T, D]

        # Compute trajectory curvature
        curvature = self._compute_trajectory_curvature(features)

        return float(curvature)

    def _compute_trajectory_curvature(self, features: np.ndarray) -> float:
        """
        Compute curvature of trajectory in feature space.

        Uses angle between consecutive velocity vectors.
        """
        if len(features) < 3:
            return 0.0

        # Compute velocity vectors (frame-to-frame displacement)
        velocities = np.diff(features, axis=0)

        # Normalize velocities
        norms = np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-8
        normalized_velocities = velocities / norms

        # Compute angles between consecutive velocity vectors
        angles = []
        for i in range(len(normalized_velocities) - 1):
            v1 = normalized_velocities[i]
            v2 = normalized_velocities[i + 1]

            # Cosine similarity -> angle
            cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_theta)
            angles.append(angle)

        return float(np.mean(angles)) if angles else 0.0

    def _check_kling_leakage(self, video_path: str) -> tuple[float, int]:
        """
        Detect Kling-style I2V artifacts via SSIM analysis.

        Kling videos often have a sudden quality jump around frame 5-10
        where the model transitions from the conditioned input image
        to fully generated content.

        Returns:
            (max_ssim_drop, frame_index)
        """
        frames = self.video_loader.get_frames(
            video_path,
            num_frames=self.config.sample_frames_leakage,
            strategy="start"
        )

        if frames is None or len(frames) < 3:
            return 0.0, 0

        # Compute SSIM sequence
        ssim_scores = self.signal_processor.compute_ssim_sequence(frames)

        # Find maximum drop
        drop_value, drop_idx = self.signal_processor.find_ssim_drop(ssim_scores)

        return drop_value, drop_idx

    def _check_rppg(self, video_path: str) -> float:
        """
        Detect AI-generated faces via rPPG (remote heart rate) analysis.

        Real faces show subtle color changes due to blood flow.
        AI-generated faces typically lack this biological signal.

        Returns:
            Heart rate energy ratio (lower = more likely AI)
        """
        # Need more frames for reliable rPPG
        frames = self.video_loader.get_frames(
            video_path,
            num_frames=self.config.sample_frames_rppg,
            strategy="uniform"
        )

        if frames is None or len(frames) < 30:
            return 1.0  # Not enough data, assume real

        # Get video FPS for accurate frequency analysis
        video_info = self.video_loader.get_video_info(video_path)
        fps = video_info.get("fps", 30.0)

        # Extract rPPG signal
        signal = self.signal_processor.extract_rppg_signal(frames)

        # Compute heart rate energy ratio
        ratio = self.signal_processor.compute_heart_rate_energy_ratio(signal, fps=fps)

        return ratio

    def _make_decision(
        self,
        result: AnalysisResult,
        include_rppg: bool
    ) -> AnalysisResult:
        """
        Make final AI/Real decision based on all scores.

        Priority:
        1. Kling leakage (specific, high confidence)
        2. High curvature (general DiT detection)
        3. rPPG (face-specific)
        """
        scores = result.scores

        # Check 1: Kling-style SSIM drop
        if scores.get("ssim_drop", 0) < self.config.threshold_ssim_drop:
            result.is_ai = True
            result.primary_reason = "Kling_I2V_Artifact"
            result.confidence = 0.92
            return result

        # Check 2: High latent curvature (DiT models)
        curvature = scores.get("curvature", 0)
        if curvature > self.config.threshold_curvature:
            result.is_ai = True
            result.primary_reason = "High_Latent_Curvature"
            # Scale confidence based on how far above threshold
            excess = curvature - self.config.threshold_curvature
            result.confidence = min(0.95, 0.6 + excess * 1.5)
            return result

        # Check 3: Low pulse ratio (AI faces)
        if include_rppg:
            pulse_ratio = scores.get("pulse_ratio", 1.0)
            if pulse_ratio < self.config.threshold_pulse_ratio:
                result.is_ai = True
                result.primary_reason = "No_Biological_Signal"
                result.confidence = 0.85
                return result

        # If nothing triggers, classify as real
        result.is_ai = False
        result.primary_reason = "Passed_All_Checks"
        # Confidence in "real" based on how far from thresholds
        curvature_margin = self.config.threshold_curvature - curvature
        result.confidence = min(0.9, 0.5 + curvature_margin * 2)

        return result

    def batch_analyze(
        self,
        video_paths: list[str],
        run_rppg: bool = False,
        verbose: bool = True
    ) -> list[AnalysisResult]:
        """
        Analyze multiple videos.

        Args:
            video_paths: List of video file paths
            run_rppg: Whether to run rPPG analysis
            verbose: Print progress

        Returns:
            List of AnalysisResult objects
        """
        results = []

        for i, path in enumerate(video_paths):
            if verbose:
                print(f"[{i+1}/{len(video_paths)}] Analyzing: {os.path.basename(path)}")

            result = self.analyze(path, run_rppg=run_rppg)
            results.append(result)

            if verbose:
                tag = "AI" if result.is_ai else "REAL"
                print(f"  -> {tag} ({result.confidence:.1%}) - {result.primary_reason}")

        return results


def main():
    """CLI entry point for batch analysis."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="TrueSight: AI-Generated Video Detection"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to video file or directory"
    )
    parser.add_argument(
        "--rppg",
        action="store_true",
        help="Enable rPPG analysis (slower, for face videos)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "base", "large", "giant"],
        help="DINOv2 model size (default: small)"
    )

    args = parser.parse_args()

    # Configure engine
    config = EngineConfig(dino_model_size=args.model_size)
    engine = TrueSightEngine(config)

    # Collect video files
    valid_exts = ('.mp4', '.mov', '.avi', '.webm', '.mkv')

    if os.path.isdir(args.input):
        video_paths = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(valid_exts)
        ]
    else:
        video_paths = [args.input]

    if not video_paths:
        print("No video files found.")
        return

    print(f"\nAnalyzing {len(video_paths)} video(s)...\n")
    print("-" * 80)

    # Run analysis
    results = engine.batch_analyze(video_paths, run_rppg=args.rppg)

    # Print summary
    print("-" * 80)
    print("\nSUMMARY:")
    print(f"{'FILENAME':<35} {'RESULT':<8} {'CONFIDENCE':<12} {'REASON'}")
    print("-" * 80)

    for r in results:
        tag = "AI" if r.is_ai else "REAL"
        print(f"{r.filename[:33]:<35} {tag:<8} {r.confidence:.1%}        {r.primary_reason}")

    # Save to JSON if requested
    if args.output:
        output_data = [r.to_dict() for r in results]
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
