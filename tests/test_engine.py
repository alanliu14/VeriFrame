"""
TrueSight Engine Unit Tests
===========================
Basic tests to verify engine initialization and output format.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.truesight_engine import TrueSightEngine, EngineConfig, AnalysisResult
from core.utils import VideoLoader, SignalProcessor


class TestEngineConfig:
    """Test EngineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EngineConfig()

        assert config.threshold_curvature == 0.25
        assert config.threshold_ssim_drop == -0.05
        assert config.threshold_pulse_ratio == 0.04
        assert config.sample_frames_geometry == 16
        assert config.dino_model_size == "small"

    def test_custom_config(self):
        """Test custom configuration."""
        config = EngineConfig(
            threshold_curvature=0.30,
            dino_model_size="large"
        )

        assert config.threshold_curvature == 0.30
        assert config.dino_model_size == "large"


class TestAnalysisResult:
    """Test AnalysisResult dataclass."""

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = AnalysisResult(
            filename="test.mp4",
            is_ai=True,
            confidence=0.85,
            primary_reason="High_Latent_Curvature",
            scores={"curvature": 0.35}
        )

        d = result.to_dict()

        assert d["filename"] == "test.mp4"
        assert d["is_ai"] is True
        assert d["confidence"] == 0.85
        assert d["scores"]["curvature"] == 0.35


class TestVideoLoader:
    """Test VideoLoader utility class."""

    def test_loader_initialization(self):
        """Test VideoLoader can be instantiated."""
        loader = VideoLoader()
        assert loader is not None

    def test_get_indices_uniform(self):
        """Test uniform frame index generation."""
        loader = VideoLoader()

        indices = loader._get_indices(100, 10, "uniform")

        assert len(indices) == 10
        assert indices[0] == 0
        assert indices[-1] == 99

    def test_get_indices_start(self):
        """Test start-based frame index generation."""
        loader = VideoLoader()

        indices = loader._get_indices(100, 15, "start")

        assert len(indices) == 15
        assert indices[0] == 0
        assert indices[-1] == 14

    def test_get_indices_short_video(self):
        """Test handling of video shorter than requested frames."""
        loader = VideoLoader()

        indices = loader._get_indices(5, 16, "uniform")

        assert len(indices) == 5


class TestSignalProcessor:
    """Test SignalProcessor utility class."""

    def test_ssim_drop_detection(self):
        """Test SSIM drop detection logic."""
        processor = SignalProcessor()

        # Simulate SSIM scores with a drop at index 3
        ssim_scores = np.array([0.98, 0.97, 0.96, 0.85, 0.84, 0.83])

        drop_value, drop_idx = processor.find_ssim_drop(ssim_scores)

        assert drop_idx == 2  # Drop happens between index 2 and 3
        assert drop_value < -0.05  # Significant drop

    def test_heart_rate_energy_ratio(self):
        """Test rPPG energy ratio computation."""
        processor = SignalProcessor()

        # Create synthetic signal with heart-rate-like oscillation (1 Hz)
        t = np.linspace(0, 3, 90)  # 3 seconds at 30 fps
        signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz = 60 BPM

        ratio = processor.compute_heart_rate_energy_ratio(signal, fps=30.0)

        # Should have significant energy in heart rate band
        assert ratio > 0.1

    def test_heart_rate_energy_ratio_noise(self):
        """Test rPPG with pure noise (no heart rate)."""
        processor = SignalProcessor()

        # Pure high-frequency noise
        np.random.seed(42)
        signal = np.random.randn(90) * 0.1

        ratio = processor.compute_heart_rate_energy_ratio(signal, fps=30.0)

        # Should have low energy in heart rate band
        assert ratio < 0.5


class TestTrueSightEngine:
    """Test TrueSightEngine main class."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Create engine instance (shared across tests in this class)."""
        config = EngineConfig(dino_model_size="small")
        return TrueSightEngine(config)

    def test_engine_initialization(self, engine):
        """Test that engine initializes successfully."""
        assert engine is not None
        assert engine.dino is not None
        assert engine.video_loader is not None

    def test_engine_device(self, engine):
        """Test that engine detects correct device."""
        import torch
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert engine.device == expected

    def test_trajectory_curvature_straight(self, engine):
        """Test curvature computation for straight trajectory."""
        # Straight line in feature space (low curvature)
        features = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4]
        ], dtype=np.float32)

        curvature = engine._compute_trajectory_curvature(features)

        # Straight line should have near-zero curvature
        assert curvature < 0.1

    def test_trajectory_curvature_curved(self, engine):
        """Test curvature computation for curved trajectory."""
        # Zigzag path (high curvature)
        features = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [2, 0, 0],
            [3, 1, 0],
            [4, 0, 0]
        ], dtype=np.float32)

        curvature = engine._compute_trajectory_curvature(features)

        # Zigzag should have high curvature
        assert curvature > 0.5


class TestIntegration:
    """Integration tests (require actual video files)."""

    @pytest.mark.skip(reason="Requires test video file")
    def test_analyze_real_video(self):
        """Test analysis on a real video file."""
        config = EngineConfig(dino_model_size="small")
        engine = TrueSightEngine(config)

        result = engine.analyze("dataset/test/sample.mp4")

        assert isinstance(result, AnalysisResult)
        assert result.filename == "sample.mp4"
        assert "curvature" in result.scores
        assert "ssim_drop" in result.scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
