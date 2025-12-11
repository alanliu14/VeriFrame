"""
TrueSight Utility Functions
===========================
Video loading, frame extraction, and signal processing utilities.
"""

import numpy as np
import cv2
from typing import Optional, Literal
from scipy.signal import welch
from skimage.metrics import structural_similarity as ssim


class VideoLoader:
    """Efficient video frame extraction using decord."""

    def __init__(self):
        # Lazy import decord to allow graceful fallback
        try:
            from decord import VideoReader, cpu
            self._VideoReader = VideoReader
            self._ctx = cpu(0)
            self._use_decord = True
        except ImportError:
            self._use_decord = False
            print("Warning: decord not available, falling back to OpenCV")

    def get_frames(
        self,
        video_path: str,
        num_frames: int = 16,
        strategy: Literal["uniform", "start", "end"] = "uniform"
    ) -> Optional[np.ndarray]:
        """
        Extract frames from video.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            strategy:
                - "uniform": evenly spaced across video
                - "start": first N frames (for Kling detection)
                - "end": last N frames

        Returns:
            numpy array of shape [N, H, W, 3] (RGB) or None if failed
        """
        if self._use_decord:
            return self._load_with_decord(video_path, num_frames, strategy)
        else:
            return self._load_with_opencv(video_path, num_frames, strategy)

    def _load_with_decord(
        self,
        video_path: str,
        num_frames: int,
        strategy: str
    ) -> Optional[np.ndarray]:
        """Load frames using decord (faster, GPU-capable)."""
        try:
            vr = self._VideoReader(video_path, ctx=self._ctx)
            total = len(vr)

            if total == 0:
                return None

            indices = self._get_indices(total, num_frames, strategy)
            frames = vr.get_batch(indices).asnumpy()
            return frames

        except Exception as e:
            print(f"Error reading video with decord: {e}")
            return None

    def _load_with_opencv(
        self,
        video_path: str,
        num_frames: int,
        strategy: str
    ) -> Optional[np.ndarray]:
        """Fallback: load frames using OpenCV."""
        try:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total == 0:
                cap.release()
                return None

            indices = self._get_indices(total, num_frames, strategy)
            frames = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            if len(frames) == 0:
                return None

            return np.stack(frames, axis=0)

        except Exception as e:
            print(f"Error reading video with OpenCV: {e}")
            return None

    def _get_indices(
        self,
        total: int,
        num_frames: int,
        strategy: str
    ) -> np.ndarray:
        """Calculate frame indices based on strategy."""
        if total <= num_frames:
            return np.arange(total)

        if strategy == "uniform":
            return np.linspace(0, total - 1, num_frames).astype(int)
        elif strategy == "start":
            return np.arange(min(total, num_frames))
        elif strategy == "end":
            return np.arange(max(0, total - num_frames), total)
        else:
            return np.linspace(0, total - 1, num_frames).astype(int)

    def get_video_info(self, video_path: str) -> dict:
        """Get basic video information."""
        try:
            cap = cv2.VideoCapture(video_path)
            info = {
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
            }
            cap.release()
            return info
        except Exception as e:
            return {"error": str(e)}


class SignalProcessor:
    """Signal processing utilities for rPPG and frequency analysis."""

    @staticmethod
    def compute_ssim_sequence(frames: np.ndarray) -> np.ndarray:
        """
        Compute SSIM between consecutive frames.

        Args:
            frames: numpy array [N, H, W, 3]

        Returns:
            SSIM scores array of length N-1
        """
        ssim_scores = []

        for i in range(len(frames) - 1):
            # Convert to grayscale for faster SSIM computation
            f1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            f2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)

            score = ssim(f1, f2)
            ssim_scores.append(score)

        return np.array(ssim_scores)

    @staticmethod
    def find_ssim_drop(ssim_scores: np.ndarray) -> tuple[float, int]:
        """
        Find the maximum SSIM drop (sudden quality change).

        Returns:
            (max_drop_value, drop_index)
        """
        if len(ssim_scores) < 2:
            return 0.0, 0

        diffs = np.diff(ssim_scores)
        min_idx = np.argmin(diffs)
        return float(diffs[min_idx]), int(min_idx)

    @staticmethod
    def extract_rppg_signal(
        frames: np.ndarray,
        roi_ratio: float = 0.3
    ) -> np.ndarray:
        """
        Extract rPPG signal from video frames.

        Uses center region green channel mean as simplified pulse signal.

        Args:
            frames: numpy array [N, H, W, 3]
            roi_ratio: fraction of image to use as ROI (centered)

        Returns:
            1D signal array
        """
        signals = []

        for frame in frames:
            h, w, _ = frame.shape

            # Center crop ROI
            margin_h = int(h * (1 - roi_ratio) / 2)
            margin_w = int(w * (1 - roi_ratio) / 2)

            roi = frame[margin_h:h-margin_h, margin_w:w-margin_w, :]

            # Extract green channel mean (most sensitive to blood flow)
            g_mean = np.mean(roi[:, :, 1])
            signals.append(g_mean)

        return np.array(signals)

    @staticmethod
    def compute_heart_rate_energy_ratio(
        signal: np.ndarray,
        fps: float = 30.0,
        hr_range: tuple[float, float] = (0.8, 2.5)
    ) -> float:
        """
        Compute the ratio of energy in heart rate frequency band.

        Real faces should have detectable pulse (~0.8-2.5 Hz = 48-150 BPM).
        AI-generated faces typically lack this signal.

        Args:
            signal: 1D time series signal
            fps: video frame rate
            hr_range: (min_freq, max_freq) for heart rate band in Hz

        Returns:
            Energy ratio (0-1), lower values suggest AI
        """
        if len(signal) < 10:
            return 1.0  # Not enough data, assume real

        # Remove DC component
        signal = signal - np.mean(signal)

        # Compute power spectral density
        freqs, psd = welch(signal, fs=fps, nperseg=min(len(signal), 256))

        # Calculate energy in heart rate band
        hr_mask = (freqs >= hr_range[0]) & (freqs <= hr_range[1])
        hr_energy = np.sum(psd[hr_mask])
        total_energy = np.sum(psd) + 1e-9

        return float(hr_energy / total_energy)

    @staticmethod
    def compute_fft_spectrum(frame: np.ndarray) -> np.ndarray:
        """
        Compute 2D FFT magnitude spectrum for frequency analysis.

        Args:
            frame: single frame [H, W, 3] or [H, W]

        Returns:
            Log magnitude spectrum
        """
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Compute 2D FFT
        f_transform = np.fft.fft2(frame.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Log scale for visualization
        log_magnitude = np.log1p(magnitude)

        return log_magnitude
