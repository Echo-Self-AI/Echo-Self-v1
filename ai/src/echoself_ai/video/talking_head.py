"""Talking-head video generator for EchoSelf.

Produces a real MP4 video from a portrait image and audio waveform.

Technique (honest description):
--------------------------------
This is a **geometric talking-head animator**, not a neural talking-face model.
It works as follows:

1. The user's portrait image is used as every frame's background.
2. Audio amplitude is extracted per video frame.
3. The mouth region (bottom-centre of the face area) is geometrically
   deformed based on the per-frame amplitude: a dark oval is blended
   over the mouth area, with size proportional to the audio level.
4. Subtle eye-blink animations are added at regular intervals.
5. Frames are assembled and encoded to MP4 using imageio-ffmpeg.

This produces a genuinely animated video with the user's portrait, mouth
movement correlated with their voice, and eye blinks — all computed
deterministically from the audio signal.

What this is NOT:
- It is not a neural talking-face model (Wav2Lip, SadTalker, etc.).
- It does not use facial landmark detection or mesh deformation.
- The mouth deformation is an ellipse composite, not a real mouth shape.
- It will not produce photo-realistic lip synchronization.

The output is labelled "geometric portrait animation" in all UI surfaces.
A real Wav2Lip or SadTalker integration would replace this module while
keeping the same interface.
"""

from __future__ import annotations

import base64
import io
import math
import struct
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import imageio


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class TalkingHeadConfig:
    fps: int = 25
    output_width: int = 512
    output_height: int = 512
    # Mouth region as fraction of face height from bottom of image
    mouth_y_frac: float = 0.72   # centre of mouth (0=top, 1=bottom)
    mouth_x_frac: float = 0.50   # horizontal centre
    mouth_max_height_frac: float = 0.07  # max mouth-open height as frac of frame
    mouth_max_width_frac: float = 0.18   # max mouth width
    blink_interval_sec: float = 3.5
    blink_duration_frames: int = 4
    output_quality: int = 8      # imageio quality (1-10)
    # Motion: subtle head drift
    drift_px: int = 2            # max pixel drift for head-alive feel


@dataclass
class VideoResult:
    video_path: str
    duration_sec: float
    fps: int
    width: int
    height: int
    num_frames: int
    video_base64: str            # base64-encoded MP4
    audio_base64: str            # base64-encoded WAV (same audio, for reference)
    generation_method: str       # "geometric-portrait-animation"
    has_audio: bool


class TalkingHeadGenerator:
    """Generates a talking-head MP4 from a portrait and audio."""

    def __init__(self, config: Optional[TalkingHeadConfig] = None) -> None:
        self.config = config or TalkingHeadConfig()

    def generate(
        self,
        portrait_bytes: bytes,
        audio_wav_bytes: bytes,
        output_dir: str = "/tmp",
        job_id: str = "video",
    ) -> VideoResult:
        """Generate a talking-head video.

        Args:
            portrait_bytes: PNG/JPEG/WEBP portrait image bytes.
            audio_wav_bytes: 16-bit mono PCM WAV audio bytes.
            output_dir: Directory to save the MP4 file.
            job_id: Used for output filename.

        Returns:
            VideoResult with path, metadata, and base64-encoded MP4.
        """
        cfg = self.config

        # --- Load and prepare portrait frame ---
        portrait = Image.open(io.BytesIO(portrait_bytes)).convert("RGB")
        portrait = portrait.resize((cfg.output_width, cfg.output_height), Image.LANCZOS)

        # --- Parse audio ---
        amplitudes, sample_rate, audio_duration = self._parse_audio(audio_wav_bytes)

        # --- Determine video duration ---
        # Pad to at least 1 second, cap at audio length
        video_duration = max(1.0, audio_duration)
        num_frames = max(cfg.fps, int(video_duration * cfg.fps))

        # --- Resample amplitudes to num_frames ---
        frame_amplitudes = self._resample(amplitudes, num_frames)

        # --- Generate frames ---
        frames = self._make_frames(portrait, frame_amplitudes, num_frames)

        # --- Write MP4 ---
        out_path = Path(output_dir) / f"{job_id}.mp4"
        self._write_mp4(frames, audio_wav_bytes, out_path, cfg.fps)

        # --- Encode to base64 ---
        video_b64 = base64.b64encode(out_path.read_bytes()).decode("ascii")
        audio_b64 = base64.b64encode(audio_wav_bytes).decode("ascii")

        return VideoResult(
            video_path=str(out_path),
            duration_sec=video_duration,
            fps=cfg.fps,
            width=cfg.output_width,
            height=cfg.output_height,
            num_frames=num_frames,
            video_base64=video_b64,
            audio_base64=audio_b64,
            generation_method="geometric-portrait-animation",
            has_audio=True,
        )

    # ------------------------------------------------------------------
    # Frame generation
    # ------------------------------------------------------------------

    def _make_frames(
        self,
        portrait: Image.Image,
        frame_amplitudes: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        cfg = self.config
        W, H = cfg.output_width, cfg.output_height

        frames: list[np.ndarray] = []
        blink_period = int(cfg.blink_interval_sec * cfg.fps)

        for i, amp in enumerate(frame_amplitudes):
            frame = portrait.copy()
            draw = ImageDraw.Draw(frame)

            # --- Subtle head drift (alive feeling) ---
            drift_x = int(math.sin(i * 0.08) * cfg.drift_px)
            drift_y = int(math.cos(i * 0.11) * cfg.drift_px * 0.5)
            if drift_x != 0 or drift_y != 0:
                frame = frame.transform(
                    (W, H), Image.AFFINE,
                    (1, 0, -drift_x, 0, 1, -drift_y),
                    resample=Image.BILINEAR,
                )
                draw = ImageDraw.Draw(frame)

            # --- Mouth animation ---
            self._draw_mouth(draw, W, H, float(amp))

            # --- Eye blink ---
            blink_phase = i % blink_period
            if blink_phase < cfg.blink_duration_frames:
                blink_alpha = self._blink_alpha(blink_phase, cfg.blink_duration_frames)
                self._draw_blink(frame, draw, W, H, blink_alpha)

            frames.append(np.array(frame))

        return frames

    def _draw_mouth(self, draw: ImageDraw.ImageDraw, W: int, H: int, amplitude: float) -> None:
        """Draw mouth-open shape proportional to audio amplitude."""
        cfg = self.config
        cx = int(W * cfg.mouth_x_frac)
        cy = int(H * cfg.mouth_y_frac)

        # Smooth amplitude with sigmoid-like scaling
        a = max(0.0, min(1.0, amplitude))
        open_h = int(H * cfg.mouth_max_height_frac * a)
        open_w = int(W * cfg.mouth_max_width_frac * (0.5 + 0.5 * a))

        if open_h < 2:
            # Closed mouth — draw a thin line
            draw.line(
                [(cx - open_w // 2, cy), (cx + open_w // 2, cy)],
                fill=(40, 20, 20), width=max(1, open_w // 10),
            )
            return

        # Outer lip shape (dark outline)
        bbox_outer = [cx - open_w, cy - open_h // 3, cx + open_w, cy + open_h]
        draw.ellipse(bbox_outer, fill=(50, 25, 25), outline=(30, 15, 15), width=2)

        # Mouth interior (dark cavity)
        bbox_inner = [cx - open_w + 4, cy - open_h // 3 + 2, cx + open_w - 4, cy + open_h - 2]
        if bbox_inner[2] > bbox_inner[0] and bbox_inner[3] > bbox_inner[1]:
            draw.ellipse(bbox_inner, fill=(15, 8, 8))

        # Teeth hint when mouth is significantly open
        if a > 0.4:
            teeth_h = max(1, open_h // 4)
            bbox_teeth = [cx - open_w + 6, cy - teeth_h // 2, cx + open_w - 6, cy + teeth_h // 2]
            if bbox_teeth[2] > bbox_teeth[0]:
                draw.rectangle(bbox_teeth, fill=(210, 205, 195))

    def _blink_alpha(self, phase: int, duration: int) -> float:
        """0→1→0 blink curve."""
        t = phase / max(1, duration - 1)
        return math.sin(t * math.pi)

    def _draw_blink(
        self, frame: Image.Image, draw: ImageDraw.ImageDraw,
        W: int, H: int, alpha: float,
    ) -> None:
        """Overlay a semi-transparent eyelid bar."""
        eye_y_top = int(H * 0.33)
        eye_y_bot = int(H * 0.33 + H * 0.07 * alpha)
        for x_frac, eye_w in [(0.33, 0.12), (0.67, 0.12)]:
            cx = int(W * x_frac)
            ew = int(W * eye_w)
            bbox = [cx - ew, eye_y_top, cx + ew, eye_y_bot]
            if bbox[3] > bbox[1]:
                # Sample skin tone from above eye area
                try:
                    skin_px = frame.getpixel((cx, eye_y_top - 4))
                    lid_color = tuple(min(255, int(c * 0.8)) for c in skin_px[:3])
                except Exception:
                    lid_color = (180, 150, 130)
                draw.ellipse(bbox, fill=lid_color)

    # ------------------------------------------------------------------
    # Audio parsing
    # ------------------------------------------------------------------

    def _parse_audio(
        self, wav_bytes: bytes
    ) -> tuple[np.ndarray, int, float]:
        """Extract per-sample amplitude from WAV bytes.

        Returns (amplitudes_0_to_1, sample_rate, duration_sec).
        """
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                raw = wf.readframes(n_frames)
            # 16-bit PCM
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            # Normalise to [0, 1]
            peak = np.abs(samples).max()
            if peak > 0:
                samples = np.abs(samples) / peak
            duration = len(samples) / sr
            return samples, sr, duration
        except Exception:
            # Fallback: 1 second of moderate amplitude
            sr = 16000
            t = np.linspace(0, 1, sr)
            samples = 0.3 + 0.3 * np.sin(2 * np.pi * 3 * t)
            return samples, sr, 1.0

    def _resample(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Resample 1D array to target_len using linear interpolation."""
        if len(arr) == target_len:
            return arr
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, arr)

    # ------------------------------------------------------------------
    # MP4 writing
    # ------------------------------------------------------------------

    def _write_mp4(
        self,
        frames: list[np.ndarray],
        audio_wav_bytes: bytes,
        out_path: Path,
        fps: int,
    ) -> None:
        """Write frames + audio to an MP4 file using imageio-ffmpeg."""
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        H, W = frames[0].shape[:2]

        # Write silent video first
        silent_path = out_path.with_suffix(".silent.mp4")
        writer = imageio.get_writer(
            str(silent_path),
            fps=fps,
            codec="libx264",
            quality=self.config.output_quality,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # Write audio WAV to temp file
        audio_path = out_path.with_suffix(".audio.wav")
        audio_path.write_bytes(audio_wav_bytes)

        # Mux video + audio with ffmpeg
        import subprocess
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i", str(silent_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(out_path),
            ],
            capture_output=True,
        )

        # Cleanup temp files
        silent_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

        if result.returncode != 0:
            # Fallback: return silent video
            if silent_path.exists():
                silent_path.rename(out_path)
            else:
                raise RuntimeError(f"ffmpeg mux failed: {result.stderr.decode()[:200]}")
