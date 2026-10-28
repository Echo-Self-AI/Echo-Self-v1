"""Video frame representation, temporal container, and sequence I/O."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

from PIL import Image
import torch

from echoself_ai.vision.image_ops import load_image, save_image, tensor_to_pil


@dataclass
class VideoConfig:
    """Hyperparameters and configuration for video frame sequences."""

    fps: int = 16
    num_frames: int = 16
    width: int = 64
    height: int = 64
    channels: int = 3


class FrameSequence:
    """Temporal sequence of video frames stored as normalized PyTorch tensors.

    Tensor shape is strictly (T, C, H, W) with pixel values in [-1.0, 1.0].
    """

    def __init__(self, frames: torch.Tensor, fps: int = 16) -> None:
        if frames.dim() == 5:
            # Squeeze batch dimension if singleton: (1, T, C, H, W) -> (T, C, H, W)
            if frames.shape[0] == 1:
                frames = frames.squeeze(0)
            else:
                raise ValueError(
                    f"FrameSequence expects single video tensor (T, C, H, W), got {frames.shape}"
                )

        if frames.dim() != 4:
            raise ValueError(
                f"FrameSequence requires 4D tensor (T, C, H, W), got {frames.dim()}D tensor: {frames.shape}"
            )

        if frames.shape[1] != 3 and frames.shape[1] != 1:
            raise ValueError(
                f"Expected 1 or 3 channels (dim 1), got {frames.shape[1]} channels in {frames.shape}"
            )

        self.frames = frames.float().clamp(-1.0, 1.0)
        self.fps = max(1, int(fps))

    @property
    def num_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def channels(self) -> int:
        return self.frames.shape[1]

    @property
    def height(self) -> int:
        return self.frames.shape[2]

    @property
    def width(self) -> int:
        return self.frames.shape[3]

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: Union[int, slice]) -> Union[torch.Tensor, "FrameSequence"]:
        if isinstance(index, int):
            return self.frames[index]
        return FrameSequence(self.frames[index], fps=self.fps)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(self.num_frames):
            yield self.frames[i]

    def to_pil_frames(self) -> list[Image.Image]:
        """Convert all frames to PIL Image objects."""
        return [tensor_to_pil(frame) for frame in self.frames]

    @classmethod
    def from_pil_frames(cls, images: list[Image.Image], fps: int = 16) -> "FrameSequence":
        """Construct FrameSequence from a list of PIL Images."""
        if not images:
            raise ValueError("Cannot create FrameSequence from empty image list")

        w, h = images[0].size
        tensors: list[torch.Tensor] = []
        for img in images:
            if img.size != (w, h):
                img = img.resize((w, h), Image.Resampling.BILINEAR)
            img_rgb = img.convert("RGB")
            # Convert to (3, H, W) in [-1.0, 1.0]
            arr = torch.frombuffer(bytearray(img_rgb.tobytes()), dtype=torch.uint8).view(h, w, 3)
            arr = arr.permute(2, 0, 1).float() / 127.5 - 1.0
            tensors.append(arr)

        stacked = torch.stack(tensors, dim=0)
        return cls(stacked, fps=fps)

    def save_gif(self, path: Union[Path, str], loop: int = 0) -> Path:
        """Save frame sequence as an animated GIF."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        pil_frames = self.to_pil_frames()
        if not pil_frames:
            raise ValueError("No frames to save")

        # Frame duration in milliseconds
        duration_ms = int(1000.0 / self.fps)
        pil_frames[0].save(
            target_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=loop,
            format="GIF",
        )
        return target_path

    @classmethod
    def load_gif(cls, path: Union[Path, str], fps: Optional[int] = None) -> "FrameSequence":
        """Load an animated GIF file into a FrameSequence."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"GIF file not found: {target_path}")

        img = Image.open(target_path)
        images: list[Image.Image] = []

        try:
            frame_duration_ms = img.info.get("duration", 62)  # default ~16fps
            detected_fps = int(1000.0 / frame_duration_ms) if frame_duration_ms > 0 else 16
        except Exception:
            detected_fps = 16

        final_fps = fps if fps is not None else detected_fps

        frame_idx = 0
        while True:
            try:
                img.seek(frame_idx)
                images.append(img.copy().convert("RGB"))
                frame_idx += 1
            except EOFError:
                break

        return cls.from_pil_frames(images, fps=final_fps)

    def save_frame_directory(self, dir_path: Union[Path, str], prefix: str = "frame") -> Path:
        """Save individual frames as PNG images inside target directory."""
        target_dir = Path(dir_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(self.frames):
            frame_path = target_dir / f"{prefix}_{i:05d}.png"
            save_image(frame_path, frame)
        return target_dir


    @classmethod
    def load_frame_directory(
        cls,
        dir_path: Union[Path, str],
        fps: int = 16,
        pattern: str = "*.png",
        target_size: Optional[tuple[int, int]] = None,
    ) -> "FrameSequence":
        """Load directory of sequenced image frames into a FrameSequence."""
        target_dir = Path(dir_path)
        if not target_dir.exists():
            raise FileNotFoundError(f"Directory not found: {target_dir}")

        frame_files = sorted(target_dir.glob(pattern))
        if not frame_files:
            raise ValueError(f"No frames matching '{pattern}' in {target_dir}")

        tensors: list[torch.Tensor] = []
        for file_path in frame_files:
            if target_size is not None:
                from echoself_ai.vision.image_ops import ImageConfig
                tensors.append(load_image(file_path, config=ImageConfig(target_size=target_size)))
            else:
                with Image.open(file_path) as img:
                    img_rgb = img.convert("RGB")
                    w, h = img_rgb.size
                    arr = torch.frombuffer(bytearray(img_rgb.tobytes()), dtype=torch.uint8).view(h, w, 3)
                    tensor = arr.permute(2, 0, 1).float() / 127.5 - 1.0
                    tensors.append(tensor)

        stacked = torch.stack(tensors, dim=0)
        return cls(stacked, fps=fps)



def generate_synthetic_video(
    num_frames: int = 16,
    width: int = 64,
    height: int = 64,
    fps: int = 16,
    pattern: str = "bounce",
) -> FrameSequence:
    """Generate deterministic synthetic video sequence for unit testing and validation.

    Patterns:
        - "bounce": A bright square bouncing horizontally across a dark background.
        - "sweep": A smooth color gradient sweeping through hue/brightness over time.
    """
    frames = torch.zeros(num_frames, 3, height, width, dtype=torch.float32)

    if pattern == "bounce":
        box_size = max(4, min(width, height) // 4)
        for t in range(num_frames):
            # Triangle wave position across width
            norm_t = (t / max(1, num_frames - 1)) * 2.0
            pos_ratio = norm_t if norm_t <= 1.0 else (2.0 - norm_t)
            start_x = int(pos_ratio * (width - box_size))
            start_y = (height - box_size) // 2

            # Background: dark gray (-0.8)
            frames[t].fill_(-0.8)
            # Moving box: bright (+0.9)
            frames[t, :, start_y : start_y + box_size, start_x : start_x + box_size] = 0.9

    elif pattern == "sweep":
        for t in range(num_frames):
            ratio = t / max(1, num_frames - 1)
            # Channel 0 sweeps -1.0 to 1.0, Channel 1 sweeps 1.0 to -1.0, Channel 2 is stationary
            frames[t, 0].fill_(ratio * 2.0 - 1.0)
            frames[t, 1].fill_(1.0 - ratio * 2.0)
            frames[t, 2].fill_(0.0)
    else:
        raise ValueError(f"Unknown pattern '{pattern}'. Available: 'bounce', 'sweep'")

    return FrameSequence(frames, fps=fps)
