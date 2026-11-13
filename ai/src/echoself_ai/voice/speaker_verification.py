"""Speaker enrollment and verification using d-vector embeddings.

Implements a lightweight x-vector/ECAPA-inspired architecture in pure PyTorch.
Speaker profiles are stored securely as JSON with base64-encoded tensors.
Raw embeddings are never exposed through any public API.
"""
import base64
import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from echoself_ai.voice.features import LogMelSpectrogram

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Challenge phrases (fixed list of 20)
# ---------------------------------------------------------------------------

_CHALLENGE_PHRASES: list[str] = [
    "The quick brown fox jumps over the lazy dog",
    "She sells seashells by the seashore",
    "How much wood would a woodchuck chuck",
    "Peter Piper picked a peck of pickled peppers",
    "Round the rugged rocks the ragged rascal ran",
    "The rain in Spain stays mainly in the plain",
    "I saw Susie sitting in a shoeshine shop",
    "Fresh French fried fish fill five flat fish dishes",
    "Unique New York unique New York you know you need unique New York",
    "Red lorry yellow lorry red lorry yellow lorry",
    "If two witches watched two watches which witch watched which watch",
    "Whether the weather is warm whether the weather is hot",
    "A big black bear sat on a big black rug",
    "Many mumbling mice are making merry music in the moonlight",
    "Four furious friends fought for the phone",
    "Eleven benevolent elephants balanced on bookshelves",
    "She stood on the balcony inexplicably mimicking him",
    "Fuzzy wuzzy was a bear fuzzy wuzzy had no hair",
    "How can a clam cram in a clean cream can",
    "I scream you scream we all scream for ice cream",
]


def generate_challenge_phrase() -> str:
    """Return a random challenge phrase from the fixed list of 20 phrases."""
    return random.choice(_CHALLENGE_PHRASES)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpeakerEncoderConfig:
    """Configuration for the speaker encoder network."""

    input_dim: int = 80          # Number of mel-filterbank channels
    hidden_dim: int = 128        # Number of conv channels
    embedding_dim: int = 192     # Size of the final d-vector

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {self.embedding_dim}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _Conv1dBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2  # same-length output
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SpeakerEncoder(nn.Module):
    """Lightweight d-vector speaker encoder.

    Architecture:
        3 × Conv1dBlock(BatchNorm + ReLU) → temporal average pooling → 2 × Linear → L2 norm

    Args:
        config: SpeakerEncoderConfig controlling dimensions.

    Input:
        x: Log-mel spectrogram tensor of shape (B, n_mels, T).

    Output:
        L2-normalised d-vector embedding of shape (B, embedding_dim).
    """

    def __init__(self, config: Optional[SpeakerEncoderConfig] = None) -> None:
        super().__init__()
        cfg = config or SpeakerEncoderConfig()

        self.conv_blocks = nn.Sequential(
            _Conv1dBlock(cfg.input_dim, cfg.hidden_dim, kernel_size=5),
            _Conv1dBlock(cfg.hidden_dim, cfg.hidden_dim, kernel_size=5),
            _Conv1dBlock(cfg.hidden_dim, cfg.hidden_dim, kernel_size=5),
        )

        # Two-layer projection head
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.embedding_dim)
        self._embedding_dim = cfg.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised speaker embedding.

        Args:
            x: (B, n_mels, T) log-mel spectrogram.

        Returns:
            (B, embedding_dim) L2-normalised d-vector.
        """
        # (B, n_mels, T) → (B, hidden_dim, T)
        h = self.conv_blocks(x)

        # Temporal average pooling: (B, hidden_dim, T) → (B, hidden_dim)
        h = h.mean(dim=-1)

        # MLP projection
        h = F.relu(self.fc1(h))
        h = self.fc2(h)

        # L2 normalise to unit sphere
        h = F.normalize(h, p=2, dim=-1)
        return h


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass
class SpeakerProfile:
    """Persisted speaker identity record.

    ``mean_embedding`` is stored as a plain Python list of floats so it can
    be JSON-serialised. The raw tensor form is reconstructed on demand by
    :meth:`_to_tensor` (internal helper) and is never exposed through any API.
    """

    user_id: str
    enrollment_id: str
    mean_embedding: list[float]
    enrolled_at: str
    num_samples: int

    # ------------------------------------------------------------------
    # Private helpers — not part of the public API surface
    # ------------------------------------------------------------------

    def _to_tensor(self) -> torch.Tensor:
        """Reconstruct the mean embedding as a 1-D float32 tensor (private)."""
        return torch.tensor(self.mean_embedding, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

_DEFAULT_SPEAKERS_DIR = Path("ai/data/speakers")


def _profile_path(storage_dir: Path, user_id: str) -> Path:
    """Return the JSON file path for a given user_id."""
    # Sanitise user_id to prevent path traversal
    safe_id = user_id.replace("/", "_").replace("\\", "_").replace("..", "_")
    return storage_dir / f"{safe_id}.json"


def _save_profile(profile: SpeakerProfile, storage_dir: Path) -> None:
    """Serialise a SpeakerProfile to JSON with base64-encoded embedding."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = _profile_path(storage_dir, profile.user_id)

    # Encode the embedding tensor as base64 to keep it opaque in storage
    tensor = profile._to_tensor()
    buf = tensor.numpy().tobytes()
    b64 = base64.b64encode(buf).decode("ascii")

    payload = {
        "user_id": profile.user_id,
        "enrollment_id": profile.enrollment_id,
        "mean_embedding_b64": b64,
        "embedding_dim": len(profile.mean_embedding),
        "enrolled_at": profile.enrolled_at,
        "num_samples": profile.num_samples,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_profile_from_file(path: Path) -> SpeakerProfile:
    """Deserialise a SpeakerProfile from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    b64 = data["mean_embedding_b64"]
    buf = base64.b64decode(b64)
    import numpy as np  # local import — already in venv (NumPy ships with PyTorch)
    arr = np.frombuffer(buf, dtype=np.float32).copy()
    mean_embedding: list[float] = arr.tolist()
    return SpeakerProfile(
        user_id=data["user_id"],
        enrollment_id=data["enrollment_id"],
        mean_embedding=mean_embedding,
        enrolled_at=data["enrolled_at"],
        num_samples=data["num_samples"],
    )


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def _extract_embedding(
    encoder: SpeakerEncoder,
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int = 80,
) -> torch.Tensor:
    """Run feature extraction + encoder on a 1-D waveform tensor.

    Returns a 1-D embedding tensor of shape (embedding_dim,).
    """
    # Build feature extractor on the same device as the encoder
    device = next(encoder.parameters()).device
    extractor = LogMelSpectrogram(sample_rate=sample_rate, n_mels=n_mels).to(device)

    wav = waveform.to(device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # (1, T)

    with torch.no_grad():
        mel = extractor(wav)          # (1, n_mels, frames)
        embedding = encoder(mel)      # (1, embedding_dim)

    return embedding.squeeze(0).cpu()  # (embedding_dim,)


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

class SpeakerEnrollment:
    """Enroll speakers and manage their stored profiles.

    Args:
        encoder: A :class:`SpeakerEncoder` instance used to extract embeddings.
        storage_dir: Directory where speaker profiles are persisted.
            Defaults to ``ai/data/speakers`` relative to the working directory.
    """

    def __init__(
        self,
        encoder: Optional[SpeakerEncoder] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.encoder = encoder or SpeakerEncoder()
        self.encoder.eval()
        self.storage_dir = Path(storage_dir) if storage_dir else _DEFAULT_SPEAKERS_DIR

    # ------------------------------------------------------------------

    def enroll(
        self,
        user_id: str,
        waveform_list: list[torch.Tensor],
        sample_rate: int,
        consent_manager=None,
    ) -> SpeakerProfile:
        """Enroll a speaker from one or more waveform samples.

        Extracts an embedding from each waveform, averages them into a single
        mean d-vector, L2-normalises the result, and persists the profile to
        disk.

        Args:
            user_id: Unique identifier for the speaker.
            waveform_list: List of 1-D float32 waveform tensors.
            sample_rate: Sample rate of the waveforms (Hz).
            consent_manager: Optional :class:`VoiceConsentManager`. If
                provided its ``enforce_gate`` is called with scope
                ``"enrollment"`` before any processing.

        Returns:
            The persisted :class:`SpeakerProfile`.
        """
        if not waveform_list:
            raise ValueError("waveform_list must contain at least one waveform")

        # Consent gate
        if consent_manager is not None:
            consent_manager.enforce_gate(user_id, user_id, scope="enrollment")

        embeddings: list[torch.Tensor] = []
        for waveform in waveform_list:
            emb = _extract_embedding(self.encoder, waveform, sample_rate)
            embeddings.append(emb)

        # Stack and compute mean, then re-normalise
        stacked = torch.stack(embeddings, dim=0)       # (N, embedding_dim)
        mean_emb = stacked.mean(dim=0)                 # (embedding_dim,)
        mean_emb = F.normalize(mean_emb, p=2, dim=0)  # keep on unit sphere

        profile = SpeakerProfile(
            user_id=user_id,
            enrollment_id=str(uuid.uuid4()),
            mean_embedding=mean_emb.tolist(),
            enrolled_at=datetime.now(timezone.utc).isoformat(),
            num_samples=len(waveform_list),
        )

        _save_profile(profile, self.storage_dir)
        logger.info("Speaker enrolled: user_id=%s enrollment_id=%s samples=%d",
                    user_id, profile.enrollment_id, profile.num_samples)
        return profile

    # ------------------------------------------------------------------

    def load_profile(self, user_id: str) -> Optional[SpeakerProfile]:
        """Load a stored speaker profile, or return None if not found."""
        path = _profile_path(self.storage_dir, user_id)
        if not path.is_file():
            return None
        try:
            return _load_profile_from_file(path)
        except Exception as exc:
            logger.warning("Failed to load speaker profile for %s: %s", user_id, exc)
            return None

    # ------------------------------------------------------------------

    def revoke(self, user_id: str) -> bool:
        """Delete the stored profile for ``user_id``.

        Returns:
            ``True`` if a profile existed and was deleted, ``False`` otherwise.
        """
        path = _profile_path(self.storage_dir, user_id)
        if path.is_file():
            path.unlink()
            logger.info("Speaker profile revoked: user_id=%s", user_id)
            return True
        return False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class SpeakerVerifier:
    """Verify a speaker against their enrolled profile using cosine similarity.

    Args:
        encoder: A :class:`SpeakerEncoder` instance (should be the same
            architecture used during enrollment).
        storage_dir: Directory where speaker profiles are stored.
    """

    def __init__(
        self,
        encoder: Optional[SpeakerEncoder] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.encoder = encoder or SpeakerEncoder()
        self.encoder.eval()
        self._enrollment = SpeakerEnrollment(
            encoder=self.encoder,
            storage_dir=storage_dir,
        )

    # ------------------------------------------------------------------

    def verify(
        self,
        user_id: str,
        waveform: torch.Tensor,
        sample_rate: int,
        threshold: float = 0.75,
    ) -> tuple[bool, float]:
        """Verify whether ``waveform`` belongs to the enrolled speaker.

        Args:
            user_id: The claimed speaker identity to verify against.
            waveform: 1-D float32 waveform tensor.
            sample_rate: Sample rate of the waveform (Hz).
            threshold: Cosine similarity acceptance threshold in [0, 1].

        Returns:
            ``(is_accepted, similarity)`` where ``is_accepted`` is ``True``
            when the cosine similarity meets or exceeds ``threshold``.
            Returns ``(False, 0.0)`` if no profile exists for ``user_id``.
        """
        profile = self._enrollment.load_profile(user_id)
        if profile is None:
            logger.info("Verification failed (no profile): user_id=%s", user_id)
            return False, 0.0

        # Enrolled mean embedding (unit-norm by construction)
        enrolled = profile._to_tensor()           # (embedding_dim,)
        enrolled = F.normalize(enrolled, p=2, dim=0)

        # Query embedding
        query = _extract_embedding(self.encoder, waveform, sample_rate)  # (embedding_dim,)
        query = F.normalize(query, p=2, dim=0)

        # Cosine similarity: dot product of two unit vectors
        similarity: float = torch.dot(query, enrolled).item()
        # Clamp to [-1, 1] for numerical robustness
        similarity = max(-1.0, min(1.0, similarity))

        is_accepted = similarity >= threshold
        logger.info(
            "Verification result: user_id=%s similarity=%.4f threshold=%.4f accepted=%s",
            user_id, similarity, threshold, is_accepted,
        )
        return is_accepted, float(similarity)
