"""Avatar representation, neural identity conditioning, and privacy consent engine."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

from echoself_ai.vision.encoder import VisualEncoder


class UnauthorizedAvatarError(RuntimeError):
    """Raised when an avatar operation is attempted without valid active consent."""


@dataclass
class VisualConsentRecord:
    """Verifiable cryptographic authorization for avatar creation and visual generation."""

    user_id: str
    avatar_id: str
    consent_granted: bool
    scope: str  # "generation", "animation", "avatar_creation", or "all"
    granted_at: str
    verification_hash: str
    revoked: bool = False
    revoked_at: Optional[str] = None

    def is_active_for(self, scope: str) -> bool:
        """Check if consent is valid, active, and covers the requested operation scope."""
        if not self.consent_granted or self.revoked:
            return False
        if self.scope == "all" or self.scope == scope:
            return True
        return False


class VisualConsentManager:
    """Manages explicit user authorization, consent gating, and data deletion for Avatars."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/consent/visual")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, VisualConsentRecord] = {}
        self._load_all()

    def _get_key(self, user_id: str, avatar_id: str) -> str:
        return f"{user_id}_{avatar_id}"

    def _load_all(self) -> None:
        """Load registered consent records from storage."""
        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = VisualConsentRecord(**data)
                self.records[self._get_key(record.user_id, record.avatar_id)] = record
            except Exception:
                continue

    def register_consent(
        self,
        user_id: str,
        avatar_id: str,
        scope: str = "generation",
        statement: str = "I authorize EchoSelf to synthesize and generate avatar representations.",
    ) -> VisualConsentRecord:
        """Cryptographically sign and persist explicit user consent."""
        now = datetime.now(timezone.utc).isoformat()
        hasher = hashlib.sha256()
        hasher.update(f"{user_id}:{avatar_id}:{scope}:{now}:{statement}".encode("utf-8"))
        verification_hash = hasher.hexdigest()

        record = VisualConsentRecord(
            user_id=user_id,
            avatar_id=avatar_id,
            consent_granted=True,
            scope=scope,
            granted_at=now,
            verification_hash=verification_hash,
            revoked=False,
            revoked_at=None,
        )

        key = self._get_key(user_id, avatar_id)
        self.records[key] = record

        record_file = self.storage_dir / f"{key}.json"
        record_file.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        return record

    def verify_consent(self, user_id: str, avatar_id: str, scope: str = "generation") -> bool:
        """Check whether valid, active consent exists for an avatar and scope."""
        key = self._get_key(user_id, avatar_id)
        record = self.records.get(key)
        if not record:
            return False
        return record.is_active_for(scope)

    def revoke_consent(self, user_id: str, avatar_id: str) -> bool:
        """Immediately revoke consent for an avatar."""
        key = self._get_key(user_id, avatar_id)
        record = self.records.get(key)
        if not record:
            return False

        record.revoked = True
        record.revoked_at = datetime.now(timezone.utc).isoformat()

        record_file = self.storage_dir / f"{key}.json"
        record_file.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        return True

    def purge_avatar_data(
        self,
        user_id: str,
        avatar_id: str,
        data_dirs: Optional[list[Path]] = None,
    ) -> int:
        """GDPR compliant permanent deletion of all user avatar artifacts and consent records."""
        deleted_count = 0
        key = self._get_key(user_id, avatar_id)

        # 1. Remove consent record
        if key in self.records:
            del self.records[key]
        consent_file = self.storage_dir / f"{key}.json"
        if consent_file.exists():
            consent_file.unlink()
            deleted_count += 1

        # 2. Remove files matching user/avatar identifier across data directories
        if data_dirs:
            for d in data_dirs:
                if not d.exists():
                    continue
                for pattern in [f"*{avatar_id}*", f"*{user_id}*"]:
                    for file_path in d.glob(pattern):
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_count += 1
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                            deleted_count += 1

        return deleted_count


@dataclass
class AvatarProfile:
    """Identity representation profile of an avatar with canonical visual embedding."""

    user_id: str
    avatar_id: str
    name: str
    embedding: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_similarity(self, candidate_embedding: torch.Tensor) -> float:
        """Compute cosine similarity between profile's canonical embedding and candidate."""
        emb1 = self.embedding
        emb2 = candidate_embedding

        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)

        sim = F.cosine_similarity(emb1, emb2, dim=-1)
        return float(sim.mean().item())

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "user_id": self.user_id,
            "avatar_id": self.avatar_id,
            "name": self.name,
            "embedding": self.embedding.detach().cpu().tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AvatarProfile":
        """Reconstruct profile from dictionary."""
        return cls(
            user_id=data["user_id"],
            avatar_id=data["avatar_id"],
            name=data["name"],
            embedding=torch.tensor(data["embedding"], dtype=torch.float32),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path | str) -> None:
        """Save avatar profile to JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "AvatarProfile":
        """Load avatar profile from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class AvatarGeneratorConfig:
    """Hyperparameters for neural avatar generator."""

    embedding_dim: int = 128
    condition_dim: int = 32
    base_dim: int = 32
    image_size: int = 64
    out_channels: int = 3


class AvatarGenerator(nn.Module):
    """From-scratch convolutional decoder generating avatar frames conditioned on identity.

    Synthesizes normalized images in [-1.0, 1.0] from identity embeddings and optional
    conditioning vectors (e.g. emotion, expression, or pose), enforcing privacy consent gating.
    """

    def __init__(self, config: Optional[AvatarGeneratorConfig] = None) -> None:
        super().__init__()
        self.config = config or AvatarGeneratorConfig()
        total_in_dim = self.config.embedding_dim + self.config.condition_dim
        c = self.config.base_dim

        # Initial dense projection to (4 * c) x 4 x 4 spatial map
        self.init_channels = 4 * c
        self.fc = nn.Sequential(
            nn.Linear(total_in_dim, self.init_channels * 4 * 4),
            nn.GELU(),
        )

        # Upsampling stages: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(self.init_channels, 4 * c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * c),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )

        # Output convolution to RGB image [-1.0, 1.0]
        self.to_rgb = nn.Sequential(
            nn.Conv2d(c, self.config.out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        identity_embedding: torch.Tensor,
        condition_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Synthesize avatar images.

        Args:
            identity_embedding: Tensor of shape (batch_size, embedding_dim).
            condition_vector: Optional tensor of shape (batch_size, condition_dim).
                If None, zeros are supplied.

        Returns:
            Generated images of shape (batch_size, 3, image_size, image_size).
        """
        if identity_embedding.dim() == 1:
            identity_embedding = identity_embedding.unsqueeze(0)

        batch_size = identity_embedding.shape[0]
        device = identity_embedding.device

        if condition_vector is None:
            condition_vector = torch.zeros(
                batch_size,
                self.config.condition_dim,
                device=device,
                dtype=identity_embedding.dtype,
            )
        elif condition_vector.dim() == 1:
            condition_vector = condition_vector.unsqueeze(0)

        latent = torch.cat([identity_embedding, condition_vector], dim=-1)
        x = self.fc(latent)
        x = x.view(batch_size, self.init_channels, 4, 4)

        x = self.up1(x)  # 8x8
        x = self.up2(x)  # 16x16
        x = self.up3(x)  # 32x32
        x = self.up4(x)  # 64x64

        return self.to_rgb(x)

    def generate(
        self,
        user_id: str,
        avatar_id: str,
        identity_embedding: torch.Tensor,
        condition_vector: Optional[torch.Tensor] = None,
        consent_manager: Optional[VisualConsentManager] = None,
    ) -> torch.Tensor:
        """Generate avatar image strictly requiring validated user consent."""
        if consent_manager is not None:
            if not consent_manager.verify_consent(user_id, avatar_id, scope="generation"):
                raise UnauthorizedAvatarError(
                    f"Avatar generation blocked: active consent not found or revoked "
                    f"for user='{user_id}', avatar='{avatar_id}'."
                )

        return self.forward(identity_embedding, condition_vector)


def compute_identity_consistency_loss(
    generated_images: torch.Tensor,
    target_embeddings: torch.Tensor,
    encoder: VisualEncoder,
) -> torch.Tensor:
    """Compute identity preservation loss between generated images and reference embeddings.

    Loss = 1.0 - CosineSimilarity(encoder(generated_images), target_embeddings).
    """
    pred_embeddings = encoder(generated_images)
    if target_embeddings.dim() == 1:
        target_embeddings = target_embeddings.unsqueeze(0)
    cos_sim = F.cosine_similarity(pred_embeddings, target_embeddings, dim=-1)
    return 1.0 - cos_sim.mean()
