"""Visual AI and avatar subsystem for EchoSelf."""
from echoself_ai.vision.avatar import (
    AvatarGenerator,
    AvatarGeneratorConfig,
    AvatarProfile,
    UnauthorizedAvatarError,
    VisualConsentManager,
    VisualConsentRecord,
    compute_identity_consistency_loss,
)
from echoself_ai.vision.encoder import VisualEncoder, VisualEncoderConfig
from echoself_ai.vision.image_ops import (
    ImageConfig,
    generate_synthetic_image,
    load_image,
    save_image,
    tensor_to_pil,
)

__all__ = [
    "AvatarGenerator",
    "AvatarGeneratorConfig",
    "AvatarProfile",
    "ImageConfig",
    "UnauthorizedAvatarError",
    "VisualConsentManager",
    "VisualConsentRecord",
    "VisualEncoder",
    "VisualEncoderConfig",
    "compute_identity_consistency_loss",
    "generate_synthetic_image",
    "load_image",
    "save_image",
    "tensor_to_pil",
]

