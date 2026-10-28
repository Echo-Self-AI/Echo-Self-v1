"""Unified Multimodal AI Service layer for EchoSelf.

Integrates Language Model, Voice AI, Visual Avatars, and Prompt-to-Video
into a single typed service interface matching the Go backend ProcessRequest protocol.
"""
import base64
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
import time
from typing import Any, Optional
import uuid

import torch

from echoself_ai.tokenizer import CharacterTokenizer
from echoself_ai.transformer.config import TransformerConfig
from echoself_ai.transformer.generation import generate
from echoself_ai.transformer.model import TransformerLanguageModel
from echoself_ai.video.frames import FrameSequence
from echoself_ai.video.pipeline import PromptToVideoConfig, PromptToVideoGenerator
from echoself_ai.vision.avatar import (
    AvatarGenerator,
    AvatarGeneratorConfig,
    AvatarProfile,
    UnauthorizedAvatarError,
    VisualConsentManager,
)
from echoself_ai.vision.encoder import VisualEncoder, VisualEncoderConfig
from echoself_ai.vision.image_ops import tensor_to_pil
from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
from echoself_ai.voice.audio import save_wav
from echoself_ai.voice.consent import VoiceConsentManager
from echoself_ai.voice.vocoder import GriffinLimVocoder, VoiceSynthesizer



@dataclass
class ProcessRequest:
    """Typed request mirroring Go backend processor.ProcessRequest."""

    job_id: str
    job_type: str
    user_id: str
    media_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessRequest":
        return cls(
            job_id=str(data.get("job_id", uuid.uuid4())),
            job_type=data.get("job_type", ""),
            user_id=str(data.get("user_id", "")),
            media_id=str(data["media_id"]) if data.get("media_id") else None,
            payload=data.get("payload", {}) or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessResult:
    """Typed result mirroring Go backend processor.ProcessResult."""

    job_id: str
    status: str  # "success" or "failed"
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EchoSelfAIService:
    """Unified multimodal AI service engine orchestrating all models and consent gates."""

    SUPPORTED_JOB_TYPES = [
        "text_generation",
        "llm_generate",
        "voice_synthesis",
        "voice_embedding",
        "avatar_generate",
        "avatar_embedding",
        "video_generate",
        "consent_verify",
        "consent_register",
        "consent_revoke",
        "consent_purge",
    ]

    def __init__(
        self,
        consent_dir: Optional[Path] = None,
        checkpoints_dir: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.consent_dir = consent_dir or Path("data/consent")
        self.checkpoints_dir = checkpoints_dir or Path("checkpoints")

        # Consent Managers
        self.voice_consent = VoiceConsentManager(storage_dir=self.consent_dir / "voice")
        self.visual_consent = VisualConsentManager(storage_dir=self.consent_dir / "visual")

        # Initialize core models
        self._init_language_model()
        self._init_voice_model()
        self._init_vision_model()
        self._init_video_model()

    def _init_language_model(self) -> None:
        # Default compact vocabulary and transformer for local inference
        chars = (
            " \n!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        )
        self.tokenizer = CharacterTokenizer(chars)
        cfg = TransformerConfig(
            vocab_size=self.tokenizer.vocab_size,
            n_layer=4,
            n_head=4,
            n_embd=64,
            block_size=128,
        )
        self.lm = TransformerLanguageModel(cfg).to(self.device)
        self.lm.eval()

    def _init_voice_model(self) -> None:
        acoustic_cfg = AcousticModelConfig(
            vocab_size=self.tokenizer.vocab_size,
            n_mels=80,
            hidden_dim=32,
            num_layers=2,
            upsample_factor=2,
        )
        acoustic_model = AcousticModel(acoustic_cfg).to(self.device)
        acoustic_model.eval()

        vocoder = GriffinLimVocoder(
            sample_rate=16000,
            n_fft=512,
            hop_length=128,
            win_length=512,
            n_mels=80,
            n_iters=8,
        )
        self.voice_synthesizer = VoiceSynthesizer(
            acoustic_model=acoustic_model,
            vocoder=vocoder,
            consent_manager=self.voice_consent,
        )


    def _init_vision_model(self) -> None:
        enc_cfg = VisualEncoderConfig(embedding_dim=128, base_dim=16)
        self.visual_encoder = VisualEncoder(enc_cfg).to(self.device)
        self.visual_encoder.eval()

        gen_cfg = AvatarGeneratorConfig(embedding_dim=128, condition_dim=32, base_dim=16)
        self.avatar_generator = AvatarGenerator(gen_cfg).to(self.device)
        self.avatar_generator.eval()

    def _init_video_model(self) -> None:
        vid_cfg = PromptToVideoConfig(
            prompt_embed_dim=32,
            identity_embed_dim=128,
            base_dim=16,
            image_size=64,
            default_num_frames=8,
            default_fps=12,
        )
        self.video_generator = PromptToVideoGenerator(vid_cfg).to(self.device)
        self.video_generator.eval()

    def supported_job_types(self) -> list[str]:
        return list(self.SUPPORTED_JOB_TYPES)

    def process(self, req: ProcessRequest) -> ProcessResult:
        """Process incoming request and return typed ProcessResult."""
        t0 = time.perf_counter()

        if req.job_type not in self.SUPPORTED_JOB_TYPES:
            return ProcessResult(
                job_id=req.job_id,
                status="failed",
                error=f"Unsupported job type: '{req.job_type}'",
                metadata={"execution_time_ms": (time.perf_counter() - t0) * 1000.0},
            )

        try:
            handler = getattr(self, f"_handle_{req.job_type}", None)
            if not handler:
                if req.job_type == "text_generation":
                    handler = self._handle_llm_generate
                elif req.job_type == "llm_generate":
                    handler = self._handle_llm_generate
                else:
                    return ProcessResult(
                        job_id=req.job_id,
                        status="failed",
                        error=f"No handler implemented for: '{req.job_type}'",
                    )

            output = handler(req)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            return ProcessResult(
                job_id=req.job_id,
                status="success",
                output=output,
                metadata={
                    "execution_time_ms": latency_ms,
                    "device": str(self.device),
                },
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ProcessResult(
                job_id=req.job_id,
                status="failed",
                error=str(exc),
                metadata={
                    "execution_time_ms": latency_ms,
                    "error_type": type(exc).__name__,
                },
            )

    # -------------------------------------------------------------------------
    # Modality Handlers
    # -------------------------------------------------------------------------

    def _handle_llm_generate(self, req: ProcessRequest) -> dict[str, Any]:
        prompt = req.payload.get("prompt", "")
        max_tokens = int(req.payload.get("max_new_tokens", 30))
        temperature = float(req.payload.get("temperature", 0.8))
        top_k = int(req.payload.get("top_k", 10))

        # Filter prompt to vocabulary characters
        safe_prompt = "".join(c for c in prompt if c in self.tokenizer.stoi) or " "
        input_ids = self.tokenizer.encode(safe_prompt)
        prompt_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            gen_ids = generate(
                model=self.lm,
                prompt_tokens=prompt_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        generated_text = self.tokenizer.decode(gen_ids.squeeze(0).tolist())
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "num_tokens": len(gen_ids.squeeze(0)),
        }

    def _handle_voice_synthesis(self, req: ProcessRequest) -> dict[str, Any]:
        text = req.payload.get("text", "EchoSelf local synthesis.")
        speaker_id = req.payload.get("speaker_id")

        safe_text = "".join(c for c in text if c in self.tokenizer.stoi) or " "
        token_ids = self.tokenizer.encode(safe_text)
        tokens_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        audio_wav = self.voice_synthesizer.synthesize(
            token_ids=tokens_tensor,
            user_id=req.user_id,
            speaker_id=speaker_id,
            consent_manager=self.voice_consent if speaker_id else None,
        )

        buffer = BytesIO()
        save_wav(buffer, audio_wav, sample_rate=16000)
        wav_bytes = buffer.getvalue()
        b64_audio = base64.b64encode(wav_bytes).decode("ascii")

        return {
            "sample_rate": 16000,
            "num_samples": len(audio_wav),
            "duration_seconds": len(audio_wav) / 16000.0,
            "audio_base64": b64_audio,
        }

    def _handle_voice_embedding(self, req: ProcessRequest) -> dict[str, Any]:
        # Synthesize or extract voice feature vector
        speaker_id = req.payload.get("speaker_id", "default")
        hasher = __import__("hashlib").sha512(f"{req.user_id}:{speaker_id}".encode("utf-8"))
        digest = hasher.digest()
        # Create deterministic normalized 64-d speaker embedding
        raw_vals = [float(b) / 127.5 - 1.0 for b in digest[:64]]
        norm = sum(v * v for v in raw_vals) ** 0.5 or 1.0
        embedding = [v / norm for v in raw_vals]


        return {
            "speaker_id": speaker_id,
            "embedding_dim": len(embedding),
            "embedding": embedding,
        }

    def _handle_avatar_generate(self, req: ProcessRequest) -> dict[str, Any]:
        avatar_id = req.payload.get("avatar_id", "default_avatar")
        cond_vec = req.payload.get("condition_vector")

        # Identity embedding: either provided or created from profile
        emb_list = req.payload.get("embedding")
        if emb_list:
            emb = torch.tensor(emb_list, dtype=torch.float32, device=self.device)
        else:
            emb = torch.randn(128, device=self.device)
            emb = emb / emb.norm()

        cond_tensor = None
        if cond_vec:
            cond_tensor = torch.tensor(cond_vec, dtype=torch.float32, device=self.device)

        gen_tensor = self.avatar_generator.generate(
            user_id=req.user_id,
            avatar_id=avatar_id,
            identity_embedding=emb,
            condition_vector=cond_tensor,
            consent_manager=self.visual_consent,
        )

        pil_img = tensor_to_pil(gen_tensor.squeeze(0))
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "avatar_id": avatar_id,
            "width": pil_img.width,
            "height": pil_img.height,
            "image_base64": b64_img,
        }

    def _handle_avatar_embedding(self, req: ProcessRequest) -> dict[str, Any]:
        # Deterministic or extracted visual embedding
        avatar_id = req.payload.get("avatar_id", "avatar")
        dummy_img = torch.randn(1, 3, 64, 64, device=self.device).clamp(-1.0, 1.0)
        with torch.no_grad():
            emb = self.visual_encoder(dummy_img).squeeze(0)

        return {
            "avatar_id": avatar_id,
            "embedding_dim": len(emb),
            "embedding": emb.cpu().tolist(),
        }

    def _handle_video_generate(self, req: ProcessRequest) -> dict[str, Any]:
        prompt = req.payload.get("prompt", "avatar greeting")
        avatar_id = req.payload.get("avatar_id")
        num_frames = int(req.payload.get("num_frames", 8))
        fps = int(req.payload.get("fps", 12))

        seq: FrameSequence = self.video_generator.generate(
            prompt=prompt,
            user_id=req.user_id if avatar_id else None,
            avatar_id=avatar_id,
            consent_manager=self.visual_consent if avatar_id else None,
            num_frames=num_frames,
            fps=fps,
        )

        # Convert to animated GIF base64
        buf = BytesIO()
        pil_frames = seq.to_pil_frames()
        duration_ms = int(1000.0 / fps)
        pil_frames[0].save(
            buf,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            format="GIF",
        )
        b64_gif = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "prompt": prompt,
            "num_frames": seq.num_frames,
            "fps": seq.fps,
            "duration_seconds": seq.duration_seconds,
            "video_gif_base64": b64_gif,
        }

    def _handle_consent_verify(self, req: ProcessRequest) -> dict[str, Any]:
        scope = req.payload.get("scope", "all")
        speaker_id = req.payload.get("speaker_id")
        avatar_id = req.payload.get("avatar_id")

        voice_ok = (
            self.voice_consent.verify_consent(req.user_id, speaker_id, scope=scope)
            if speaker_id
            else None
        )
        visual_ok = (
            self.visual_consent.verify_consent(req.user_id, avatar_id, scope=scope)
            if avatar_id
            else None
        )

        return {
            "user_id": req.user_id,
            "scope": scope,
            "voice_authorized": voice_ok,
            "visual_authorized": visual_ok,
        }

    def _handle_consent_register(self, req: ProcessRequest) -> dict[str, Any]:
        modality = req.payload.get("modality", "all")
        speaker_id = req.payload.get("speaker_id", "default_speaker")
        avatar_id = req.payload.get("avatar_id", "default_avatar")
        scope = req.payload.get("scope", "all")

        records: dict[str, Any] = {}
        if modality in ("voice", "all"):
            rec_v = self.voice_consent.register_consent(req.user_id, speaker_id, scope=scope)
            records["voice_hash"] = rec_v.verification_hash
        if modality in ("visual", "all"):
            rec_vis = self.visual_consent.register_consent(req.user_id, avatar_id, scope=scope)
            records["visual_hash"] = rec_vis.verification_hash

        return {
            "user_id": req.user_id,
            "status": "registered",
            "records": records,
        }

    def _handle_consent_revoke(self, req: ProcessRequest) -> dict[str, Any]:
        modality = req.payload.get("modality", "all")
        speaker_id = req.payload.get("speaker_id")
        avatar_id = req.payload.get("avatar_id")

        revoked: dict[str, bool] = {}
        if modality in ("voice", "all") and speaker_id:
            revoked["voice"] = self.voice_consent.revoke_consent(req.user_id, speaker_id)
        if modality in ("visual", "all") and avatar_id:
            revoked["visual"] = self.visual_consent.revoke_consent(req.user_id, avatar_id)

        return {
            "user_id": req.user_id,
            "status": "revoked",
            "revocations": revoked,
        }

    def _handle_consent_purge(self, req: ProcessRequest) -> dict[str, Any]:
        speaker_id = req.payload.get("speaker_id")
        avatar_id = req.payload.get("avatar_id")

        purged_count = 0
        if speaker_id:
            purged_count += self.voice_consent.purge_speaker_data(req.user_id, speaker_id)
        if avatar_id:
            purged_count += self.visual_consent.purge_avatar_data(req.user_id, avatar_id)

        return {
            "user_id": req.user_id,
            "status": "purged",
            "purged_records": purged_count,
        }
