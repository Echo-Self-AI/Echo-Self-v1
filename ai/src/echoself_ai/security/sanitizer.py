"""Input validation, safety filtering, and prompt injection defense."""
from dataclasses import dataclass, field
import re
from typing import Any, Optional


@dataclass
class InputSanitizerConfig:
    """Security bounds and validation rules for AI inputs."""

    max_prompt_length: int = 1000
    max_new_tokens_limit: int = 256
    max_video_frames_limit: int = 32
    max_fps_limit: int = 30
    strip_control_chars: bool = True
    block_prompt_injection: bool = True


@dataclass
class SanitizationResult:
    """Result of input safety evaluation."""

    sanitized_text: str
    is_safe: bool
    flags: list[str] = field(default_factory=list)


# Known prompt injection and system override patterns
INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?previous\s+instructions\b",
    r"(?i)\bdisregard\s+(all\s+)?prior\s+prompts?\b",
    r"(?i)\boverride\s+system\s+rules?\b",
    r"(?i)\byou\s+are\s+now\s+(DAN|jailbroken|unfiltered)\b",
    r"(?i)\bbypass\s+(all\s+)?safeguards?\b",
    r"(?i)\bleak\s+(the\s+)?(system|internal)\s+prompt\b",
    r"(?i)\bforget\s+(all\s+)?constraints?\b",
]

COMPILED_INJECTION_RE = [re.compile(p) for p in INJECTION_PATTERNS]


class InputSanitizer:
    """Sanitizes text, payloads, and parameter bounds for safe local AI execution."""

    def __init__(self, config: Optional[InputSanitizerConfig] = None) -> None:
        self.config = config or InputSanitizerConfig()

    def sanitize_text(self, text: str) -> SanitizationResult:
        """Sanitize an arbitrary input string, stripping control characters and checking bounds."""
        flags: list[str] = []

        if not isinstance(text, str):
            return SanitizationResult(sanitized_text="", is_safe=True, flags=["non_string_coerced"])

        # 1. Strip control characters (except newline, tab, carriage return)
        if self.config.strip_control_chars:
            cleaned = "".join(c for c in text if c in ("\n", "\r", "\t") or (ord(c) >= 32 and ord(c) != 127))
            if len(cleaned) != len(text):
                flags.append("control_characters_stripped")
        else:
            cleaned = text

        # 2. Length boundary enforcement
        if len(cleaned) > self.config.max_prompt_length:
            cleaned = cleaned[: self.config.max_prompt_length]
            flags.append("length_truncated")

        # 3. Prompt injection detection
        is_safe = True
        if self.config.block_prompt_injection:
            for pattern in COMPILED_INJECTION_RE:
                if pattern.search(cleaned):
                    is_safe = False
                    flags.append("prompt_injection_pattern_detected")
                    break

        return SanitizationResult(
            sanitized_text=cleaned,
            is_safe=is_safe,
            flags=flags,
        )

    def sanitize_payload(self, job_type: str, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Validate and clamp numeric bounds and string fields across job payloads."""
        cleaned = dict(payload)
        all_flags: list[str] = []

        # Sanitize text prompts
        for text_key in ("prompt", "text"):
            if text_key in cleaned and isinstance(cleaned[text_key], str):
                res = self.sanitize_text(cleaned[text_key])
                cleaned[text_key] = res.sanitized_text
                all_flags.extend(res.flags)

        # Enforce token generation limits
        if "max_new_tokens" in cleaned:
            try:
                tokens = int(float(cleaned["max_new_tokens"]))
                if tokens > self.config.max_new_tokens_limit:
                    tokens = self.config.max_new_tokens_limit
                    all_flags.append("max_new_tokens_clamped")
                elif tokens < 1:
                    tokens = 1
                    all_flags.append("min_new_tokens_clamped")
                cleaned["max_new_tokens"] = tokens
            except (ValueError, TypeError):
                cleaned["max_new_tokens"] = 30

        # Enforce temperature limits
        if "temperature" in cleaned:
            try:
                temp = float(cleaned["temperature"])
                if temp < 0.01:
                    temp = 0.01
                    all_flags.append("min_temperature_clamped")
                elif temp > 2.0:
                    temp = 2.0
                    all_flags.append("max_temperature_clamped")
                cleaned["temperature"] = temp
            except (ValueError, TypeError):
                cleaned["temperature"] = 0.8

        # Enforce top-k limits
        if "top_k" in cleaned:
            try:
                k = int(float(cleaned["top_k"]))
                if k < 1:
                    k = 1
                elif k > 100:
                    k = 100
                cleaned["top_k"] = k
            except (ValueError, TypeError):
                cleaned["top_k"] = 10

        # Enforce video frame count and FPS
        if "num_frames" in cleaned:
            try:
                frames = int(float(cleaned["num_frames"]))
                if frames > self.config.max_video_frames_limit:
                    frames = self.config.max_video_frames_limit
                    all_flags.append("max_frames_clamped")
                elif frames < 1:
                    frames = 1
                cleaned["num_frames"] = frames
            except (ValueError, TypeError):
                cleaned["num_frames"] = 8

        if "fps" in cleaned:
            try:
                fps = int(float(cleaned["fps"]))
                if fps > self.config.max_fps_limit:
                    fps = self.config.max_fps_limit
                    all_flags.append("max_fps_clamped")
                elif fps < 1:
                    fps = 1
                cleaned["fps"] = fps
            except (ValueError, TypeError):
                cleaned["fps"] = 12

        return cleaned, all_flags
