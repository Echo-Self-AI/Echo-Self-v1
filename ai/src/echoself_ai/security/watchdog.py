"""Resource governance, memory headroom monitoring, and rate limiting for EchoSelf AI."""
from collections import defaultdict, deque
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional

import torch


@dataclass
class WatchdogConfig:
    """Configuration thresholds for resource management and rate limiting."""

    min_free_ram_mb: float = 128.0
    max_ram_usage_pct: float = 96.0
    rate_limit_requests_per_window: int = 120
    rate_limit_window_seconds: float = 60.0
    auto_gc_threshold_pct: float = 85.0


class RateLimiter:
    """Thread-safe sliding window rate limiter per user identifier."""

    def __init__(self, max_requests: int = 120, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check_rate_limit(self, user_id: str) -> tuple[bool, float]:
        """Check if request is permitted under sliding window.
        
        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            q = self._history[user_id]
            # Prune timestamps older than window
            while q and q[0] < cutoff:
                q.popleft()

            if len(q) >= self.max_requests:
                oldest = q[0]
                retry_after = max(0.1, round((oldest + self.window_seconds) - now, 2))
                return False, retry_after

            q.append(now)
            return True, 0.0

    def reset(self, user_id: Optional[str] = None) -> None:
        """Reset rate limit history for a specific user or all users."""
        with self._lock:
            if user_id:
                self._history.pop(user_id, None)
            else:
                self._history.clear()


class ResourceWatchdog:
    """Monitors system memory headroom and triggers preventive cache cleanups."""

    def __init__(self, config: Optional[WatchdogConfig] = None) -> None:
        self.config = config or WatchdogConfig()

    def get_memory_stats(self) -> dict[str, Any]:
        """Read system memory stats without external third-party dependencies."""
        stats = {
            "mem_total_mb": 0.0,
            "mem_available_mb": 0.0,
            "mem_used_pct": 0.0,
            "gpu_available": torch.cuda.is_available(),
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
        }

        # Read /proc/meminfo on Linux
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.is_file():
            try:
                content = meminfo_path.read_text(encoding="utf-8")
                lookup: dict[str, int] = {}
                for line in content.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        parts = v.strip().split()
                        if parts and parts[0].isdigit():
                            lookup[k.strip()] = int(parts[0])  # in kB

                if "MemTotal" in lookup and "MemAvailable" in lookup:
                    total_mb = lookup["MemTotal"] / 1024.0
                    avail_mb = lookup["MemAvailable"] / 1024.0
                    used_pct = ((total_mb - avail_mb) / total_mb) * 100.0 if total_mb > 0 else 0.0
                    stats["mem_total_mb"] = round(total_mb, 2)
                    stats["mem_available_mb"] = round(avail_mb, 2)
                    stats["mem_used_pct"] = round(used_pct, 2)
            except Exception:
                pass

        # Check PyTorch CUDA VRAM if available
        if stats["gpu_available"]:
            try:
                stats["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024.0 * 1024.0), 2)
                stats["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024.0 * 1024.0), 2)
            except Exception:
                pass

        return stats

    def cleanup_cache(self) -> None:
        """Perform garbage collection and release cached PyTorch CUDA tensors."""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def check_memory_headroom(self) -> tuple[bool, str]:
        """Verify whether the system has sufficient memory headroom to process requests.
        
        Returns:
            (has_headroom, reason_message)
        """
        stats = self.get_memory_stats()
        avail_mb = stats.get("mem_available_mb", 0.0)
        used_pct = stats.get("mem_used_pct", 0.0)

        # Trigger preemptive cleanup if memory usage is elevated
        if used_pct >= self.config.auto_gc_threshold_pct:
            self.cleanup_cache()
            # Recheck after cleanup
            stats = self.get_memory_stats()
            avail_mb = stats.get("mem_available_mb", 0.0)
            used_pct = stats.get("mem_used_pct", 0.0)

        # Enforce bounds if /proc/meminfo was successfully read
        if stats.get("mem_total_mb", 0.0) > 0:
            if avail_mb < self.config.min_free_ram_mb:
                return (
                    False,
                    f"Insufficient memory headroom: {avail_mb}MB available, required minimum {self.config.min_free_ram_mb}MB",
                )
            if used_pct > self.config.max_ram_usage_pct:
                return (
                    False,
                    f"Memory usage critical: {used_pct}% utilized, exceeds safety limit {self.config.max_ram_usage_pct}%",
                )

        return True, "Headroom OK"
