"""Unit tests for ResourceWatchdog and RateLimiter."""
import pytest

from echoself_ai.security.watchdog import (
    RateLimiter,
    ResourceWatchdog,
    WatchdogConfig,
)


def test_rate_limiter_permits_and_blocks():
    limiter = RateLimiter(max_requests=3, window_seconds=10.0)
    user = "test-user-limit"

    # First 3 requests permitted
    ok, _ = limiter.check_rate_limit(user)
    assert ok is True
    ok, _ = limiter.check_rate_limit(user)
    assert ok is True
    ok, _ = limiter.check_rate_limit(user)
    assert ok is True

    # 4th request blocked
    ok, retry_after = limiter.check_rate_limit(user)
    assert ok is False
    assert retry_after > 0.0

    # Other user is unaffected
    other_ok, _ = limiter.check_rate_limit("other-user")
    assert other_ok is True

    # Reset allows user again
    limiter.reset(user)
    ok, _ = limiter.check_rate_limit(user)
    assert ok is True


def test_resource_watchdog_memory_stats():
    watchdog = ResourceWatchdog()
    stats = watchdog.get_memory_stats()

    assert "mem_total_mb" in stats
    assert "mem_available_mb" in stats
    assert "mem_used_pct" in stats
    assert "gpu_available" in stats

    # In Linux test environment, total and available memory should be positive
    if stats["mem_total_mb"] > 0:
        assert stats["mem_total_mb"] > 100
        assert stats["mem_available_mb"] > 0
        assert 0.0 <= stats["mem_used_pct"] <= 100.0


def test_resource_watchdog_headroom_enforcement():
    # Normal config should pass headroom check
    watchdog = ResourceWatchdog(WatchdogConfig(min_free_ram_mb=10.0, max_ram_usage_pct=99.9))
    ok, msg = watchdog.check_memory_headroom()
    assert ok is True
    assert msg == "Headroom OK"

    # Unrealistic high requirement should fail safely
    strict_watchdog = ResourceWatchdog(WatchdogConfig(min_free_ram_mb=999999999.0))
    stats = strict_watchdog.get_memory_stats()
    if stats["mem_total_mb"] > 0:
        ok, reason = strict_watchdog.check_memory_headroom()
        assert ok is False
        assert "Insufficient memory headroom" in reason


def test_resource_watchdog_cleanup_cache():
    watchdog = ResourceWatchdog()
    # Ensure cache cleanup runs cleanly without exception
    watchdog.cleanup_cache()


def test_service_enforces_rate_limiting():
    from echoself_ai.service.unified import EchoSelfAIService, ProcessRequest

    service = EchoSelfAIService()
    service.rate_limiter = RateLimiter(max_requests=2, window_seconds=10.0)

    req = ProcessRequest(
        job_id="rate-test-1",
        job_type="llm_generate",
        user_id="rate-limited-user",
        payload={"prompt": "hi", "max_new_tokens": 1},
    )

    # 1 and 2 succeed
    res1 = service.process(req)
    assert res1.status == "success"
    res2 = service.process(req)
    assert res2.status == "success"

    # 3 is blocked by rate limiter
    res3 = service.process(req)
    assert res3.status == "failed"
    assert "Rate limit exceeded" in (res3.error or "")
    assert res3.metadata.get("retry_after_seconds", 0) > 0


def test_service_handles_insufficient_headroom():
    from echoself_ai.service.unified import EchoSelfAIService, ProcessRequest

    service = EchoSelfAIService()
    service.watchdog = ResourceWatchdog(WatchdogConfig(min_free_ram_mb=999999999.0))

    req = ProcessRequest(
        job_id="headroom-test-1",
        job_type="llm_generate",
        user_id="user-headroom",
        payload={"prompt": "hi", "max_new_tokens": 1},
    )

    stats = service.watchdog.get_memory_stats()
    if stats["mem_total_mb"] > 0:
        res = service.process(req)
        assert res.status == "failed"
        assert "Resource constraint" in (res.error or "")

