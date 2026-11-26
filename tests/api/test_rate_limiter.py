"""The sliding-window limiter on its own, with a fake clock (no sleeping)."""

from belay.api import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_the_limit_then_blocks():
    limiter = RateLimiter(3, 60, FakeClock())
    assert [limiter.retry_after("a") for _ in range(3)] == [None, None, None]
    assert limiter.retry_after("a") == 60


def test_window_slides_instead_of_resetting():
    clock = FakeClock()
    limiter = RateLimiter(2, 60, clock)
    limiter.retry_after("a")  # t = 1000
    clock.now += 30
    limiter.retry_after("a")  # t = 1030
    clock.now += 31  # t = 1061: the first call has left the window, the second hasn't
    assert limiter.retry_after("a") is None
    assert limiter.retry_after("a") == 29  # 1030 + 60 - 1061


def test_blocked_calls_do_not_extend_the_wait():
    clock = FakeClock()
    limiter = RateLimiter(1, 10, clock)
    limiter.retry_after("a")
    for _ in range(5):
        limiter.retry_after("a")  # refused, and not recorded
    clock.now += 10
    assert limiter.retry_after("a") is None


def test_clients_are_counted_separately():
    limiter = RateLimiter(1, 60, FakeClock())
    assert limiter.retry_after("a") is None
    assert limiter.retry_after("b") is None
    assert limiter.retry_after("a") is not None


def test_zero_switches_the_limit_off():
    limiter = RateLimiter(0, 60, FakeClock())
    assert all(limiter.retry_after("a") is None for _ in range(1000))
