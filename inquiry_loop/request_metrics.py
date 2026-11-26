"""Aggregate model speed across every tool/answer turn of one agent request."""

import math


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


class RequestMetrics:
    def __init__(self):
        self.tokens = 0.0
        self.seconds = 0.0
        self.kind = None
        self.complete = True
        self.calls = 0

    def record(self, stats):
        self.calls += 1
        if not isinstance(stats, dict):
            self.complete = False
            return
        tokens = _number(stats.get("generation_tokens"))
        seconds = _number(stats.get("generation_seconds"))
        rate = _number(stats.get("generation_tps"))
        kind = stats.get("speed_kind", "generation")
        # Strata's engine clock/rate measures native decode work, which can
        # differ from API-visible tokens after parsing or speculative decoding.
        if seconds is not None and seconds > 0 and rate is not None and rate > 0:
            tokens = rate * seconds
        elif tokens is not None and tokens > 0 and rate is not None and rate > 0:
            seconds = tokens / rate
        if (tokens is None or tokens <= 0 or seconds is None or seconds <= 0
                or kind not in ("generation", "request")
                or self.kind is not None and kind != self.kind):
            self.complete = False
            return
        self.kind = kind
        self.tokens += tokens
        self.seconds += seconds

    def as_dict(self):
        return {"calls": self.calls, "complete": self.complete, "speed_kind": self.kind,
                "measured_tokens": self.tokens, "measurement_seconds": self.seconds,
                "tokens_per_second": self.tokens / self.seconds if self.complete and self.calls and self.seconds else None}

    def summary(self, elapsed, backend):
        rate = f"{self.tokens / self.seconds:.2f}" if self.complete and self.calls and self.seconds else "N/A"
        suffix = " (includes prompt processing and request overhead)" if rate != "N/A" and self.kind == "request" else ""
        return f"{backend} processed request in {elapsed:.2f} seconds, tokens per second = {rate}{suffix}"
