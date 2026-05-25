"""
Drift Monitor — detects distribution shift in incoming text.

Strategy (lightweight, no GPU needed):
  • Text-length distribution  → mean / std drift via z-score
  • Vocabulary novelty ratio  → % of tokens not seen in baseline vocab

Baseline is built from the first BASELINE_WINDOW requests.
After that, every EVAL_WINDOW requests we compare against baseline.
A z-score on length mean or high vocab novelty triggers the alert.
"""

import logging
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque

# Use standard logger here (not structured) to avoid circular import
_logger = logging.getLogger("drift_monitor")

BASELINE_WINDOW    = 200    # requests to warm up baseline
EVAL_WINDOW        = 50     # sliding window for live comparison
LENGTH_Z_THRESH    = 2.5    # standard deviations before flagging length shift
VOCAB_NOVEL_THRESH = 0.40   # 40% unseen tokens → flag vocab shift


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer — no external deps needed."""
    return re.findall(r"\b[a-z]+\b", text.lower())


@dataclass
class WindowStats:
    lengths: list[int] = field(default_factory=list)
    vocab: Counter = field(default_factory=Counter)

    def add(self, text: str):
        self.lengths.append(len(text))
        self.vocab.update(_tokenize(text))

    @property
    def mean_length(self) -> float:
        return sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    @property
    def std_length(self) -> float:
        if len(self.lengths) < 2:
            return 1.0
        m = self.mean_length
        return math.sqrt(sum((x - m) ** 2 for x in self.lengths) / len(self.lengths))

    @property
    def vocab_set(self) -> set[str]:
        return set(self.vocab.keys())


class DriftMonitor:
    def __init__(
        self,
        baseline_window: int = BASELINE_WINDOW,
        eval_window: int = EVAL_WINDOW,
    ):
        self._baseline_window = baseline_window
        self._eval_window = eval_window
        self._baseline: WindowStats | None = None
        self._buffer: list[str] = []
        self._live = WindowStats()
        self._total = 0
        self._drift_flags: Deque[bool] = deque(maxlen=10)

    # ── Public API ────────────────────────────────────────────────────────────
    def record(self, text: str):
        """Call once per incoming request to feed the monitor."""
        self._total += 1

        if self._baseline is None:
            # Still warming up — collect into buffer
            self._buffer.append(text)
            if len(self._buffer) >= self._baseline_window:
                self._build_baseline()
        else:
            self._live.add(text)
            if len(self._live.lengths) >= self._eval_window:
                flag = self._evaluate()
                self._drift_flags.append(flag)
                self._live = WindowStats()  # reset live window after eval

    def report(self) -> dict:
        """Return current drift statistics — served at GET /drift."""
        if self._baseline is None:
            return {
                "drift_detected": False,
                "status": "warming_up",
                "baseline_progress": f"{len(self._buffer)}/{self._baseline_window}",
                "total_requests": self._total,
            }

        flags = list(self._drift_flags)
        recent_drift = any(flags[-3:]) if flags else False   # any of last 3 evals
        drift_severity = "high" if recent_drift else "low"

        summary: dict = {
            "drift_detected": recent_drift,
            "status": "monitoring",
            "total_requests": self._total,
            "baseline_mean_length": round(self._baseline.mean_length, 1),
            "baseline_std_length": round(self._baseline.std_length, 1),
            "live_buffer_size": len(self._live.lengths),
            "eval_history": flags,
            "drift_severity": drift_severity,
        }

        # Add live stats when enough data is available
        if len(self._live.lengths) >= 5:
            z     = self._length_z(self._live)
            novel = self._vocab_novelty(self._live)
            summary["live_mean_length"]      = round(self._live.mean_length, 1)
            summary["length_z_score"]        = round(z, 3)
            summary["vocab_novelty_ratio"]   = round(novel, 3)

        return summary

    def reset(self):
        """Reset all state — call after every model swap."""
        self._baseline = None
        self._buffer = []
        self._live = WindowStats()
        self._drift_flags.clear()
        self._total = 0

    # ── Internal ──────────────────────────────────────────────────────────────
    def _build_baseline(self):
        self._baseline = WindowStats()
        for t in self._buffer:
            self._baseline.add(t)
        self._buffer = []
        _logger.info("Drift baseline built from %d requests.", self._baseline_window)

    def _length_z(self, window: WindowStats) -> float:
        if self._baseline is None:
            return 0.0
        std = self._baseline.std_length or 1.0
        return abs(window.mean_length - self._baseline.mean_length) / std

    def _vocab_novelty(self, window: WindowStats) -> float:
        if self._baseline is None or not window.vocab:
            return 0.0
        novel = window.vocab_set - self._baseline.vocab_set
        return len(novel) / len(window.vocab_set)

    def _evaluate(self) -> bool:
        z     = self._length_z(self._live)
        novel = self._vocab_novelty(self._live)
        drift = z > LENGTH_Z_THRESH or novel > VOCAB_NOVEL_THRESH

        _logger.info(
            "drift_evaluation z=%.3f novelty=%.3f drift=%s",
            z, novel, drift,
        )
        return drift