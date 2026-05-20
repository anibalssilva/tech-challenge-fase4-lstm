import os
import time
from collections import deque
from threading import Lock
from typing import Any

import psutil

_MAX_HISTORY = 120  # ~1 hora a 30s de intervalo


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.started_at = time.time()
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.last_latency_ms = 0.0
        self._history: deque = deque(maxlen=_MAX_HISTORY)

    def register_request(self, latency_ms: float, is_error: bool = False) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += latency_ms
            self.last_latency_ms = latency_ms
            if is_error:
                self.error_count += 1

    def register_prediction(self) -> None:
        with self._lock:
            self.prediction_count += 1

    def snapshot(self) -> dict[str, Any]:
        process = psutil.Process(os.getpid())
        with self._lock:
            avg_latency_ms = (
                self.total_latency_ms / self.request_count
                if self.request_count
                else 0.0
            )
            return {
                "uptime_seconds": round(time.time() - self.started_at, 2),
                "request_count": self.request_count,
                "prediction_count": self.prediction_count,
                "error_count": self.error_count,
                "avg_latency_ms": round(avg_latency_ms, 2),
                "last_latency_ms": round(self.last_latency_ms, 2),
                "cpu_percent": process.cpu_percent(interval=None),
                "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            }

    def record_snapshot(self) -> None:
        point = self.snapshot()
        point["ts"] = time.time()
        with self._lock:
            self._history.append(point)

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)


metrics_store = MetricsStore()
