"""CloudWatch custom metrics emitter with background batching.

Publishes per-API-call metrics (count, latency, errors) for every
external service the agent interacts with (Calendly, Anthropic).

Design
------
* Metrics are collected in a thread-safe in-memory buffer.
* A daemon thread flushes the buffer to CloudWatch every
  ``FLUSH_INTERVAL_SECONDS`` (default 60 s).
* When running locally (``METRICS_ENABLED != "true"``), metrics are
  logged to stdout at DEBUG level but **not** pushed to CloudWatch.
* Each ``put_metric_data`` call sends up to 1 000 metric data points
  (the CloudWatch API limit per request).

Usage
-----
>>> from src.services.metrics import metrics
>>> metrics.record_success("calendly", "GET /event_types", latency_ms=123.4)
>>> metrics.record_failure("anthropic", "llm_invoke", error_type="timeout")
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

NAMESPACE = "AcmeDental"
FLUSH_INTERVAL_SECONDS = 60
MAX_BATCH_SIZE = 1_000  # CloudWatch API limit per PutMetricData call


class MetricsClient:
    """Batched CloudWatch metrics publisher."""

    def __init__(self) -> None:
        self._enabled = os.getenv("METRICS_ENABLED", "false").lower() == "true"
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._cw_client = None  # lazy-init

        if self._enabled:
            self._start_flush_thread()

    # ── Lazy CloudWatch client ────────────────────────────────────────

    def _get_cw_client(self):
        """Create the boto3 CloudWatch client on first use."""
        if self._cw_client is None:
            import boto3

            self._cw_client = boto3.client("cloudwatch")
        return self._cw_client

    # ── Public API ────────────────────────────────────────────────────

    def record_success(
        self,
        service: str,
        operation: str,
        latency_ms: float,
    ) -> None:
        """Record a successful API call."""
        now = datetime.now(UTC)
        dims_base = [
            {"Name": "Service", "Value": service},
        ]
        dims_op = dims_base + [{"Name": "Operation", "Value": operation}]

        self._append(
            {
                "MetricName": "ExternalAPI/RequestCount",
                "Dimensions": dims_base + [{"Name": "Status", "Value": "success"}],
                "Timestamp": now,
                "Value": 1,
                "Unit": "Count",
            }
        )
        self._append(
            {
                "MetricName": "ExternalAPI/Latency",
                "Dimensions": dims_op,
                "Timestamp": now,
                "Value": latency_ms,
                "Unit": "Milliseconds",
            }
        )
        logger.debug(
            "Metric: %s %s success latency=%.1fms", service, operation, latency_ms,
        )

    def record_failure(
        self,
        service: str,
        operation: str,
        error_type: str,
        latency_ms: float = 0,
    ) -> None:
        """Record a failed API call."""
        now = datetime.now(UTC)
        dims_base = [
            {"Name": "Service", "Value": service},
        ]

        self._append(
            {
                "MetricName": "ExternalAPI/RequestCount",
                "Dimensions": dims_base + [{"Name": "Status", "Value": "failure"}],
                "Timestamp": now,
                "Value": 1,
                "Unit": "Count",
            }
        )
        self._append(
            {
                "MetricName": "ExternalAPI/ErrorCount",
                "Dimensions": dims_base + [{"Name": "ErrorType", "Value": error_type}],
                "Timestamp": now,
                "Value": 1,
                "Unit": "Count",
            }
        )
        if latency_ms > 0:
            self._append(
                {
                    "MetricName": "ExternalAPI/Latency",
                    "Dimensions": dims_base + [{"Name": "Operation", "Value": operation}],
                    "Timestamp": now,
                    "Value": latency_ms,
                    "Unit": "Milliseconds",
                }
            )
        logger.debug(
            "Metric: %s %s failure error=%s latency=%.1fms",
            service, operation, error_type, latency_ms,
        )

    def flush(self) -> int:
        """Send buffered metrics to CloudWatch.  Returns count sent."""
        with self._lock:
            if not self._buffer:
                return 0
            batch = self._buffer[:]
            self._buffer.clear()

        if not self._enabled:
            logger.debug("Metrics flush skipped (not enabled): %d items", len(batch))
            return 0

        sent = 0
        try:
            cw = self._get_cw_client()
            # CloudWatch accepts max 1000 metric data points per call
            for i in range(0, len(batch), MAX_BATCH_SIZE):
                chunk = batch[i : i + MAX_BATCH_SIZE]
                cw.put_metric_data(Namespace=NAMESPACE, MetricData=chunk)
                sent += len(chunk)
            logger.info("Flushed %d metrics to CloudWatch", sent)
        except Exception:
            logger.exception("Failed to flush metrics to CloudWatch")
        return sent

    # ── Internal ──────────────────────────────────────────────────────

    def _append(self, metric_data: dict[str, Any]) -> None:
        with self._lock:
            self._buffer.append(metric_data)

    def _start_flush_thread(self) -> None:
        """Start a daemon thread that flushes metrics periodically."""

        def _loop():
            while True:
                time.sleep(FLUSH_INTERVAL_SECONDS)
                try:
                    self.flush()
                except Exception:
                    logger.exception("Metrics flush thread error")

        t = threading.Thread(target=_loop, daemon=True, name="metrics-flush")
        t.start()
        atexit.register(self.flush)  # flush on process exit
        logger.info(
            "Metrics flush thread started (interval=%ds)", FLUSH_INTERVAL_SECONDS,
        )


# ── Module-level singleton ──────────────────────────────────────────
metrics = MetricsClient()
