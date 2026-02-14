"""Tests for the CloudWatch metrics client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.services.metrics import MetricsClient


class TestMetricsRecording:
    """Verify that record_success / record_failure buffer the right data."""

    def _make_client(self, *, enabled: bool = False) -> MetricsClient:
        with patch.dict("os.environ", {"METRICS_ENABLED": str(enabled).lower()}):
            return MetricsClient()

    def test_record_success_appends_two_data_points(self):
        client = self._make_client()
        client.record_success("calendly", "GET /event_types", latency_ms=123.4)
        # Should buffer RequestCount + Latency
        assert len(client._buffer) == 2
        names = {m["MetricName"] for m in client._buffer}
        assert names == {"ExternalAPI/RequestCount", "ExternalAPI/Latency"}

    def test_record_failure_appends_count_and_error(self):
        client = self._make_client()
        client.record_failure("anthropic", "llm_invoke", error_type="timeout")
        # Should buffer RequestCount + ErrorCount (no latency since default 0)
        assert len(client._buffer) == 2
        names = {m["MetricName"] for m in client._buffer}
        assert names == {"ExternalAPI/RequestCount", "ExternalAPI/ErrorCount"}

    def test_record_failure_with_latency_appends_three_data_points(self):
        client = self._make_client()
        client.record_failure(
            "calendly", "POST /invitees",
            error_type="4xx", latency_ms=500.0,
        )
        assert len(client._buffer) == 3
        names = {m["MetricName"] for m in client._buffer}
        assert names == {
            "ExternalAPI/RequestCount",
            "ExternalAPI/ErrorCount",
            "ExternalAPI/Latency",
        }

    def test_success_dimensions_include_service_and_status(self):
        client = self._make_client()
        client.record_success("calendly", "GET /users/me", latency_ms=50.0)
        count_metric = next(
            m for m in client._buffer
            if m["MetricName"] == "ExternalAPI/RequestCount"
        )
        dim_map = {d["Name"]: d["Value"] for d in count_metric["Dimensions"]}
        assert dim_map["Service"] == "calendly"
        assert dim_map["Status"] == "success"

    def test_failure_dimensions_include_error_type(self):
        client = self._make_client()
        client.record_failure("anthropic", "llm_invoke", error_type="BadRequestError")
        error_metric = next(
            m for m in client._buffer
            if m["MetricName"] == "ExternalAPI/ErrorCount"
        )
        dim_map = {d["Name"]: d["Value"] for d in error_metric["Dimensions"]}
        assert dim_map["ErrorType"] == "BadRequestError"


class TestMetricsFlush:
    """Verify flush behaviour with and without CloudWatch enabled."""

    def test_flush_when_disabled_does_not_call_boto3(self):
        with patch.dict("os.environ", {"METRICS_ENABLED": "false"}):
            client = MetricsClient()
        client.record_success("calendly", "GET /event_types", latency_ms=100.0)
        sent = client.flush()
        assert sent == 0  # disabled — nothing sent

    def test_flush_clears_buffer(self):
        with patch.dict("os.environ", {"METRICS_ENABLED": "false"}):
            client = MetricsClient()
        client.record_success("calendly", "GET /event_types", latency_ms=100.0)
        assert len(client._buffer) == 2
        client.flush()
        assert len(client._buffer) == 0

    def test_flush_when_enabled_calls_put_metric_data(self):
        with patch.dict("os.environ", {"METRICS_ENABLED": "true"}):
            client = MetricsClient()

        mock_cw = MagicMock()
        client._cw_client = mock_cw  # inject mock

        client.record_success("calendly", "GET /event_types", latency_ms=100.0)
        sent = client.flush()

        assert sent == 2
        mock_cw.put_metric_data.assert_called_once()
        call_args = mock_cw.put_metric_data.call_args
        assert call_args[1]["Namespace"] == "AcmeDental"
        assert len(call_args[1]["MetricData"]) == 2

    def test_flush_empty_buffer_returns_zero(self):
        with patch.dict("os.environ", {"METRICS_ENABLED": "true"}):
            client = MetricsClient()
        assert client.flush() == 0
