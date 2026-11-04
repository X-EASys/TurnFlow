"""
Route Metrics Collector

Collects per-call timing and success metrics for comparing
relay (Vercel proxy) vs direct (SSH tunnel) Gemini API routing.

Usage:
    from benchmark.utils.route_metrics import RouteMetricsCollector

    collector = RouteMetricsCollector.get_instance()
    collector.record_call(model="gemini-2.5-flash", route_mode="relay",
                          latency_ms=1234.5, success=True)
    collector.print_summary()
"""

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class CallMetric:
    """Single API call measurement."""

    timestamp: str
    model: str
    route_mode: str  # "relay" or "direct"
    latency_ms: float
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    error_type: Optional[str] = None
    error_msg: Optional[str] = None


@dataclass
class RouteStats:
    """Aggregated statistics for a single route mode."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    success_rate: float = 0.0
    latency_mean_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class RouteComparisonReport:
    """Aggregated comparison between relay and direct routing."""

    relay_stats: Dict[str, Any]
    direct_stats: Dict[str, Any]
    summary: str
    generated_at: str


class RouteMetricsCollector:
    """
    Thread-safe metrics collector (singleton).

    Records every LLM API call with its route mode, latency, and outcome.
    Generates comparison reports between relay and direct modes.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._metrics: List[CallMetric] = []
        self._metrics_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "RouteMetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def record_call(
        self,
        model: str,
        route_mode: str,
        latency_ms: float,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error_type: Optional[str] = None,
        error_msg: Optional[str] = None,
    ):
        """Record a single API call metric."""
        metric = CallMetric(
            timestamp=datetime.now().isoformat(),
            model=model,
            route_mode=route_mode,
            latency_ms=latency_ms,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error_type=error_type,
            error_msg=error_msg,
        )
        with self._metrics_lock:
            self._metrics.append(metric)

    def get_metrics(
        self, route_mode: Optional[str] = None
    ) -> List[CallMetric]:
        """Get collected metrics, optionally filtered by route mode."""
        with self._metrics_lock:
            if route_mode:
                return [
                    m for m in self._metrics if m.route_mode == route_mode
                ]
            return list(self._metrics)

    def _compute_stats(self, metrics: List[CallMetric]) -> Dict[str, Any]:
        """Compute aggregate statistics for a set of metrics."""
        if not metrics:
            return asdict(RouteStats())

        successes = [m for m in metrics if m.success]
        failures = [m for m in metrics if not m.success]
        latencies = sorted([m.latency_ms for m in successes])

        stats = RouteStats(
            total_calls=len(metrics),
            successful_calls=len(successes),
            failed_calls=len(failures),
            success_rate=len(successes) / len(metrics),
        )

        if latencies:
            stats.latency_mean_ms = sum(latencies) / len(latencies)
            stats.latency_min_ms = latencies[0]
            stats.latency_max_ms = latencies[-1]
            stats.latency_p50_ms = latencies[len(latencies) // 2]
            stats.latency_p90_ms = latencies[
                min(int(len(latencies) * 0.9), len(latencies) - 1)
            ]
            stats.latency_p99_ms = latencies[
                min(int(len(latencies) * 0.99), len(latencies) - 1)
            ]

        stats.total_input_tokens = sum(m.input_tokens for m in successes)
        stats.total_output_tokens = sum(m.output_tokens for m in successes)

        error_types: Dict[str, int] = {}
        for m in failures:
            et = m.error_type or "unknown"
            error_types[et] = error_types.get(et, 0) + 1
        stats.error_types = error_types

        return asdict(stats)

    def generate_comparison_report(self) -> RouteComparisonReport:
        """Generate a comparison report between relay and direct modes."""
        relay_metrics = self.get_metrics("relay")
        direct_metrics = self.get_metrics("direct")

        relay_stats = self._compute_stats(relay_metrics)
        direct_stats = self._compute_stats(direct_metrics)

        lines = [
            "Route Comparison: Relay (Third-party Proxy) vs Direct (SSH Tunnel + Google API)",
            "=" * 78,
        ]

        for label, stats in [("RELAY", relay_stats), ("DIRECT", direct_stats)]:
            lines.append(f"\n  {label}:")
            lines.append(f"    Total calls:    {stats['total_calls']}")
            lines.append(f"    Success rate:   {stats['success_rate']:.1%}")
            lines.append(
                f"    Latency (ms):   mean={stats['latency_mean_ms']:.0f}  "
                f"p50={stats['latency_p50_ms']:.0f}  "
                f"p90={stats['latency_p90_ms']:.0f}  "
                f"p99={stats['latency_p99_ms']:.0f}  "
                f"min={stats['latency_min_ms']:.0f}  "
                f"max={stats['latency_max_ms']:.0f}"
            )
            lines.append(
                f"    Tokens in/out:  {stats['total_input_tokens']}/{stats['total_output_tokens']}"
            )
            if stats.get("error_types"):
                lines.append(f"    Errors:         {stats['error_types']}")

        # Speedup comparison
        relay_mean = relay_stats.get("latency_mean_ms", 0)
        direct_mean = direct_stats.get("latency_mean_ms", 0)
        if (
            relay_stats.get("total_calls", 0) > 0
            and direct_stats.get("total_calls", 0) > 0
            and relay_mean > 0
        ):
            speedup = (relay_mean - direct_mean) / relay_mean * 100
            direction = "faster" if speedup > 0 else "slower"
            lines.append(
                f"\n  >> Direct is {abs(speedup):.1f}% {direction} than relay on average"
            )
            lines.append(
                f"  >> Relay mean: {relay_mean:.0f}ms vs Direct mean: {direct_mean:.0f}ms"
            )

        # Reliability comparison
        relay_rate = relay_stats.get("success_rate", 0)
        direct_rate = direct_stats.get("success_rate", 0)
        if (
            relay_stats.get("total_calls", 0) > 0
            and direct_stats.get("total_calls", 0) > 0
        ):
            if direct_rate > relay_rate:
                lines.append(
                    f"  >> Direct is more reliable ({direct_rate:.1%} vs {relay_rate:.1%})"
                )
            elif relay_rate > direct_rate:
                lines.append(
                    f"  >> Relay is more reliable ({relay_rate:.1%} vs {direct_rate:.1%})"
                )
            else:
                lines.append(
                    f"  >> Both have equal reliability ({relay_rate:.1%})"
                )

        summary = "\n".join(lines)

        return RouteComparisonReport(
            relay_stats=relay_stats,
            direct_stats=direct_stats,
            summary=summary,
            generated_at=datetime.now().isoformat(),
        )

    def save_report(self, output_path: str):
        """Save comparison report and raw metrics to disk."""
        report = self.generate_comparison_report()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "report": asdict(report),
            "raw_metrics": [asdict(m) for m in self._metrics],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Route metrics report saved to {path}")

    def print_summary(self):
        """Print comparison summary to stdout."""
        report = self.generate_comparison_report()
        print(report.summary)
