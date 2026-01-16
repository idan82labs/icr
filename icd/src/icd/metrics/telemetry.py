"""
Local telemetry collection for ICD.

Collects and stores metrics locally for analysis and optimization.
No data is sent externally.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from icd.config import Config

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected."""

    RETRIEVAL = "retrieval"
    INDEXING = "indexing"
    EMBEDDING = "embedding"
    PACK = "pack"
    RLM = "rlm"
    ERROR = "error"
    PERFORMANCE = "performance"


@dataclass
class Metric:
    """A single metric data point."""

    metric_type: MetricType
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""

    name: str
    count: int
    sum: float
    mean: float
    min: float
    max: float
    std: float
    period_start: datetime
    period_end: datetime


class TelemetryCollector:
    """
    Local telemetry collector and storage.

    Features:
    - SQLite-based local storage
    - Automatic aggregation
    - Retention management
    - Query interface
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_type TEXT NOT NULL,
        name TEXT NOT NULL,
        value REAL NOT NULL,
        timestamp TEXT NOT NULL,
        tags TEXT DEFAULT '{}',
        metadata TEXT DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type);
    CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

    CREATE TABLE IF NOT EXISTS metric_aggregates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_type TEXT NOT NULL,
        name TEXT NOT NULL,
        period_start TEXT NOT NULL,
        period_end TEXT NOT NULL,
        count INTEGER NOT NULL,
        sum_value REAL NOT NULL,
        min_value REAL NOT NULL,
        max_value REAL NOT NULL,
        mean_value REAL NOT NULL,
        std_value REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_aggregates_name ON metric_aggregates(name);
    CREATE INDEX IF NOT EXISTS idx_aggregates_period ON metric_aggregates(period_start);
    """

    def __init__(self, config: "Config") -> None:
        """
        Initialize the telemetry collector.

        Args:
            config: ICD configuration.
        """
        self.config = config
        self.enabled = config.telemetry.enabled
        self.db_path = config.metrics_path
        self.retention_days = config.telemetry.retention_days

        self._conn: sqlite3.Connection | None = None
        self._buffer: list[Metric] = []
        self._buffer_size = 100
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the telemetry storage."""
        if not self.enabled:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.executescript(self.SCHEMA)

        logger.info("Telemetry collector initialized", db_path=str(self.db_path))

    async def close(self) -> None:
        """Close the telemetry storage."""
        if self._buffer:
            await self._flush_buffer()

        if self._conn:
            self._conn.close()
            self._conn = None

    async def record(
        self,
        metric_type: MetricType,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a metric.

        Args:
            metric_type: Type of metric.
            name: Metric name.
            value: Metric value.
            tags: Optional tags.
            metadata: Optional metadata.
        """
        if not self.enabled:
            return

        metric = Metric(
            metric_type=metric_type,
            name=name,
            value=value,
            tags=tags or {},
            metadata=metadata or {},
        )

        async with self._lock:
            self._buffer.append(metric)

            if len(self._buffer) >= self._buffer_size:
                await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """Flush buffered metrics to storage."""
        if not self._conn or not self._buffer:
            return

        cursor = self._conn.cursor()

        for metric in self._buffer:
            cursor.execute(
                """
                INSERT INTO metrics (metric_type, name, value, timestamp, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.metric_type.value,
                    metric.name,
                    metric.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.tags),
                    json.dumps(metric.metadata),
                ),
            )

        self._conn.commit()
        self._buffer.clear()

    async def query(
        self,
        name: str | None = None,
        metric_type: MetricType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[Metric]:
        """
        Query metrics.

        Args:
            name: Filter by metric name.
            metric_type: Filter by metric type.
            start_time: Filter by start time.
            end_time: Filter by end time.
            limit: Maximum results.

        Returns:
            List of matching metrics.
        """
        if not self._conn:
            return []

        # Flush buffer first
        if self._buffer:
            await self._flush_buffer()

        conditions = []
        params: list[Any] = []

        if name:
            conditions.append("name = ?")
            params.append(name)

        if metric_type:
            conditions.append("metric_type = ?")
            params.append(metric_type.value)

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cursor = self._conn.execute(
            f"""
            SELECT metric_type, name, value, timestamp, tags, metadata
            FROM metrics
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        )

        metrics = []
        for row in cursor.fetchall():
            metrics.append(
                Metric(
                    metric_type=MetricType(row[0]),
                    name=row[1],
                    value=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    tags=json.loads(row[4]),
                    metadata=json.loads(row[5]),
                )
            )

        return metrics

    async def get_summary(
        self,
        name: str,
        period_hours: int = 24,
    ) -> MetricSummary | None:
        """
        Get summary statistics for a metric.

        Args:
            name: Metric name.
            period_hours: Period to summarize in hours.

        Returns:
            MetricSummary or None.
        """
        if not self._conn:
            return None

        # Flush buffer first
        if self._buffer:
            await self._flush_buffer()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=period_hours)

        cursor = self._conn.execute(
            """
            SELECT
                COUNT(*) as count,
                SUM(value) as sum,
                AVG(value) as mean,
                MIN(value) as min,
                MAX(value) as max
            FROM metrics
            WHERE name = ? AND timestamp >= ?
            """,
            (name, start_time.isoformat()),
        )

        row = cursor.fetchone()
        if not row or row[0] == 0:
            return None

        count, total, mean, min_val, max_val = row

        # Compute std deviation
        cursor = self._conn.execute(
            """
            SELECT value FROM metrics
            WHERE name = ? AND timestamp >= ?
            """,
            (name, start_time.isoformat()),
        )

        values = [r[0] for r in cursor.fetchall()]
        import numpy as np

        std = float(np.std(values)) if values else 0.0

        return MetricSummary(
            name=name,
            count=count,
            sum=total,
            mean=mean,
            min=min_val,
            max=max_val,
            std=std,
            period_start=start_time,
            period_end=end_time,
        )

    async def aggregate_and_prune(self) -> None:
        """
        Aggregate old metrics and prune raw data.

        Keeps detailed data for recent period, aggregates older data.
        """
        if not self._conn:
            return

        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)

        # Get unique metric names older than cutoff
        cursor = self._conn.execute(
            """
            SELECT DISTINCT metric_type, name
            FROM metrics
            WHERE timestamp < ?
            """,
            (cutoff.isoformat(),),
        )

        metric_names = cursor.fetchall()

        for metric_type, name in metric_names:
            # Aggregate by day
            cursor = self._conn.execute(
                """
                SELECT
                    date(timestamp) as day,
                    COUNT(*) as count,
                    SUM(value) as sum,
                    MIN(value) as min,
                    MAX(value) as max,
                    AVG(value) as mean
                FROM metrics
                WHERE metric_type = ? AND name = ? AND timestamp < ?
                GROUP BY date(timestamp)
                """,
                (metric_type, name, cutoff.isoformat()),
            )

            for row in cursor.fetchall():
                day, count, total, min_val, max_val, mean = row

                # Compute std (simplified)
                std = 0.0

                # Store aggregate
                self._conn.execute(
                    """
                    INSERT INTO metric_aggregates
                    (metric_type, name, period_start, period_end, count, sum_value,
                     min_value, max_value, mean_value, std_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metric_type,
                        name,
                        f"{day}T00:00:00",
                        f"{day}T23:59:59",
                        count,
                        total,
                        min_val,
                        max_val,
                        mean,
                        std,
                    ),
                )

        # Delete old raw metrics
        self._conn.execute(
            "DELETE FROM metrics WHERE timestamp < ?",
            (cutoff.isoformat(),),
        )

        self._conn.commit()
        logger.info("Telemetry aggregation and pruning complete")

    async def get_all_metric_names(self) -> list[str]:
        """Get all recorded metric names."""
        if not self._conn:
            return []

        cursor = self._conn.execute("SELECT DISTINCT name FROM metrics")
        return [row[0] for row in cursor.fetchall()]

    async def export_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        format: str = "json",
    ) -> str:
        """
        Export metrics to a string.

        Args:
            start_time: Filter by start time.
            end_time: Filter by end time.
            format: Export format (json, csv).

        Returns:
            Exported metrics string.
        """
        metrics = await self.query(start_time=start_time, end_time=end_time, limit=10000)

        if format == "csv":
            lines = ["metric_type,name,value,timestamp,tags"]
            for m in metrics:
                tags_str = ";".join(f"{k}={v}" for k, v in m.tags.items())
                lines.append(
                    f"{m.metric_type.value},{m.name},{m.value},{m.timestamp.isoformat()},{tags_str}"
                )
            return "\n".join(lines)

        else:  # json
            data = [
                {
                    "metric_type": m.metric_type.value,
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "tags": m.tags,
                    "metadata": m.metadata,
                }
                for m in metrics
            ]
            return json.dumps(data, indent=2)


# Convenience functions for common metrics
async def record_retrieval_latency(
    collector: TelemetryCollector,
    latency_ms: float,
    query_type: str = "hybrid",
) -> None:
    """Record retrieval latency."""
    await collector.record(
        metric_type=MetricType.RETRIEVAL,
        name="retrieval_latency_ms",
        value=latency_ms,
        tags={"query_type": query_type},
    )


async def record_retrieval_entropy(
    collector: TelemetryCollector,
    entropy: float,
) -> None:
    """Record retrieval entropy."""
    await collector.record(
        metric_type=MetricType.RETRIEVAL,
        name="retrieval_entropy",
        value=entropy,
    )


async def record_indexing_rate(
    collector: TelemetryCollector,
    chunks_per_second: float,
) -> None:
    """Record indexing rate."""
    await collector.record(
        metric_type=MetricType.INDEXING,
        name="indexing_rate_cps",
        value=chunks_per_second,
    )


async def record_embedding_latency(
    collector: TelemetryCollector,
    latency_ms: float,
    batch_size: int,
) -> None:
    """Record embedding generation latency."""
    await collector.record(
        metric_type=MetricType.EMBEDDING,
        name="embedding_latency_ms",
        value=latency_ms,
        tags={"batch_size": str(batch_size)},
    )


async def record_pack_compilation(
    collector: TelemetryCollector,
    tokens: int,
    chunks: int,
    latency_ms: float,
) -> None:
    """Record pack compilation metrics."""
    await collector.record(
        metric_type=MetricType.PACK,
        name="pack_tokens",
        value=float(tokens),
    )
    await collector.record(
        metric_type=MetricType.PACK,
        name="pack_chunks",
        value=float(chunks),
    )
    await collector.record(
        metric_type=MetricType.PACK,
        name="pack_latency_ms",
        value=latency_ms,
    )


async def record_error(
    collector: TelemetryCollector,
    error_type: str,
    component: str,
) -> None:
    """Record an error occurrence."""
    await collector.record(
        metric_type=MetricType.ERROR,
        name="error_count",
        value=1.0,
        tags={"error_type": error_type, "component": component},
    )
