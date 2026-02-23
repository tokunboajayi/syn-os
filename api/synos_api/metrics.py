"""
InfluxDB Metrics Writer for Syn OS

Writes task execution metrics to InfluxDB for time-series analysis.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

# InfluxDB client is optional
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    HAS_INFLUXDB = True
except ImportError:
    HAS_INFLUXDB = False
    logger.warning("influxdb-client not installed, metrics will be logged only")


@dataclass
class TaskMetric:
    """Single task execution metric."""
    
    task_id: str
    task_name: str
    status: str  # completed, failed, cancelled
    duration_ms: int
    queue_wait_ms: int
    cpu_used_percent: float
    memory_used_mb: float
    priority: int
    node_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class SystemMetric:
    """System-level metric."""
    
    metric_name: str
    value: float
    tags: Optional[Dict[str, str]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class InfluxDBWriter:
    """
    Writes metrics to InfluxDB.
    
    Supports batched writes for efficiency.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "synos-token",
        org: str = "synos",
        bucket: str = "metrics",
        batch_size: int = 100,
    ):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.batch_size = batch_size
        
        self._client: Optional[Any] = None
        self._write_api: Optional[Any] = None
        self._batch: List[Any] = []
        
        if HAS_INFLUXDB:
            try:
                self._client = InfluxDBClient(url=url, token=token, org=org)
                self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
                logger.info(f"Connected to InfluxDB at {url}")
            except Exception as e:
                logger.warning(f"Failed to connect to InfluxDB: {e}")
                self._client = None
    
    def write_task_metric(self, metric: TaskMetric) -> bool:
        """Write a task execution metric."""
        if not HAS_INFLUXDB or self._write_api is None:
            logger.debug(f"Task metric: {metric}")
            return False
        
        try:
            point = (
                Point("task_execution")
                .tag("task_id", metric.task_id)
                .tag("task_name", metric.task_name)
                .tag("status", metric.status)
                .tag("priority", str(metric.priority))
                .field("duration_ms", metric.duration_ms)
                .field("queue_wait_ms", metric.queue_wait_ms)
                .field("cpu_used_percent", metric.cpu_used_percent)
                .field("memory_used_mb", metric.memory_used_mb)
                .time(metric.timestamp, WritePrecision.MS)
            )
            
            if metric.node_id:
                point = point.tag("node_id", metric.node_id)
            
            if metric.tags:
                for key, value in metric.tags.items():
                    point = point.tag(key, value)
            
            self._batch.append(point)
            
            if len(self._batch) >= self.batch_size:
                self.flush()
            
            return True
        except Exception as e:
            logger.error(f"Failed to write task metric: {e}")
            return False
    
    def write_system_metric(self, metric: SystemMetric) -> bool:
        """Write a system-level metric."""
        if not HAS_INFLUXDB or self._write_api is None:
            logger.debug(f"System metric: {metric}")
            return False
        
        try:
            point = (
                Point("system_metrics")
                .field(metric.metric_name, metric.value)
                .time(metric.timestamp, WritePrecision.MS)
            )
            
            if metric.tags:
                for key, value in metric.tags.items():
                    point = point.tag(key, value)
            
            self._batch.append(point)
            
            if len(self._batch) >= self.batch_size:
                self.flush()
            
            return True
        except Exception as e:
            logger.error(f"Failed to write system metric: {e}")
            return False
    
    def write_ml_metric(
        self,
        model_name: str,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
    ) -> bool:
        """Write ML model training/inference metric."""
        if not HAS_INFLUXDB or self._write_api is None:
            logger.debug(f"ML metric: {model_name}.{metric_name}={value}")
            return False
        
        try:
            point = (
                Point("ml_metrics")
                .tag("model", model_name)
                .field(metric_name, value)
                .time(datetime.utcnow(), WritePrecision.MS)
            )
            
            if step is not None:
                point = point.field("step", step)
            
            self._batch.append(point)
            
            if len(self._batch) >= self.batch_size:
                self.flush()
            
            return True
        except Exception as e:
            logger.error(f"Failed to write ML metric: {e}")
            return False
    
    def flush(self):
        """Flush batched points to InfluxDB."""
        if not self._batch:
            return
        
        if self._write_api:
            try:
                self._write_api.write(bucket=self.bucket, org=self.org, record=self._batch)
                logger.debug(f"Flushed {len(self._batch)} points to InfluxDB")
            except Exception as e:
                logger.error(f"Failed to flush metrics: {e}")
        
        self._batch = []
    
    def close(self):
        """Close the connection."""
        self.flush()
        if self._client:
            self._client.close()
            logger.info("Closed InfluxDB connection")


# Singleton instance
_writer: Optional[InfluxDBWriter] = None


def get_metrics_writer() -> InfluxDBWriter:
    """Get the global metrics writer instance."""
    global _writer
    if _writer is None:
        import os
        _writer = InfluxDBWriter(
            url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
            token=os.getenv("INFLUXDB_TOKEN", "synos-token"),
            org=os.getenv("INFLUXDB_ORG", "synos"),
            bucket=os.getenv("INFLUXDB_BUCKET", "metrics"),
        )
    return _writer


async def record_task_completion(
    task_id: str,
    task_name: str,
    success: bool,
    duration_ms: int,
    queue_wait_ms: int = 0,
    cpu_percent: float = 0.0,
    memory_mb: float = 0.0,
    priority: int = 5,
    node_id: Optional[str] = None,
):
    """Convenience function to record task completion."""
    writer = get_metrics_writer()
    writer.write_task_metric(TaskMetric(
        task_id=task_id,
        task_name=task_name,
        status="completed" if success else "failed",
        duration_ms=duration_ms,
        queue_wait_ms=queue_wait_ms,
        cpu_used_percent=cpu_percent,
        memory_used_mb=memory_mb,
        priority=priority,
        node_id=node_id,
    ))


async def record_ml_training_step(
    model_name: str,
    loss: float,
    step: int,
    **additional_metrics,
):
    """Record ML training step metrics."""
    writer = get_metrics_writer()
    writer.write_ml_metric(model_name, "loss", loss, step)
    for name, value in additional_metrics.items():
        writer.write_ml_metric(model_name, name, value, step)
