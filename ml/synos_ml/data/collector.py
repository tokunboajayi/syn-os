"""
Execution Data Collector for Syn OS

Collects and stores task execution metrics for ML model training.
Supports both real-time streaming and batch export.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import json
import gzip
from collections import deque
from loguru import logger

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False


@dataclass
class ExecutionRecord:
    """Single task execution record for ML training."""
    
    # Task identification
    task_id: str
    task_name: str
    task_type: str = "generic"
    
    # Scheduling info
    priority: int = 5
    queue_position: int = 0
    queue_depth_at_submit: int = 0
    scheduled_node: Optional[str] = None
    
    # Resource requests
    requested_cpu_cores: int = 1
    requested_memory_mb: int = 1024
    requested_gpu: bool = False
    
    # Actual resource usage
    actual_cpu_percent: float = 0.0
    actual_memory_mb: float = 0.0
    actual_memory_peak_mb: float = 0.0
    actual_gpu_util_percent: float = 0.0
    actual_io_read_mb: float = 0.0
    actual_io_write_mb: float = 0.0
    
    # Timing
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Derived timing (computed)
    queue_wait_ms: int = 0
    execution_duration_ms: int = 0
    total_duration_ms: int = 0
    
    # Outcome
    status: str = "unknown"  # completed, failed, timeout, cancelled
    exit_code: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Dependencies
    dependency_count: int = 0
    dependency_wait_ms: int = 0
    
    # System state at execution time
    system_cpu_util: float = 0.0
    system_memory_util: float = 0.0
    active_tasks_count: int = 0
    
    # ML predictions (for comparison)
    predicted_duration_ms: Optional[int] = None
    predicted_memory_mb: Optional[float] = None
    prediction_error_ms: Optional[int] = None
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    
    def compute_derived_fields(self):
        """Compute derived timing fields."""
        if self.scheduled_at and self.submitted_at:
            self.queue_wait_ms = int((self.scheduled_at - self.submitted_at).total_seconds() * 1000)
        
        if self.completed_at and self.started_at:
            self.execution_duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
        
        if self.completed_at and self.submitted_at:
            self.total_duration_ms = int((self.completed_at - self.submitted_at).total_seconds() * 1000)
        
        if self.predicted_duration_ms and self.execution_duration_ms:
            self.prediction_error_ms = abs(self.execution_duration_ms - self.predicted_duration_ms)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Convert datetime objects
        for key in ['submitted_at', 'scheduled_at', 'started_at', 'completed_at']:
            if d[key] is not None:
                d[key] = d[key].isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionRecord":
        """Create from dictionary."""
        # Convert datetime strings
        for key in ['submitted_at', 'scheduled_at', 'started_at', 'completed_at']:
            if d.get(key) is not None and isinstance(d[key], str):
                d[key] = datetime.fromisoformat(d[key])
        return cls(**d)


class DataCollector:
    """
    Collects task execution data for ML training.
    
    Features:
    - In-memory buffer with configurable size
    - Automatic disk persistence
    - Batch export for training
    - Real-time streaming hooks
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        persist_path: Optional[Path] = None,
        persist_interval_seconds: int = 300,
        compress: bool = True,
    ):
        self.buffer_size = buffer_size
        self.persist_path = Path(persist_path) if persist_path else Path("./data/execution_records")
        self.persist_interval = persist_interval_seconds
        self.compress = compress
        
        self._buffer: deque[ExecutionRecord] = deque(maxlen=buffer_size)
        self._total_collected: int = 0
        self._persist_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[ExecutionRecord], None]] = []
        
        # Statistics
        self._stats = {
            "total_collected": 0,
            "total_persisted": 0,
            "completed_count": 0,
            "failed_count": 0,
            "avg_duration_ms": 0.0,
            "avg_queue_wait_ms": 0.0,
        }
        
        # Ensure persist path exists
        self.persist_path.mkdir(parents=True, exist_ok=True)
    
    def record(self, record: ExecutionRecord):
        """Add an execution record."""
        record.compute_derived_fields()
        self._buffer.append(record)
        self._total_collected += 1
        self._stats["total_collected"] += 1
        
        # Update statistics
        if record.status == "completed":
            self._stats["completed_count"] += 1
        elif record.status == "failed":
            self._stats["failed_count"] += 1
        
        # Running average of duration
        n = self._stats["completed_count"]
        if n > 0:
            self._stats["avg_duration_ms"] = (
                (self._stats["avg_duration_ms"] * (n - 1) + record.execution_duration_ms) / n
            )
            self._stats["avg_queue_wait_ms"] = (
                (self._stats["avg_queue_wait_ms"] * (n - 1) + record.queue_wait_ms) / n
            )
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(record)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def add_callback(self, callback: Callable[[ExecutionRecord], None]):
        """Add a callback for new records."""
        self._callbacks.append(callback)
    
    def get_recent(self, n: int = 100) -> List[ExecutionRecord]:
        """Get the most recent n records."""
        records = list(self._buffer)
        return records[-n:]
    
    def get_records_since(self, since: datetime) -> List[ExecutionRecord]:
        """Get records since a given timestamp."""
        return [r for r in self._buffer if r.submitted_at >= since]
    
    def get_training_batch(
        self,
        size: int = 1000,
        completed_only: bool = True,
    ) -> List[ExecutionRecord]:
        """Get a batch of records suitable for training."""
        records = list(self._buffer)
        if completed_only:
            records = [r for r in records if r.status == "completed"]
        return records[-size:]
    
    async def persist(self, force: bool = False):
        """Persist buffer to disk."""
        if not self._buffer or not force:
            return
        
        records = list(self._buffer)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"execution_records_{timestamp}.jsonl"
        
        if self.compress:
            filename += ".gz"
        
        filepath = self.persist_path / filename
        
        try:
            if HAS_AIOFILES:
                if self.compress:
                    data = "\n".join(json.dumps(r.to_dict()) for r in records)
                    compressed = gzip.compress(data.encode())
                    async with aiofiles.open(filepath, "wb") as f:
                        await f.write(compressed)
                else:
                    async with aiofiles.open(filepath, "w") as f:
                        for record in records:
                            await f.write(json.dumps(record.to_dict()) + "\n")
            else:
                # Sync fallback
                if self.compress:
                    with gzip.open(filepath, "wt") as f:
                        for record in records:
                            f.write(json.dumps(record.to_dict()) + "\n")
                else:
                    with open(filepath, "w") as f:
                        for record in records:
                            f.write(json.dumps(record.to_dict()) + "\n")
            
            self._stats["total_persisted"] += len(records)
            logger.info(f"Persisted {len(records)} records to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to persist records: {e}")
    
    async def start_background_persist(self):
        """Start background persistence task."""
        async def _persist_loop():
            while True:
                await asyncio.sleep(self.persist_interval)
                await self.persist()
        
        self._persist_task = asyncio.create_task(_persist_loop())
        logger.info(f"Started background persistence (interval: {self.persist_interval}s)")
    
    async def stop(self):
        """Stop collector and persist remaining data."""
        if self._persist_task:
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass
        
        await self.persist(force=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            **self._stats,
            "buffer_size": len(self._buffer),
            "buffer_capacity": self.buffer_size,
        }
    
    def export_for_training(
        self,
        output_path: Path,
        format: str = "parquet",
    ) -> Path:
        """Export data in training-ready format."""
        import pandas as pd
        
        records = [r.to_dict() for r in self._buffer if r.status == "completed"]
        df = pd.DataFrame(records)
        
        # Convert datetime columns
        for col in ['submitted_at', 'scheduled_at', 'started_at', 'completed_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            filepath = output_path.with_suffix(".parquet")
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            filepath = output_path.with_suffix(".csv")
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Exported {len(df)} records to {filepath}")
        return filepath


# Global collector instance
_collector: Optional[DataCollector] = None


def get_data_collector() -> DataCollector:
    """Get global data collector instance."""
    global _collector
    if _collector is None:
        import os
        _collector = DataCollector(
            buffer_size=int(os.getenv("DATA_BUFFER_SIZE", "10000")),
            persist_path=Path(os.getenv("DATA_PERSIST_PATH", "./data/execution_records")),
        )
    return _collector


async def record_task_execution(
    task_id: str,
    task_name: str,
    priority: int,
    requested_cpu: int,
    requested_memory_mb: int,
    submitted_at: datetime,
    started_at: datetime,
    completed_at: datetime,
    status: str,
    actual_cpu_percent: float = 0.0,
    actual_memory_mb: float = 0.0,
    exit_code: int = 0,
    predicted_duration_ms: Optional[int] = None,
    **kwargs,
):
    """Convenience function to record task execution."""
    collector = get_data_collector()
    
    record = ExecutionRecord(
        task_id=task_id,
        task_name=task_name,
        priority=priority,
        requested_cpu_cores=requested_cpu,
        requested_memory_mb=requested_memory_mb,
        submitted_at=submitted_at,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        actual_cpu_percent=actual_cpu_percent,
        actual_memory_mb=actual_memory_mb,
        exit_code=exit_code,
        predicted_duration_ms=predicted_duration_ms,
        **kwargs,
    )
    
    collector.record(record)
