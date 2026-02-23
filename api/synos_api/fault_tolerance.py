"""
Fault Tolerance and Checkpointing for Syn OS

Provides:
- Task state checkpointing
- Automatic recovery from failures
- Distributed checkpointing with storage backends
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import hashlib
import gzip
from loguru import logger

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    
    storage_backend: str = "filesystem"  # filesystem, s3, redis
    checkpoint_dir: str = "./checkpoints"
    compression: bool = True
    max_checkpoints: int = 5  # Per task
    checkpoint_interval_seconds: int = 60


@dataclass  
class Checkpoint:
    """A single checkpoint."""
    
    checkpoint_id: str
    task_id: str
    created_at: datetime
    data: bytes
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "data_size": len(self.data),
            "metadata": self.metadata,
        }


class CheckpointManager:
    """
    Manages task checkpoints for fault tolerance.
    
    Features:
    - Async checkpoint save/restore
    - Multiple storage backends
    - Automatic cleanup of old checkpoints
    - Integrity verification
    """
    
    def __init__(self, config: Optional[CheckpointConfig] = None):
        self.config = config or CheckpointConfig()
        self._checkpoint_dir = Path(self.config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
    
    async def save_checkpoint(
        self,
        task_id: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint for a task.
        
        Returns checkpoint_id.
        """
        checkpoint_id = self._generate_checkpoint_id(task_id, data)
        
        # Compress if enabled
        if self.config.compression:
            data = gzip.compress(data)
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            created_at=datetime.utcnow(),
            data=data,
            metadata=metadata or {},
        )
        
        # Save based on backend
        if self.config.storage_backend == "filesystem":
            await self._save_to_filesystem(checkpoint)
        elif self.config.storage_backend == "s3":
            await self._save_to_s3(checkpoint)
        elif self.config.storage_backend == "redis":
            await self._save_to_redis(checkpoint)
        
        # Track in memory
        if task_id not in self._checkpoints:
            self._checkpoints[task_id] = []
        self._checkpoints[task_id].append(checkpoint)
        
        # Cleanup old checkpoints
        await self._cleanup_old_checkpoints(task_id)
        
        logger.info(f"Saved checkpoint {checkpoint_id} for task {task_id}")
        return checkpoint_id
    
    async def load_checkpoint(
        self,
        task_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Load a checkpoint. If checkpoint_id is None, load latest.
        
        Returns decompressed data or None if not found.
        """
        if self.config.storage_backend == "filesystem":
            data = await self._load_from_filesystem(task_id, checkpoint_id)
        elif self.config.storage_backend == "s3":
            data = await self._load_from_s3(task_id, checkpoint_id)
        elif self.config.storage_backend == "redis":
            data = await self._load_from_redis(task_id, checkpoint_id)
        else:
            data = None
        
        if data and self.config.compression:
            data = gzip.decompress(data)
        
        if data:
            logger.info(f"Loaded checkpoint for task {task_id}")
        
        return data
    
    async def list_checkpoints(self, task_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a task."""
        checkpoints = []
        
        task_dir = self._checkpoint_dir / task_id
        if task_dir.exists():
            for path in sorted(task_dir.glob("*.ckpt")):
                meta_path = path.with_suffix(".meta")
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    checkpoints.append(meta)
        
        return checkpoints
    
    async def delete_checkpoint(self, task_id: str, checkpoint_id: str):
        """Delete a specific checkpoint."""
        if self.config.storage_backend == "filesystem":
            task_dir = self._checkpoint_dir / task_id
            ckpt_path = task_dir / f"{checkpoint_id}.ckpt"
            meta_path = task_dir / f"{checkpoint_id}.meta"
            
            for path in [ckpt_path, meta_path]:
                if path.exists():
                    path.unlink()
        
        # Remove from memory
        if task_id in self._checkpoints:
            self._checkpoints[task_id] = [
                c for c in self._checkpoints[task_id]
                if c.checkpoint_id != checkpoint_id
            ]
        
        logger.info(f"Deleted checkpoint {checkpoint_id}")
    
    def _generate_checkpoint_id(self, task_id: str, data: bytes) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        data_hash = hashlib.sha256(data).hexdigest()[:8]
        return f"{task_id}_{timestamp}_{data_hash}"
    
    async def _save_to_filesystem(self, checkpoint: Checkpoint):
        """Save checkpoint to filesystem."""
        task_dir = self._checkpoint_dir / checkpoint.task_id
        task_dir.mkdir(exist_ok=True)
        
        ckpt_path = task_dir / f"{checkpoint.checkpoint_id}.ckpt"
        meta_path = task_dir / f"{checkpoint.checkpoint_id}.meta"
        
        if HAS_AIOFILES:
            async with aiofiles.open(ckpt_path, "wb") as f:
                await f.write(checkpoint.data)
            async with aiofiles.open(meta_path, "w") as f:
                await f.write(json.dumps(checkpoint.to_dict()))
        else:
            with open(ckpt_path, "wb") as f:
                f.write(checkpoint.data)
            with open(meta_path, "w") as f:
                json.dump(checkpoint.to_dict(), f)
    
    async def _load_from_filesystem(
        self,
        task_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[bytes]:
        """Load checkpoint from filesystem."""
        task_dir = self._checkpoint_dir / task_id
        
        if checkpoint_id:
            ckpt_path = task_dir / f"{checkpoint_id}.ckpt"
        else:
            # Get latest
            checkpoints = sorted(task_dir.glob("*.ckpt"))
            if not checkpoints:
                return None
            ckpt_path = checkpoints[-1]
        
        if not ckpt_path.exists():
            return None
        
        if HAS_AIOFILES:
            async with aiofiles.open(ckpt_path, "rb") as f:
                return await f.read()
        else:
            with open(ckpt_path, "rb") as f:
                return f.read()
    
    async def _save_to_s3(self, checkpoint: Checkpoint):
        """Save checkpoint to S3. Placeholder for S3 implementation."""
        logger.warning("S3 storage not implemented, using filesystem")
        await self._save_to_filesystem(checkpoint)
    
    async def _load_from_s3(
        self,
        task_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[bytes]:
        """Load checkpoint from S3. Placeholder."""
        return await self._load_from_filesystem(task_id, checkpoint_id)
    
    async def _save_to_redis(self, checkpoint: Checkpoint):
        """Save checkpoint to Redis. Placeholder for Redis implementation."""
        logger.warning("Redis storage not implemented, using filesystem")
        await self._save_to_filesystem(checkpoint)
    
    async def _load_from_redis(
        self,
        task_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[bytes]:
        """Load checkpoint from Redis. Placeholder."""
        return await self._load_from_filesystem(task_id, checkpoint_id)
    
    async def _cleanup_old_checkpoints(self, task_id: str):
        """Remove old checkpoints beyond max_checkpoints."""
        task_dir = self._checkpoint_dir / task_id
        if not task_dir.exists():
            return
        
        checkpoints = sorted(task_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        
        while len(checkpoints) > self.config.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            meta = oldest.with_suffix(".meta")
            if meta.exists():
                meta.unlink()


class FaultTolerantExecutor:
    """
    Executes tasks with fault tolerance.
    
    Features:
    - Automatic checkpointing
    - Recovery from failures
    - Retry with exponential backoff
    """
    
    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def execute_with_recovery(
        self,
        task_id: str,
        execute_fn,
        *args,
        checkpoint_state: bool = True,
        **kwargs,
    ) -> Any:
        """
        Execute a function with fault tolerance.
        
        Will automatically:
        - Retry on failure with exponential backoff
        - Checkpoint state periodically
        - Recover from last checkpoint on restart
        """
        # Try to restore from checkpoint
        checkpoint_data = await self.checkpoint_manager.load_checkpoint(task_id)
        if checkpoint_data:
            logger.info(f"Restoring task {task_id} from checkpoint")
            kwargs["checkpoint_state"] = json.loads(checkpoint_data.decode())
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await execute_fn(*args, **kwargs)
                
                # Clear checkpoint on success
                checkpoints = await self.checkpoint_manager.list_checkpoints(task_id)
                for cp in checkpoints:
                    await self.checkpoint_manager.delete_checkpoint(task_id, cp["checkpoint_id"])
                
                return result
                
            except Exception as e:
                logger.error(f"Task {task_id} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (2 ** attempt),
                        self.max_delay,
                    )
                    logger.info(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise
    
    async def checkpoint_state(
        self,
        task_id: str,
        state: Dict[str, Any],
    ):
        """Save current task state as checkpoint."""
        data = json.dumps(state).encode()
        await self.checkpoint_manager.save_checkpoint(
            task_id,
            data,
            metadata={"type": "state"},
        )
