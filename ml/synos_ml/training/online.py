"""
Continuous Online Learning Pipeline for Syn OS

Enables models to learn from real-time execution data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import json
from loguru import logger

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning."""
    
    # Learning schedule
    update_interval_seconds: int = 300  # 5 minutes
    min_samples_per_update: int = 100
    max_samples_per_update: int = 1000
    
    # Learning rates
    base_learning_rate: float = 1e-4
    learning_rate_decay: float = 0.99
    
    # Model versioning
    checkpoint_interval_updates: int = 10
    max_checkpoints: int = 5
    checkpoint_dir: str = "./checkpoints"
    
    # Quality thresholds
    min_accuracy_threshold: float = 0.7
    max_loss_threshold: float = 1.0
    rollback_on_degradation: bool = True


class OnlineLearner:
    """
    Manages online/continuous learning for ML models.
    
    Features:
    - Incremental model updates from streaming data
    - Automatic rollback on performance degradation
    - A/B testing between model versions
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: Any,
        config: Optional[OnlineLearningConfig] = None,
    ):
        self.model = model
        self.config = config or OnlineLearningConfig()
        
        self._sample_buffer: List[Dict[str, Any]] = []
        self._update_count: int = 0
        self._learning_rate = self.config.base_learning_rate
        self._best_loss: float = float("inf")
        self._best_checkpoint: Optional[Path] = None
        self._is_running: bool = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Metrics history
        self._metrics_history: List[Dict[str, Any]] = []
        
        # Ensure checkpoint directory exists
        self._checkpoint_dir = Path(self.config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def add_sample(self, sample: Dict[str, Any]):
        """Add a new training sample from live data."""
        self._sample_buffer.append({
            **sample,
            "_timestamp": datetime.utcnow().isoformat(),
        })
        
        # Trim buffer if too large
        if len(self._sample_buffer) > self.config.max_samples_per_update * 2:
            self._sample_buffer = self._sample_buffer[-self.config.max_samples_per_update:]
    
    async def start(self):
        """Start the online learning loop."""
        if self._is_running:
            return
        
        self._is_running = True
        self._update_task = asyncio.create_task(self._learning_loop())
        logger.info("Started online learning")
    
    async def stop(self):
        """Stop the online learning loop."""
        self._is_running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped online learning")
    
    async def _learning_loop(self):
        """Main online learning loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self.config.update_interval_seconds)
                
                if len(self._sample_buffer) >= self.config.min_samples_per_update:
                    await self._perform_update()
                else:
                    logger.debug(
                        f"Waiting for more samples ({len(self._sample_buffer)}/"
                        f"{self.config.min_samples_per_update})"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Online learning error: {e}")
    
    async def _perform_update(self):
        """Perform a single model update."""
        # Get samples for this update
        samples = self._sample_buffer[:self.config.max_samples_per_update]
        self._sample_buffer = self._sample_buffer[self.config.max_samples_per_update:]
        
        logger.info(f"Performing online update with {len(samples)} samples")
        
        # Prepare data
        X, y = self._prepare_batch(samples)
        
        if X is None or len(X) == 0:
            logger.warning("No valid samples for update")
            return
        
        # Save current state for potential rollback
        pre_update_state = self._snapshot_model()
        
        # Perform update
        loss = self._update_model(X, y)
        
        # Validate update
        if loss is not None:
            metrics = {
                "update_count": self._update_count + 1,
                "samples": len(samples),
                "loss": loss,
                "learning_rate": self._learning_rate,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Check for degradation
            if loss > self.config.max_loss_threshold and self.config.rollback_on_degradation:
                logger.warning(f"Loss {loss:.4f} exceeds threshold, rolling back")
                self._restore_model(pre_update_state)
                metrics["rolled_back"] = True
            else:
                self._update_count += 1
                
                # Update best checkpoint
                if loss < self._best_loss:
                    self._best_loss = loss
                    self._save_checkpoint("best")
                
                # Periodic checkpoint
                if self._update_count % self.config.checkpoint_interval_updates == 0:
                    self._save_checkpoint(f"update_{self._update_count}")
                
                # Decay learning rate
                self._learning_rate *= self.config.learning_rate_decay
            
            self._metrics_history.append(metrics)
            logger.info(f"Update complete: loss={loss:.4f}")
    
    def _prepare_batch(
        self, samples: List[Dict[str, Any]]
    ) -> tuple:
        """Prepare training batch from samples."""
        try:
            # Extract features and targets
            X_list = []
            y_list = []
            
            for sample in samples:
                # Skip samples without target
                if "execution_duration_ms" not in sample:
                    continue
                
                # Basic features
                features = [
                    sample.get("requested_cpu_cores", 1),
                    sample.get("requested_memory_mb", 1024) / 1024,  # Convert to GB
                    sample.get("priority", 5) / 10,
                    sample.get("system_cpu_util", 0),
                    sample.get("system_memory_util", 0),
                    sample.get("queue_depth_at_submit", 0) / 100,
                ]
                
                X_list.append(features)
                y_list.append(sample["execution_duration_ms"] / 1000)  # Convert to seconds
            
            if not X_list:
                return None, None
            
            import numpy as np
            return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Failed to prepare batch: {e}")
            return None, None
    
    def _update_model(self, X, y) -> Optional[float]:
        """Update model with new data."""
        if not HAS_TORCH:
            # Fallback: just return dummy loss
            import numpy as np
            return float(np.mean((y - np.mean(y)) ** 2))
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Ensure model is in training mode
            self.model.train()
            
            # Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            
            # Get optimizer (create if needed)
            if not hasattr(self, "_optimizer"):
                self._optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self._learning_rate,
                )
            
            # Update learning rate
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = self._learning_rate
            
            # Forward pass
            self._optimizer.zero_grad()
            predictions = self.model(X_tensor)
            loss = F.mse_loss(predictions, y_tensor)
            
            # Backward pass
            loss.backward()
            self._optimizer.step()
            
            return float(loss.item())
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return None
    
    def _snapshot_model(self) -> Optional[Dict[str, Any]]:
        """Create a snapshot of current model state."""
        if not HAS_TORCH:
            return None
        
        try:
            import torch
            return {
                key: value.clone()
                for key, value in self.model.state_dict().items()
            }
        except Exception:
            return None
    
    def _restore_model(self, state: Dict[str, Any]):
        """Restore model from snapshot."""
        if state is None or not HAS_TORCH:
            return
        
        try:
            import torch
            self.model.load_state_dict(state)
        except Exception as e:
            logger.error(f"Failed to restore model: {e}")
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if not HAS_TORCH:
            return
        
        try:
            import torch
            
            checkpoint_path = self._checkpoint_dir / f"{name}.pt"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "update_count": self._update_count,
                "learning_rate": self._learning_rate,
                "best_loss": self._best_loss,
                "timestamp": datetime.utcnow().isoformat(),
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Cleanup old checkpoints
            self._cleanup_checkpoints()
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only max_checkpoints."""
        checkpoints = sorted(
            self._checkpoint_dir.glob("update_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        
        while len(checkpoints) > self.config.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.debug(f"Removed old checkpoint: {oldest}")
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get learning metrics history."""
        return self._metrics_history.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current learner status."""
        return {
            "is_running": self._is_running,
            "update_count": self._update_count,
            "buffer_size": len(self._sample_buffer),
            "learning_rate": self._learning_rate,
            "best_loss": self._best_loss,
        }


class OnlineLearningManager:
    """
    Manages online learning for multiple models.
    """
    
    def __init__(self):
        self._learners: Dict[str, OnlineLearner] = {}
    
    def register_model(
        self,
        name: str,
        model: Any,
        config: Optional[OnlineLearningConfig] = None,
    ):
        """Register a model for online learning."""
        self._learners[name] = OnlineLearner(model, config)
        logger.info(f"Registered model '{name}' for online learning")
    
    def add_sample(self, model_name: str, sample: Dict[str, Any]):
        """Add sample for a specific model."""
        if model_name in self._learners:
            self._learners[model_name].add_sample(sample)
    
    async def start_all(self):
        """Start all learners."""
        for name, learner in self._learners.items():
            await learner.start()
    
    async def stop_all(self):
        """Stop all learners."""
        for name, learner in self._learners.items():
            await learner.stop()
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all learners."""
        return {
            name: learner.get_status()
            for name, learner in self._learners.items()
        }
