"""
Nightly Retraining Pipeline for Syn OS

Automated scheduled model retraining with:
- Data collection and preprocessing
- Model training and validation
- Performance comparison
- Deployment decision
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import json
from loguru import logger

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False


@dataclass
class RetrainingConfig:
    """Configuration for nightly retraining."""
    
    # Schedule
    cron_expression: str = "0 2 * * *"  # 2 AM daily
    timezone: str = "UTC"
    
    # Data
    data_lookback_days: int = 7  # Use last 7 days of data
    min_samples: int = 1000
    validation_split: float = 0.2
    
    # Training
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Validation
    min_improvement: float = 0.01  # 1% improvement required
    max_regression: float = 0.05  # 5% regression allowed
    
    # Deployment
    auto_deploy: bool = False  # Require approval by default
    blue_green_enabled: bool = True
    
    # Storage
    models_dir: str = "./models"
    checkpoints_dir: str = "./checkpoints"
    logs_dir: str = "./logs"


@dataclass
class RetrainingResult:
    """Result of a retraining run."""
    
    run_id: str
    model_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Data
    samples_used: int = 0
    data_date_range: Optional[Tuple[datetime, datetime]] = None
    
    # Training
    epochs_trained: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    
    # Comparison
    baseline_metric: float = 0.0
    new_metric: float = 0.0
    improvement: float = 0.0
    
    # Decision
    deploy_approved: bool = False
    deployment_status: str = "pending"  # pending, deployed, rejected
    
    # Artifacts
    model_path: Optional[str] = None
    logs_path: Optional[str] = None


from typing import Tuple


class NightlyRetrainer:
    """
    Manages automated nightly model retraining.
    
    Features:
    - Scheduled execution with cron
    - Automatic data collection
    - Training with early stopping
    - Performance validation
    - Blue-green deployment integration
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[RetrainingConfig] = None,
        train_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
    ):
        self.model_name = model_name
        self.config = config or RetrainingConfig()
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        
        self._scheduler = None
        self._history: List[RetrainingResult] = []
        self._current_run: Optional[RetrainingResult] = None
        
        # Ensure directories exist
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)
    
    def start(self):
        """Start the retraining scheduler."""
        if not HAS_SCHEDULER:
            logger.warning("APScheduler not installed, using manual triggering")
            return
        
        self._scheduler = AsyncIOScheduler()
        trigger = CronTrigger.from_crontab(
            self.config.cron_expression,
            timezone=self.config.timezone,
        )
        
        self._scheduler.add_job(
            self._run_retraining,
            trigger=trigger,
            id=f"retrain_{self.model_name}",
            name=f"Nightly retraining for {self.model_name}",
        )
        
        self._scheduler.start()
        logger.info(f"Started nightly retraining for {self.model_name}: {self.config.cron_expression}")
    
    def stop(self):
        """Stop the scheduler."""
        if self._scheduler:
            self._scheduler.shutdown()
            logger.info(f"Stopped nightly retraining for {self.model_name}")
    
    async def trigger_now(self) -> RetrainingResult:
        """Manually trigger a retraining run."""
        return await self._run_retraining()
    
    async def _run_retraining(self) -> RetrainingResult:
        """Execute a retraining run."""
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        result = RetrainingResult(
            run_id=run_id,
            model_name=self.model_name,
            started_at=datetime.utcnow(),
        )
        self._current_run = result
        
        logger.info(f"Starting retraining run {run_id} for {self.model_name}")
        
        try:
            # 1. Collect data
            data = await self._collect_data(result)
            if data is None:
                result.deployment_status = "failed_data"
                return result
            
            # 2. Train model
            model, train_metrics = await self._train_model(data, result)
            if model is None:
                result.deployment_status = "failed_training"
                return result
            
            # 3. Evaluate
            eval_metrics = await self._evaluate_model(model, data, result)
            
            # 4. Compare with baseline
            should_deploy = self._should_deploy(result)
            result.deploy_approved = should_deploy
            
            # 5. Deploy if approved
            if should_deploy and self.config.auto_deploy:
                await self._deploy_model(model, result)
            
            result.completed_at = datetime.utcnow()
            self._history.append(result)
            
            logger.info(
                f"Retraining run {run_id} complete: "
                f"improvement={result.improvement:.2%}, "
                f"deploy={result.deploy_approved}"
            )
            
        except Exception as e:
            logger.error(f"Retraining run {run_id} failed: {e}")
            result.deployment_status = "failed"
            result.completed_at = datetime.utcnow()
        
        self._current_run = None
        return result
    
    async def _collect_data(
        self, result: RetrainingResult
    ) -> Optional[Dict[str, Any]]:
        """Collect training data."""
        from synos_ml.data import get_data_collector
        
        collector = get_data_collector()
        
        # Get recent records
        since = datetime.utcnow() - timedelta(days=self.config.data_lookback_days)
        records = collector.get_records_since(since)
        
        if len(records) < self.config.min_samples:
            logger.warning(
                f"Insufficient data: {len(records)} < {self.config.min_samples}"
            )
            return None
        
        result.samples_used = len(records)
        result.data_date_range = (
            min(r.submitted_at for r in records),
            max(r.submitted_at for r in records),
        )
        
        # Convert to training format
        from synos_ml.data import FeaturePipeline
        
        pipeline = FeaturePipeline()
        data = pipeline.prepare_training_data(
            [r.to_dict() for r in records],
            test_size=self.config.validation_split,
        )
        
        logger.info(f"Collected {len(records)} samples for training")
        return data
    
    async def _train_model(
        self,
        data: Dict[str, Any],
        result: RetrainingResult,
    ) -> Tuple[Optional[Any], Dict[str, float]]:
        """Train the model."""
        if self.train_fn is None:
            logger.error("No training function provided")
            return None, {}
        
        try:
            model, metrics = await asyncio.to_thread(
                self.train_fn,
                data["X_train"],
                data["y_train"],
                data["X_test"],
                data["y_test"],
                epochs=self.config.max_epochs,
                patience=self.config.early_stopping_patience,
            )
            
            result.epochs_trained = metrics.get("epochs", 0)
            result.final_train_loss = metrics.get("train_loss", 0)
            result.final_val_loss = metrics.get("val_loss", 0)
            
            # Save model
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_path = Path(self.config.checkpoints_dir) / f"{self.model_name}_{timestamp}.pt"
            result.model_path = str(model_path)
            
            # Save via model's save method if available
            if hasattr(model, "save"):
                model.save(model_path)
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None, {}
    
    async def _evaluate_model(
        self,
        model: Any,
        data: Dict[str, Any],
        result: RetrainingResult,
    ) -> Dict[str, float]:
        """Evaluate the trained model."""
        if self.evaluate_fn is None:
            # Default evaluation
            return {}
        
        try:
            metrics = await asyncio.to_thread(
                self.evaluate_fn,
                model,
                data["X_test"],
                data["y_test"],
            )
            
            result.new_metric = metrics.get("primary", 0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def _should_deploy(self, result: RetrainingResult) -> bool:
        """Determine if the new model should be deployed."""
        # Compare with baseline
        if result.baseline_metric == 0:
            # No baseline, first run
            return True
        
        # Calculate improvement
        result.improvement = (result.baseline_metric - result.new_metric) / result.baseline_metric
        
        # Check improvement threshold
        if result.improvement >= self.config.min_improvement:
            return True
        
        # Check regression threshold
        if result.improvement < -self.config.max_regression:
            logger.warning("Model regression detected, not deploying")
            return False
        
        # No significant change
        return False
    
    async def _deploy_model(self, model: Any, result: RetrainingResult):
        """Deploy the model."""
        from synos_ml.training.deployment import BlueGreenDeployer
        
        if self.config.blue_green_enabled:
            deployer = BlueGreenDeployer(self.model_name)
            await deployer.deploy_new_version(model, result.model_path)
        else:
            # Direct deployment
            production_path = Path(self.config.models_dir) / f"{self.model_name}_production.pt"
            if hasattr(model, "save"):
                model.save(production_path)
        
        result.deployment_status = "deployed"
        logger.info(f"Deployed new model version: {result.model_path}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return [
            {
                "run_id": r.run_id,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "samples": r.samples_used,
                "improvement": r.improvement,
                "deployed": r.deployment_status == "deployed",
            }
            for r in self._history
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "model_name": self.model_name,
            "is_running": self._current_run is not None,
            "current_run": self._current_run.run_id if self._current_run else None,
            "total_runs": len(self._history),
            "last_run": self._history[-1].run_id if self._history else None,
            "schedule": self.config.cron_expression,
        }


class RetrainingPipelineManager:
    """Manages retraining pipelines for multiple models."""
    
    def __init__(self):
        self._pipelines: Dict[str, NightlyRetrainer] = {}
    
    def register(
        self,
        model_name: str,
        train_fn: Callable,
        evaluate_fn: Callable,
        config: Optional[RetrainingConfig] = None,
    ):
        """Register a model for nightly retraining."""
        pipeline = NightlyRetrainer(
            model_name=model_name,
            config=config,
            train_fn=train_fn,
            evaluate_fn=evaluate_fn,
        )
        self._pipelines[model_name] = pipeline
        logger.info(f"Registered {model_name} for nightly retraining")
    
    def start_all(self):
        """Start all retraining pipelines."""
        for pipeline in self._pipelines.values():
            pipeline.start()
    
    def stop_all(self):
        """Stop all retraining pipelines."""
        for pipeline in self._pipelines.values():
            pipeline.stop()
    
    async def trigger(self, model_name: str) -> RetrainingResult:
        """Trigger retraining for a specific model."""
        if model_name not in self._pipelines:
            raise ValueError(f"Unknown model: {model_name}")
        
        return await self._pipelines[model_name].trigger_now()
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all pipelines."""
        return {
            name: pipeline.get_status()
            for name, pipeline in self._pipelines.items()
        }
