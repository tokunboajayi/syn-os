"""
A/B Testing Framework for Syn OS ML Models

Enables safe comparison of model versions in production.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import random
import json
from collections import defaultdict
from loguru import logger

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TrafficSplit(Enum):
    """Traffic splitting strategies."""
    RANDOM = "random"
    STICKY = "sticky"  # Same user always sees same variant
    ROUND_ROBIN = "round_robin"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""
    
    name: str
    control_model: str  # Model A (current)
    treatment_model: str  # Model B (new)
    
    # Traffic allocation
    treatment_fraction: float = 0.1  # Start with 10% traffic
    traffic_strategy: TrafficSplit = TrafficSplit.RANDOM
    
    # Duration
    min_samples: int = 1000
    max_duration_hours: int = 168  # 1 week
    
    # Metrics
    primary_metric: str = "latency_ms"
    secondary_metrics: List[str] = field(default_factory=lambda: ["accuracy", "throughput"])
    
    # Significance
    confidence_level: float = 0.95
    min_effect_size: float = 0.05  # 5% improvement
    
    # Safety
    max_error_rate: float = 0.1
    auto_rollback: bool = True


@dataclass
class ExperimentResult:
    """Results from an A/B experiment."""
    
    experiment_name: str
    status: str  # running, completed, rolled_back
    
    # Sample counts
    control_samples: int = 0
    treatment_samples: int = 0
    
    # Primary metric
    control_mean: float = 0.0
    treatment_mean: float = 0.0
    relative_improvement: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    
    # Secondary metrics
    secondary_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Timing
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    # Recommendation
    recommendation: str = "continue"  # continue, promote, rollback


class ABTestManager:
    """
    Manages A/B testing for ML models.
    
    Features:
    - Traffic splitting with configurable strategies
    - Statistical significance testing
    - Automatic rollback on errors
    - Progressive rollout support
    """
    
    def __init__(self):
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._results: Dict[str, ExperimentResult] = {}
        self._metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._sticky_assignments: Dict[str, str] = {}
        self._call_count: Dict[str, int] = defaultdict(int)
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B experiment."""
        self._experiments[config.name] = config
        self._results[config.name] = ExperimentResult(
            experiment_name=config.name,
            status="running",
            started_at=datetime.utcnow(),
        )
        
        logger.info(
            f"Created experiment '{config.name}': "
            f"{config.control_model} vs {config.treatment_model} "
            f"({config.treatment_fraction:.0%} treatment)"
        )
        
        return config.name
    
    def get_variant(
        self,
        experiment_name: str,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Get which model variant to use.
        
        Args:
            experiment_name: Name of the experiment
            user_id: Optional user ID for sticky assignments
            
        Returns:
            Model name to use (control or treatment)
        """
        if experiment_name not in self._experiments:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        config = self._experiments[experiment_name]
        result = self._results[experiment_name]
        
        # Check if experiment is still running
        if result.status != "running":
            return config.control_model
        
        # Determine variant based on strategy
        if config.traffic_strategy == TrafficSplit.STICKY and user_id:
            key = f"{experiment_name}:{user_id}"
            if key not in self._sticky_assignments:
                is_treatment = random.random() < config.treatment_fraction
                self._sticky_assignments[key] = (
                    config.treatment_model if is_treatment else config.control_model
                )
            return self._sticky_assignments[key]
        
        elif config.traffic_strategy == TrafficSplit.ROUND_ROBIN:
            self._call_count[experiment_name] += 1
            n = self._call_count[experiment_name]
            is_treatment = (n % int(1 / config.treatment_fraction)) == 0
            
        else:  # Random
            is_treatment = random.random() < config.treatment_fraction
        
        return config.treatment_model if is_treatment else config.control_model
    
    def record_metric(
        self,
        experiment_name: str,
        variant: str,
        metric_name: str,
        value: float,
    ):
        """Record a metric observation."""
        if experiment_name not in self._experiments:
            return
        
        config = self._experiments[experiment_name]
        result = self._results[experiment_name]
        
        if result.status != "running":
            return
        
        # Determine group
        if variant == config.control_model:
            group = "control"
            result.control_samples += 1
        else:
            group = "treatment"
            result.treatment_samples += 1
        
        # Store metric
        key = f"{experiment_name}:{group}"
        self._metrics[key][metric_name].append(value)
        
        # Check for auto-rollback
        if config.auto_rollback and metric_name == "error":
            error_rate = sum(self._metrics[key]["error"]) / len(self._metrics[key]["error"])
            if error_rate > config.max_error_rate:
                logger.warning(f"Auto-rollback triggered for {experiment_name}")
                self._rollback(experiment_name)
        
        # Check if we can conclude
        self._check_completion(experiment_name)
    
    def _check_completion(self, experiment_name: str):
        """Check if experiment can be concluded."""
        config = self._experiments[experiment_name]
        result = self._results[experiment_name]
        
        # Check sample counts
        if (result.control_samples < config.min_samples or
            result.treatment_samples < config.min_samples):
            return
        
        # Check duration
        elapsed = datetime.utcnow() - result.started_at
        if elapsed > timedelta(hours=config.max_duration_hours):
            self._conclude(experiment_name, reason="max_duration")
            return
        
        # Run statistical test
        control_key = f"{experiment_name}:control"
        treatment_key = f"{experiment_name}:treatment"
        
        primary = config.primary_metric
        control_values = self._metrics[control_key][primary]
        treatment_values = self._metrics[treatment_key][primary]
        
        if len(control_values) < 30 or len(treatment_values) < 30:
            return
        
        # Calculate statistics
        import numpy as np
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        
        result.control_mean = control_mean
        result.treatment_mean = treatment_mean
        
        if control_mean > 0:
            result.relative_improvement = (control_mean - treatment_mean) / control_mean
        
        # Statistical significance test
        if HAS_SCIPY:
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            result.p_value = p_value
            result.is_significant = p_value < (1 - config.confidence_level)
        else:
            # Simplified significance check without scipy
            result.is_significant = abs(result.relative_improvement) > config.min_effect_size
        
        # Update recommendation
        if result.is_significant:
            if result.relative_improvement > config.min_effect_size:
                result.recommendation = "promote"
            elif result.relative_improvement < -config.min_effect_size:
                result.recommendation = "rollback"
            else:
                result.recommendation = "continue"
        
        # Auto-conclude if highly significant
        if result.is_significant and result.p_value < 0.01:
            self._conclude(experiment_name, reason="significance")
    
    def _conclude(self, experiment_name: str, reason: str):
        """Conclude an experiment."""
        result = self._results[experiment_name]
        result.status = "completed"
        result.ended_at = datetime.utcnow()
        
        logger.info(
            f"Experiment '{experiment_name}' concluded ({reason}): "
            f"recommendation={result.recommendation}, "
            f"improvement={result.relative_improvement:.2%}"
        )
    
    def _rollback(self, experiment_name: str):
        """Rollback an experiment."""
        result = self._results[experiment_name]
        result.status = "rolled_back"
        result.recommendation = "rollback"
        result.ended_at = datetime.utcnow()
        
        logger.warning(f"Experiment '{experiment_name}' rolled back")
    
    def get_result(self, experiment_name: str) -> ExperimentResult:
        """Get current experiment results."""
        if experiment_name not in self._results:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        return self._results[experiment_name]
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get summary of all experiments."""
        return [
            {
                "name": name,
                "status": self._results[name].status,
                "control": config.control_model,
                "treatment": config.treatment_model,
                "samples": (
                    self._results[name].control_samples +
                    self._results[name].treatment_samples
                ),
                "recommendation": self._results[name].recommendation,
            }
            for name, config in self._experiments.items()
        ]
    
    def promote_winner(self, experiment_name: str) -> str:
        """
        Promote the winning variant.
        
        Returns the name of the promoted model.
        """
        result = self._results[experiment_name]
        config = self._experiments[experiment_name]
        
        if result.recommendation == "promote":
            promoted = config.treatment_model
            logger.info(f"Promoted treatment model: {promoted}")
        else:
            promoted = config.control_model
            logger.info(f"Kept control model: {promoted}")
        
        # Clean up experiment
        result.status = "completed"
        
        return promoted
    
    def increase_traffic(
        self,
        experiment_name: str,
        new_fraction: float,
    ):
        """Increase traffic to treatment."""
        if experiment_name not in self._experiments:
            return
        
        config = self._experiments[experiment_name]
        old_fraction = config.treatment_fraction
        config.treatment_fraction = min(new_fraction, 1.0)
        
        logger.info(
            f"Increased '{experiment_name}' traffic: "
            f"{old_fraction:.0%} -> {config.treatment_fraction:.0%}"
        )


# Global A/B test manager
_ab_manager: Optional[ABTestManager] = None


def get_ab_manager() -> ABTestManager:
    """Get global A/B test manager."""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestManager()
    return _ab_manager
