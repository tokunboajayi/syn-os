"""
Training module initialization.
"""

from .trainer import ModelTrainer, TrainingConfig
from .online import OnlineLearner, OnlineLearningConfig, OnlineLearningManager
from .hyperopt import HyperparameterTuner, TuningConfig, SearchSpace, ModelTuner
from .ab_testing import ABTestManager, ExperimentConfig, ExperimentResult, get_ab_manager
from .retraining import NightlyRetrainer, RetrainingConfig, RetrainingPipelineManager
from .drift import DriftDetector, DriftConfig, DriftReport
from .deployment import BlueGreenDeployer, DeploymentConfig, DeploymentSlot

__all__ = [
    # Core training
    "ModelTrainer",
    "TrainingConfig",
    # Online learning
    "OnlineLearner",
    "OnlineLearningConfig",
    "OnlineLearningManager",
    # Hyperparameter tuning
    "HyperparameterTuner",
    "TuningConfig",
    "SearchSpace",
    "ModelTuner",
    # A/B testing
    "ABTestManager",
    "ExperimentConfig",
    "ExperimentResult",
    "get_ab_manager",
    # Retraining
    "NightlyRetrainer",
    "RetrainingConfig",
    "RetrainingPipelineManager",
    # Drift detection
    "DriftDetector",
    "DriftConfig",
    "DriftReport",
    # Deployment
    "BlueGreenDeployer",
    "DeploymentConfig",
    "DeploymentSlot",
]
