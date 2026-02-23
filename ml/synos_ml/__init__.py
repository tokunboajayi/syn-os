"""
Syn OS Machine Learning Engine

Neural network powered optimization for task scheduling, 
resource forecasting, and anomaly detection.
"""

__version__ = "0.1.0"
__author__ = "Syn OS Team"

from synos_ml.models.predictor import ExecutionTimePredictor
from synos_ml.models.forecaster import TransformerLSTMHybrid
from synos_ml.models.anomaly import AnomalyDetector
# from synos_ml.models.gnn import TaskGNN, TaskGraphBuilder
from synos_ml.models.trajectory import TrajectoryPredictor

__all__ = [
    "ExecutionTimePredictor",
    "TransformerLSTMHybrid",
    "AnomalyDetector",
    # "TaskGNN",
    # "TaskGraphBuilder",
    "PPOScheduler",
    "PPOActorCritic",
    "TrajectoryPredictor",
]
