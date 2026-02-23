"""
Model Serving module initialization.
"""

from .client import InferenceClient, PredictionResult, predict_execution_time

__all__ = [
    "InferenceClient",
    "PredictionResult",
    "predict_execution_time",
]
