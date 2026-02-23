"""
Inference Client for Syn OS ML Models

Provides a Python client for making ML predictions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import json
from loguru import logger

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class PredictionResult:
    """Result from ML model prediction."""
    
    model_name: str
    prediction: Any
    confidence: float = 1.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class InferenceClient:
    """
    Client for ML model inference.
    
    Supports:
    - Execution time prediction
    - Resource demand forecasting
    - Anomaly detection
    - Task priority scoring
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 5.0,
        retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._client: Optional["httpx.AsyncClient"] = None
    
    async def __aenter__(self) -> "InferenceClient":
        """Async context manager entry."""
        if HAS_HTTPX:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    async def predict_execution_time(
        self,
        cpu_cores: int,
        memory_mb: int,
        priority: int = 5,
        system_cpu_util: float = 0.0,
        system_memory_util: float = 0.0,
        queue_depth: int = 0,
    ) -> PredictionResult:
        """
        Predict task execution time.
        
        Returns predicted execution duration in milliseconds.
        """
        features = {
            "cpu_cores": cpu_cores,
            "memory_mb": memory_mb,
            "priority": priority,
            "system_cpu_util": system_cpu_util,
            "system_memory_util": system_memory_util,
            "queue_depth": queue_depth,
        }
        
        return await self._predict("execution_predictor", features)
    
    async def forecast_demand(
        self,
        historical_data: List[Dict[str, float]],
        horizon_hours: int = 6,
    ) -> PredictionResult:
        """
        Forecast resource demand.
        
        Args:
            historical_data: List of dicts with cpu, memory, io values
            horizon_hours: How far ahead to forecast
        
        Returns predicted resource demands.
        """
        features = {
            "historical": historical_data,
            "horizon_hours": horizon_hours,
        }
        
        return await self._predict("demand_forecaster", features)
    
    async def detect_anomaly(
        self,
        execution_time_ms: int,
        memory_used_mb: float,
        cpu_percent: float,
        exit_code: int = 0,
        system_cpu_util: float = 0.0,
        system_memory_util: float = 0.0,
    ) -> PredictionResult:
        """
        Detect if execution metrics indicate an anomaly.
        
        Returns anomaly score and classification.
        """
        features = {
            "execution_time_ms": execution_time_ms,
            "memory_used_mb": memory_used_mb,
            "cpu_percent": cpu_percent,
            "exit_code": exit_code,
            "system_cpu_util": system_cpu_util,
            "system_memory_util": system_memory_util,
        }
        
        return await self._predict("anomaly_detector", features)
    
    async def score_priority(
        self,
        task_features: List[float],
        dependencies: List[int],
    ) -> PredictionResult:
        """
        Score task priority using GNN.
        
        Returns priority scores for intelligent scheduling.
        """
        features = {
            "task_features": task_features,
            "dependencies": dependencies,
        }
        
        return await self._predict("task_gnn", features)
    
    async def get_scheduling_action(
        self,
        state: List[float],
        available_resources: List[int],
    ) -> PredictionResult:
        """
        Get scheduling action from RL policy.
        
        Returns recommended resource allocation.
        """
        features = {
            "state": state,
            "available_resources": available_resources,
        }
        
        return await self._predict("ppo_scheduler", features)
    
    async def _predict(
        self,
        model_name: str,
        features: Dict[str, Any],
    ) -> PredictionResult:
        """Make prediction request to ML service."""
        import time
        start = time.perf_counter()
        
        if not HAS_HTTPX or not self._client:
            # Fallback to dummy prediction
            logger.debug(f"HTTP client not available, returning dummy prediction for {model_name}")
            return PredictionResult(
                model_name=model_name,
                prediction=self._get_fallback_prediction(model_name),
                confidence=0.5,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        
        for attempt in range(self.retries):
            try:
                response = await self._client.post(
                    "/predict",
                    json={
                        "model_name": model_name,
                        "features": features,
                    },
                )
                response.raise_for_status()
                data = response.json()
                
                return PredictionResult(
                    model_name=model_name,
                    prediction=data.get("prediction"),
                    confidence=data.get("confidence", 1.0),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
                
            except Exception as e:
                logger.warning(f"Prediction attempt {attempt + 1} failed: {e}")
                if attempt == self.retries - 1:
                    logger.error(f"All prediction attempts failed for {model_name}")
                    return PredictionResult(
                        model_name=model_name,
                        prediction=self._get_fallback_prediction(model_name),
                        confidence=0.0,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        metadata={"error": str(e)},
                    )
                await asyncio.sleep(0.1 * (attempt + 1))
        
        # Should not reach here
        return PredictionResult(
            model_name=model_name,
            prediction=None,
            confidence=0.0,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    
    def _get_fallback_prediction(self, model_name: str) -> Any:
        """Get fallback prediction when service unavailable."""
        fallbacks = {
            "execution_predictor": 1000,  # 1 second default
            "demand_forecaster": {"cpu": 0.5, "memory": 0.5, "io": 0.3},
            "anomaly_detector": {"is_anomaly": False, "score": 0.0},
            "task_gnn": {"priority_score": 0.5},
            "ppo_scheduler": {"action": 0, "confidence": 0.0},
        }
        return fallbacks.get(model_name, None)
    
    async def health_check(self) -> bool:
        """Check if ML service is healthy."""
        if not HAS_HTTPX or not self._client:
            return False
        
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        if not HAS_HTTPX or not self._client:
            return []
        
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


# Convenience function for one-off predictions
async def predict_execution_time(
    cpu_cores: int,
    memory_mb: int,
    **kwargs,
) -> int:
    """Quick execution time prediction."""
    async with InferenceClient() as client:
        result = await client.predict_execution_time(
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            **kwargs,
        )
        return result.prediction
