"""
Execution Time Predictor using XGBoost + Neural Ensemble

Predicts task execution duration based on:
- Task characteristics (CPU, memory, I/O requirements)
- System state (current load, queue depth)
- Historical execution patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# XGBoost is optional
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed, using neural-only predictor")


@dataclass
class TaskPredictionFeatures:
    """Features used for execution time prediction."""

    # Task characteristics
    cpu_cores: int
    memory_mb: int
    gpu_memory_mb: int = 0
    priority: int = 5
    
    # Command characteristics
    command_hash: int = 0  # Hash of command for grouping similar tasks
    
    # System state
    current_cpu_util: float = 0.0
    current_memory_util: float = 0.0
    queue_depth: int = 0
    running_tasks: int = 0
    
    # Temporal features
    hour_of_day: int = 0
    day_of_week: int = 0
    is_peak_hours: bool = False

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.cpu_cores,
            self.memory_mb / 1000,  # Normalize to GB
            self.gpu_memory_mb / 1000,
            self.priority / 10,
            self.command_hash % 1000 / 1000,  # Normalize hash
            self.current_cpu_util,
            self.current_memory_util,
            self.queue_depth / 100,
            self.running_tasks / 100,
            self.hour_of_day / 24,
            self.day_of_week / 7,
            float(self.is_peak_hours),
        ], dtype=np.float32)


class NeuralPredictor(nn.Module):
    """Neural network for execution time prediction."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # Mean and log-variance
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            mean: Predicted mean duration
            log_var: Predicted log variance (for uncertainty)
        """
        out = self.network(x)
        mean = F.softplus(out[:, 0])  # Ensure positive
        log_var = out[:, 1]
        return mean, log_var


# Need to import F after nn
import torch.nn.functional as F


class ExecutionTimePredictor:
    """
    Ensemble predictor combining XGBoost and Neural Network.

    Uses XGBoost for fast, interpretable predictions and Neural Network
    for capturing complex patterns. Final prediction is weighted average.
    """

    def __init__(
        self,
        xgb_model_path: Optional[str] = None,
        neural_model_path: Optional[str] = None,
        xgb_weight: float = 0.6,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.xgb_weight = xgb_weight
        self.neural_weight = 1 - xgb_weight

        # XGBoost model
        self.xgb_model = None
        if HAS_XGBOOST:
            if xgb_model_path:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_model_path)
                logger.info(f"Loaded XGBoost model from {xgb_model_path}")
            else:
                # Initialize with default parameters for training
                self.xgb_model = None  # Will be trained

        # Neural model
        self.neural_model = NeuralPredictor()
        if neural_model_path:
            self.neural_model.load_state_dict(
                torch.load(neural_model_path, map_location=self.device)
            )
            logger.info(f"Loaded neural model from {neural_model_path}")
        self.neural_model.to(self.device)
        self.neural_model.eval()

    def predict(
        self,
        features: Union[TaskPredictionFeatures, np.ndarray, List[TaskPredictionFeatures]],
        return_uncertainty: bool = False,
    ) -> Union[float, Tuple[float, float], np.ndarray]:
        """
        Predict execution time.

        Args:
            features: Task features (single or batch)
            return_uncertainty: Whether to return prediction uncertainty

        Returns:
            Predicted execution time in milliseconds (and optionally std)
        """
        # Handle different input types
        if isinstance(features, TaskPredictionFeatures):
            x = features.to_array().reshape(1, -1)
            single = True
        elif isinstance(features, list):
            x = np.array([f.to_array() for f in features])
            single = False
        else:
            x = features if features.ndim == 2 else features.reshape(1, -1)
            single = x.shape[0] == 1

        predictions = []
        uncertainties = []

        # XGBoost prediction
        if HAS_XGBOOST and self.xgb_model is not None:
            dmat = xgb.DMatrix(x)
            xgb_pred = self.xgb_model.predict(dmat)
            predictions.append(xgb_pred * self.xgb_weight)
        else:
            self.xgb_weight = 0
            self.neural_weight = 1.0

        # Neural prediction
        self.neural_model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            # Handle batch norm with single sample
            if x_tensor.shape[0] == 1:
                self.neural_model.network.eval()
            mean, log_var = self.neural_model(x_tensor)
            neural_pred = mean.cpu().numpy()
            neural_std = torch.exp(0.5 * log_var).cpu().numpy()
            predictions.append(neural_pred * self.neural_weight)
            uncertainties.append(neural_std)

        # Combine predictions
        final_pred = sum(predictions)
        final_std = uncertainties[0] if uncertainties else np.zeros_like(final_pred)

        # Convert to ms (model predicts in seconds)
        final_pred_ms = final_pred * 1000
        final_std_ms = final_std * 1000

        if single:
            if return_uncertainty:
                return float(final_pred_ms[0]), float(final_std_ms[0])
            return float(final_pred_ms[0])
        else:
            if return_uncertainty:
                return final_pred_ms, final_std_ms
            return final_pred_ms

    def train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Optional[Dict] = None,
    ):
        """
        Train the XGBoost model.

        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target values (execution times in seconds)
            params: XGBoost parameters
        """
        if not HAS_XGBOOST:
            logger.warning("XGBoost not available, skipping training")
            return

        if params is None:
            params = {
                "max_depth": 6,
                "eta": 0.1,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
            }

        dtrain = xgb.DMatrix(X, label=y)
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            verbose_eval=False,
        )
        logger.info("Trained XGBoost model")

    def train_neural(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        """
        Train the neural model.

        Args:
            X: Feature matrix
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        self.neural_model.train()
        optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                mean, log_var = self.neural_model(batch_x)

                # NLL loss with learned variance
                var = torch.exp(log_var)
                loss = 0.5 * (log_var + (mean - batch_y) ** 2 / var).mean()

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

        self.neural_model.eval()
        logger.info("Trained neural model")

    def save(self, xgb_path: str, neural_path: str):
        """Save both models."""
        if HAS_XGBOOST and self.xgb_model is not None:
            self.xgb_model.save_model(xgb_path)
        torch.save(self.neural_model.state_dict(), neural_path)
        logger.info(f"Saved models to {xgb_path}, {neural_path}")

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Returns:
            Dict with MAE, RMSE, MAPE, R2
        """
        predictions = self.predict(X)

        mae = np.mean(np.abs(predictions - y * 1000))  # Convert y to ms
        rmse = np.sqrt(np.mean((predictions - y * 1000) ** 2))
        mape = np.mean(np.abs((predictions - y * 1000) / (y * 1000 + 1e-8))) * 100
        
        # R2
        ss_res = np.sum((y * 1000 - predictions) ** 2)
        ss_tot = np.sum((y * 1000 - np.mean(y * 1000)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        return {
            "mae_ms": float(mae),
            "rmse_ms": float(rmse),
            "mape_percent": float(mape),
            "r2": float(r2),
        }
