"""
Anomaly Detection for Syn OS

Ensemble of Isolation Forest + Autoencoder for detecting:
1. Unusual task execution patterns
2. Resource usage anomalies
3. System health issues
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# Sklearn is optional but commonly available
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed, using autoencoder-only detector")


class Autoencoder(nn.Module):
    """
    Autoencoder for anomaly detection.

    Reconstructs normal patterns; high reconstruction error indicates anomaly.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: List[int] = None,
        latent_dim: int = 4,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 16]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
            ])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            reconstruction: Reconstructed input
            latent: Latent representation
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error (MSE)."""
        recon, _ = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)


@dataclass
class AnomalyFeatures:
    """Features for anomaly detection."""

    # Task metrics
    execution_time_ms: float
    memory_used_mb: float
    cpu_time_ms: float
    exit_code: int

    # System metrics
    system_cpu_util: float
    system_memory_util: float
    system_io_util: float
    
    # Queue metrics
    queue_depth: int
    active_tasks: int
    
    # Rates
    task_arrival_rate: float  # tasks/min
    task_completion_rate: float  # tasks/min
    error_rate: float  # 0-1

    # Deviation from predictions
    duration_prediction_error: float = 0.0  # (actual - predicted) / predicted
    memory_prediction_error: float = 0.0

    # Temporal
    time_since_last_anomaly_mins: float = 60.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            np.log1p(self.execution_time_ms),  # Log transform for scale
            np.log1p(self.memory_used_mb),
            np.log1p(self.cpu_time_ms),
            float(self.exit_code != 0),
            self.system_cpu_util,
            self.system_memory_util,
            self.system_io_util,
            np.log1p(self.queue_depth),
            np.log1p(self.active_tasks),
            np.log1p(self.task_arrival_rate),
            np.log1p(self.task_completion_rate),
            self.error_rate,
            self.duration_prediction_error,
            self.memory_prediction_error,
            np.log1p(self.time_since_last_anomaly_mins),
        ], dtype=np.float32)


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    is_anomaly: bool
    anomaly_score: float  # Higher = more anomalous
    confidence: float  # 0-1
    anomaly_type: Optional[str] = None
    details: Optional[Dict] = None


class AnomalyDetector:
    """
    Ensemble anomaly detector combining Isolation Forest and Autoencoder.

    The ensemble approach provides:
    - Isolation Forest: Fast, effective for high-dimensional data
    - Autoencoder: Learns complex patterns, provides reconstruction-based score
    """

    def __init__(
        self,
        input_dim: int = 15,
        contamination: float = 0.01,  # Expected anomaly rate
        autoencoder_path: Optional[str] = None,
        isolation_forest_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.contamination = contamination
        
        # Thresholds (will be calibrated during training)
        self.if_threshold = 0.0
        self.ae_threshold = 0.1
        
        # Scaler for normalization
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_fitted = False

        # Isolation Forest
        self.isolation_forest = None
        if HAS_SKLEARN:
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )

        # Autoencoder
        self.autoencoder = Autoencoder(input_dim=input_dim)
        if autoencoder_path:
            self.autoencoder.load_state_dict(
                torch.load(autoencoder_path, map_location=self.device)
            )
            logger.info(f"Loaded autoencoder from {autoencoder_path}")
        self.autoencoder.to(self.device)
        self.autoencoder.eval()

    def fit(self, X: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """
        Fit the detector on normal data.

        Args:
            X: Training data (normal samples) [n_samples, n_features]
            epochs: Autoencoder training epochs
            batch_size: Batch size for autoencoder
        """
        logger.info(f"Fitting anomaly detector on {X.shape[0]} samples")

        # Fit scaler
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Fit Isolation Forest
        if self.isolation_forest is not None:
            self.isolation_forest.fit(X_scaled)
            # Compute threshold
            scores = self.isolation_forest.decision_function(X_scaled)
            self.if_threshold = np.percentile(scores, self.contamination * 100)
            logger.info(f"Isolation Forest threshold: {self.if_threshold:.4f}")

        # Train Autoencoder
        self._train_autoencoder(X_scaled, epochs, batch_size)
        
        # Compute autoencoder threshold
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            errors = self.autoencoder.reconstruction_error(X_tensor).cpu().numpy()
            self.ae_threshold = np.percentile(errors, (1 - self.contamination) * 100)
            logger.info(f"Autoencoder threshold: {self.ae_threshold:.4f}")

        self.is_fitted = True
        logger.info("Anomaly detector fitted")

    def _train_autoencoder(
        self,
        X: np.ndarray,
        epochs: int,
        batch_size: int,
    ):
        """Train the autoencoder."""
        self.autoencoder.train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                recon, _ = self.autoencoder(batch_x)
                loss = nn.functional.mse_loss(recon, batch_x)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.debug(f"AE Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

        self.autoencoder.eval()

    def detect(
        self,
        features: Union[AnomalyFeatures, np.ndarray],
        return_details: bool = False,
    ) -> AnomalyResult:
        """
        Detect if input is anomalous.

        Args:
            features: Input features
            return_details: Include detailed scores in result

        Returns:
            AnomalyResult with detection outcome
        """
        # Convert to array if needed
        if isinstance(features, AnomalyFeatures):
            x = features.to_array().reshape(1, -1)
        else:
            x = features.reshape(1, -1) if features.ndim == 1 else features

        # Scale
        if self.scaler is not None and self.is_fitted:
            x_scaled = self.scaler.transform(x)
        else:
            x_scaled = x

        scores = {}
        is_anomaly_votes = []

        # Isolation Forest score
        if self.isolation_forest is not None and self.is_fitted:
            if_score = self.isolation_forest.decision_function(x_scaled)[0]
            is_anomaly_if = if_score < self.if_threshold
            scores["isolation_forest"] = float(if_score)
            is_anomaly_votes.append(is_anomaly_if)

        # Autoencoder score
        self.autoencoder.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)
            ae_error = self.autoencoder.reconstruction_error(x_tensor).cpu().numpy()[0]
            is_anomaly_ae = ae_error > self.ae_threshold
            scores["autoencoder"] = float(ae_error)
            is_anomaly_votes.append(is_anomaly_ae)

        # Ensemble decision (majority vote)
        is_anomaly = sum(is_anomaly_votes) > len(is_anomaly_votes) / 2

        # Compute combined score (normalized)
        if scores:
            # Higher score = more anomalous
            combined_score = 0.0
            if "isolation_forest" in scores:
                # IF: lower is anomalous, so negate
                combined_score += (self.if_threshold - scores["isolation_forest"]) * 0.5
            if "autoencoder" in scores:
                combined_score += (scores["autoencoder"] / self.ae_threshold) * 0.5
            combined_score = max(0.0, min(1.0, combined_score))
        else:
            combined_score = 0.0

        # Confidence based on agreement
        confidence = sum(is_anomaly_votes) / len(is_anomaly_votes) if is_anomaly_votes else 0.5

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=combined_score,
            confidence=confidence,
            anomaly_type=self._classify_anomaly(features, is_anomaly) if is_anomaly else None,
            details=scores if return_details else None,
        )

    def _classify_anomaly(
        self,
        features: Union[AnomalyFeatures, np.ndarray],
        is_anomaly: bool,
    ) -> Optional[str]:
        """Classify the type of anomaly based on feature patterns."""
        if not is_anomaly:
            return None

        if isinstance(features, AnomalyFeatures):
            if features.error_rate > 0.1:
                return "high_error_rate"
            elif features.duration_prediction_error > 2.0:
                return "execution_time_spike"
            elif features.memory_prediction_error > 2.0:
                return "memory_usage_spike"
            elif features.system_cpu_util > 0.95:
                return "cpu_saturation"
            elif features.system_memory_util > 0.95:
                return "memory_saturation"
            elif features.queue_depth > 1000:
                return "queue_overflow"
            else:
                return "unknown"
        return "unknown"

    def batch_detect(
        self,
        features_list: List[Union[AnomalyFeatures, np.ndarray]],
    ) -> List[AnomalyResult]:
        """Detect anomalies in a batch."""
        return [self.detect(f) for f in features_list]

    def save(self, ae_path: str, scaler_path: Optional[str] = None):
        """Save the detector."""
        torch.save(self.autoencoder.state_dict(), ae_path)
        if scaler_path and self.scaler is not None:
            import pickle
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
        logger.info(f"Saved anomaly detector to {ae_path}")

    def load(self, ae_path: str, scaler_path: Optional[str] = None):
        """Load the detector."""
        self.autoencoder.load_state_dict(
            torch.load(ae_path, map_location=self.device)
        )
        if scaler_path and self.scaler is not None:
            import pickle
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Loaded anomaly detector from {ae_path}")
