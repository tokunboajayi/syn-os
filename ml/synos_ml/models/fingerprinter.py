"""
HardwareFingerprinter — Autoencoder for Learning System State Signatures.

This model learns the "normal" operating envelope of the host machine by
compressing system metrics into a low-dimensional latent space.  A high
reconstruction error signals that the system has drifted into an
abnormal/unseen state (thermal throttling, crypto-mining, etc.).

Architecture
────────────
  Input (6) → 32 → 16 → Latent(8) → 16 → 32 → Output (6)
  Activation: GELU (smooth, modern alternative to ReLU)
  Loss: MSE + KL-divergence regulariser on latent space (light VAE flavour)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not found — HardwareFingerprinter running in stub mode.")


# ── Data contract ────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "cpu_percent",
    "memory_percent",
    "disk_io_read_mb",
    "disk_io_write_mb",
    "net_bytes_in_mb",
    "net_bytes_out_mb",
]
INPUT_DIM = len(FEATURE_NAMES)


@dataclass
class FingerPrintResult:
    """Result of a single fingerprinting inference."""
    reconstruction_error: float = 0.0
    latent_vector: List[float] = field(default_factory=list)
    health_score: float = 100.0  # 0–100, derived from error
    is_anomalous: bool = False
    timestamp: float = field(default_factory=time.time)


# ── Model Definition ─────────────────────────────────────────────────────────

if HAS_TORCH:
    class _Encoder(nn.Module):
        """Maps raw metrics → latent space."""
        def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = 8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.GELU(),
                nn.LayerNorm(32),
                nn.Linear(32, 16),
                nn.GELU(),
                nn.LayerNorm(16),
            )
            self.mu = nn.Linear(16, latent_dim)
            self.log_var = nn.Linear(16, latent_dim)

        def forward(self, x):
            h = self.net(x)
            return self.mu(h), self.log_var(h)

    class _Decoder(nn.Module):
        """Reconstructs metrics from latent space."""
        def __init__(self, latent_dim: int = 8, output_dim: int = INPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.GELU(),
                nn.LayerNorm(16),
                nn.Linear(16, 32),
                nn.GELU(),
                nn.LayerNorm(32),
                nn.Linear(32, output_dim),
            )

        def forward(self, z):
            return self.net(z)

    class AutoencoderVAE(nn.Module):
        """Variational Autoencoder for system metrics."""
        def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = 8):
            super().__init__()
            self.encoder = _Encoder(input_dim, latent_dim)
            self.decoder = _Decoder(latent_dim, input_dim)

        def reparameterise(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, log_var = self.encoder(x)
            z = self.reparameterise(mu, log_var)
            x_hat = self.decoder(z)
            return x_hat, mu, log_var, z


# ── Public Interface ─────────────────────────────────────────────────────────

class HardwareFingerprinter:
    """
    High-level wrapper around the VAE model.

    Provides:
      • fingerprint(metrics)  → FingerPrintResult
      • train_step(batch)     → loss value
      • save / load
    """

    MODEL_FILENAME = "fingerprinter_vae.pt"

    def __init__(
        self,
        model_dir: str = "models",
        latent_dim: int = 8,
        anomaly_threshold: float = 0.15,
        learning_rate: float = 1e-3,
    ):
        self.model_dir = model_dir
        self.latent_dim = latent_dim
        self.anomaly_threshold = anomaly_threshold
        self.lr = learning_rate
        self._trained_samples = 0

        if HAS_TORCH:
            self.model = AutoencoderVAE(INPUT_DIM, latent_dim)
            self.optimiser = optim.AdamW(self.model.parameters(), lr=learning_rate)
            self.model.eval()
            self._load()
        else:
            self.model = None
            self.optimiser = None

    # ── Inference ────────────────────────────────────────────────────────

    def fingerprint(self, metrics: Dict[str, float]) -> FingerPrintResult:
        """Run a single inference pass and return the system health assessment."""
        if not HAS_TORCH:
            return self._heuristic_fingerprint(metrics)

        x = self._metrics_to_tensor(metrics)
        self.model.eval()
        with torch.no_grad():
            x_hat, mu, log_var, z = self.model(x)
            mse = nn.functional.mse_loss(x_hat, x).item()

        health = max(0.0, 100.0 * (1.0 - mse / self.anomaly_threshold))
        return FingerPrintResult(
            reconstruction_error=mse,
            latent_vector=z.squeeze().tolist(),
            health_score=round(health, 2),
            is_anomalous=mse > self.anomaly_threshold,
        )

    # ── Training ─────────────────────────────────────────────────────────

    def train_step(self, batch: List[Dict[str, float]], kl_weight: float = 0.001) -> float:
        """Perform one gradient-descent step on a mini-batch of metric snapshots."""
        if not HAS_TORCH:
            return 0.0

        tensors = [self._metrics_to_tensor(m) for m in batch]
        x = torch.cat(tensors, dim=0)

        self.model.train()
        x_hat, mu, log_var, _ = self.model(x)

        # Reconstruction + KL-divergence loss
        recon_loss = nn.functional.mse_loss(x_hat, x)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        loss = recon_loss + kl_weight * kl_loss

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self._trained_samples += len(batch)
        return loss.item()

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self):
        if not HAS_TORCH:
            return
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, self.MODEL_FILENAME)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "trained_samples": self._trained_samples,
            "anomaly_threshold": self.anomaly_threshold,
        }, path)
        logger.info(f"Fingerprinter saved to {path} ({self._trained_samples} samples)")

    def _load(self):
        path = os.path.join(self.model_dir, self.MODEL_FILENAME)
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                self.model.load_state_dict(ckpt["model_state"])
                self.optimiser.load_state_dict(ckpt["optimiser_state"])
                self._trained_samples = ckpt.get("trained_samples", 0)
                self.anomaly_threshold = ckpt.get("anomaly_threshold", self.anomaly_threshold)
                logger.info(f"Fingerprinter loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load fingerprinter: {e}")

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _metrics_to_tensor(metrics: Dict[str, float]):
        vec = [metrics.get(f, 0.0) for f in FEATURE_NAMES]
        return torch.tensor([vec], dtype=torch.float32)

    @staticmethod
    def _heuristic_fingerprint(metrics: Dict[str, float]) -> FingerPrintResult:
        """Fallback when PyTorch is unavailable."""
        cpu = metrics.get("cpu_percent", 0)
        mem = metrics.get("memory_percent", 0)
        score = max(0.0, 100.0 - (cpu * 0.4 + mem * 0.3))
        return FingerPrintResult(
            reconstruction_error=0.0,
            latent_vector=[],
            health_score=round(score, 2),
            is_anomalous=cpu > 95 or mem > 95,
        )

    @property
    def total_trained_samples(self) -> int:
        return self._trained_samples
