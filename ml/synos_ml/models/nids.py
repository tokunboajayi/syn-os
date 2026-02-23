from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from loguru import logger

from .anomaly import Autoencoder, AnomalyResult

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

@dataclass
class NetworkFeatures:
    """Features for network intrusion detection."""
    
    # Traffic volume
    bytes_in_per_sec: float
    bytes_out_per_sec: float
    packets_in_per_sec: float
    packets_out_per_sec: float
    
    # Connection metrics
    active_connections: int
    new_connections_per_sec: int
    connection_duration_mean: float
    
    # Entropy / Distribution (detects scanning/DDoS)
    src_ip_entropy: float      # High = many sources (DDoS)
    dst_port_entropy: float    # High = port scan
    protocol_entropy: float    # Unusual protocol mix
    
    # Flags (TCP)
    syn_count: int
    ack_count: int
    fin_count: int
    rst_count: int
    
    # Deep Packet Inspection (DPI) Metadata
    malformed_packets: int
    unusual_payload_size: int
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            np.log1p(self.bytes_in_per_sec),
            np.log1p(self.bytes_out_per_sec),
            np.log1p(self.packets_in_per_sec),
            np.log1p(self.packets_out_per_sec),
            np.log1p(self.active_connections),
            np.log1p(self.new_connections_per_sec),
            np.log1p(self.connection_duration_mean),
            self.src_ip_entropy,
            self.dst_port_entropy,
            self.protocol_entropy,
            np.log1p(self.syn_count),
            np.log1p(self.ack_count),
            np.log1p(self.fin_count),
            np.log1p(self.rst_count),
            np.log1p(self.malformed_packets),
            np.log1p(self.unusual_payload_size),
        ], dtype=np.float32)

class NIDS:
    """
    Network Intrusion Detection System.
    
    Uses an Autoencoder + Isolation Forest ensemble to detect anomalous network traffic.
    """
    
    def __init__(
        self,
        input_dim: int = 16, # Match features count
        contamination: float = 0.005, # NIDS usually has lower tolerance
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.contamination = contamination
        self.is_fitted = False
        
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        
        # Isolation Forest for statistical outliers
        self.if_threshold = 0.0
        self.isolation_forest = None
        if HAS_SKLEARN:
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
        # Autoencoder for pattern reconstruction
        self.ae_threshold = 0.1
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            latent_dim=8 
        ).to(self.device)
        
    def fit(self, X: np.ndarray, epochs: int = 50, batch_size: int = 64):
        """Fit the NIDS on 'normal' traffic baseline."""
        logger.info(f"Training NIDS on {X.shape[0]} baseline samples")
        
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        # Train Isolation Forest
        if self.isolation_forest is not None:
            self.isolation_forest.fit(X_scaled)
            scores = self.isolation_forest.decision_function(X_scaled)
            self.if_threshold = np.percentile(scores, self.contamination * 100)
            
        # Train Autoencoder
        self.autoencoder.train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.autoencoder(batch)
                loss = torch.nn.functional.mse_loss(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        # Set threshold
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            errors = self.autoencoder.reconstruction_error(X_tensor).cpu().numpy()
            self.ae_threshold = np.percentile(errors, (1 - self.contamination) * 100)
            
        self.is_fitted = True
        logger.info(f"NIDS fitted. AE Threshold: {self.ae_threshold:.6f}")

    def detect(self, features: Union[NetworkFeatures, np.ndarray]) -> AnomalyResult:
        """Detect intrusion attempt."""
        if isinstance(features, NetworkFeatures):
            x = features.to_array().reshape(1, -1)
        else:
            x = features.reshape(1, -1)
            
        if self.scaler and self.is_fitted:
            x = self.scaler.transform(x)
            
        # 1. Check Autoencoder reconstruction
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon_error = self.autoencoder.reconstruction_error(x_tensor).item()
            
        # 2. Check Isolation Forest
        if_score = 0.0
        if self.isolation_forest is not None:
            if_score = self.isolation_forest.decision_function(x)[0]
            
        is_anomaly = (recon_error > self.ae_threshold) or (if_score < self.if_threshold)
        
        # Determine threat type
        anomaly_type = None
        if is_anomaly:
            if recon_error > self.ae_threshold * 2:
                anomaly_type = "UNKNOWN_ZERO_DAY"
            elif features.syn_count > 1000 and features.ack_count < 10:
                anomaly_type = "SYN_FLOOD_DDOS"
            elif features.dst_port_entropy > 3.0:
                anomaly_type = "PORT_SCANNING"
            elif features.bytes_out_per_sec > 10_000_000:
                anomaly_type = "DATA_EXFILTRATION"
            else:
                anomaly_type = "ANOMALOUS_TRAFFIC"
                
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(recon_error),
            confidence=0.85, # Placeholder
            anomaly_type=anomaly_type,
            details={"recon_error": recon_error, "if_score": if_score}
        )

    def save(self, path: str):
        """Save NIDS model."""
        state = {
            "autoencoder": self.autoencoder.state_dict(),
            "ae_threshold": self.ae_threshold,
            "if_threshold": self.if_threshold,
            "contamination": self.contamination,
            "isolation_forest": self.isolation_forest,
            "scaler": self.scaler,
            "input_dim": self.input_dim
        }
        import pickle
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved NIDS model to {path}")

    def load(self, path: str):
        """Load NIDS model."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.autoencoder.load_state_dict(state["autoencoder"])
        self.ae_threshold = state["ae_threshold"]
        self.if_threshold = state["if_threshold"]
        self.contamination = state["contamination"]
        self.isolation_forest = state["isolation_forest"]
        self.scaler = state["scaler"]
        self.input_dim = state["input_dim"]
        self.is_fitted = True
        self.autoencoder.eval()
        logger.info(f"Loaded NIDS model from {path}")
