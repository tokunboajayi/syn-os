from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

@dataclass
class TargetInfo:
    """Information about a target host."""
    ip: str
    open_ports: List[int]
    banners: Dict[int, str] = field(default_factory=dict)
    os_fingerprint: Optional[str] = None
    
    def to_feature_vector(self, max_ports: int = 1024) -> np.ndarray:
        """Convert target info to feature vector."""
        # Simple multi-hot encoding of common ports
        # In a real system, we'd use embeddings for ports and banners
        
        # 1. Port features (first 1024 ports)
        port_vector = np.zeros(max_ports, dtype=np.float32)
        for p in self.open_ports:
            if p < max_ports:
                port_vector[p] = 1.0
                
        # 2. Count of open ports
        port_count = len(self.open_ports)
        
        # 3. Banner presence (binary flag per port for now)
        banner_vector = np.zeros(max_ports, dtype=np.float32)
        for p in self.banners:
            if p < max_ports:
                banner_vector[p] = 1.0
                
        return np.concatenate([
            port_vector, 
            np.array([port_count], dtype=np.float32), 
            banner_vector
        ])

@dataclass
class VulnPrediction:
    """Vulnerability prediction result."""
    risk_score: float # 0-10
    criticality: str # Low, Medium, High, Critical
    likely_cves: List[str]
    confidence: float

class VulnerabilityPredictor:
    """
    Predicts vulnerability risk based on open ports and services.
    """
    
    def __init__(self, input_dim: int = 2049): # 1024 ports + 1 count + 1024 banner flags
        self.input_dim = input_dim
        self.model = None
        self.is_fitted = False
        
        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
        else:
            logger.warning("XGBoost not found, falling back to simple heuristic/metrics")
            
        # Mapping of port combinations to CVEs (simulated knowledge base)
        self.cve_map = {
            21: ["CVE-2011-2523"], # vsftpd
            22: ["CVE-2018-15473"], # OpenSSH
            80: ["CVE-2017-5638"], # Struts
            443: ["CVE-2014-0160"], # Heartbleed
            445: ["CVE-2017-0144"], # EternalBlue
            3389: ["CVE-2019-0708"], # BlueKeep
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the predictor."""
        if HAS_XGBOOST:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("VulnerabilityPredictor fitted (XGBoost)")
        else:
            logger.info("VulnerabilityPredictor: No training needed (Heuristic mode)")
            self.is_fitted = True

    def predict(self, target: TargetInfo) -> VulnPrediction:
        """Predict risk for a target."""
        features = target.to_feature_vector(max_ports=1024 // 2) # Adjust dimensions if needed
        # Note: input_dim above was 2049, logic needs to match
        # Let's fix the dimensions to match strictly
        
        # Re-implement feature vector generation to be consistent
        vec = self._feature_gen(target)
        
        score = 0.0
        if self.is_fitted and HAS_XGBOOST and self.model:
            score = float(self.model.predict(vec.reshape(1, -1))[0])
            score = max(0.0, min(10.0, score))
        else:
            # Heuristic fallback
            score = min(10.0, len(target.open_ports) * 0.5)
            if 445 in target.open_ports: score += 4.0
            if 3389 in target.open_ports: score += 3.0
            score = min(10.0, score)
            
        # Determine criticality
        if score >= 9.0: criticality = "CRITICAL"
        elif score >= 7.0: criticality = "HIGH"
        elif score >= 4.0: criticality = "MEDIUM"
        else: criticality = "LOW"
        
        # Identify likely CVEs
        likely_cves = []
        for port in target.open_ports:
            if port in self.cve_map:
                likely_cves.extend(self.cve_map[port])
                
        return VulnPrediction(
            risk_score=score,
            criticality=criticality,
            likely_cves=likely_cves,
            confidence=0.85 if HAS_XGBOOST else 0.5
        )

    def _feature_gen(self, target: TargetInfo) -> np.ndarray:
        """Internal scalable feature generation."""
        # Using fixed size 2049 for now (1024 ports + 1 + 1024 banners)
        max_ports = 1024
        vec = np.zeros(max_ports * 2 + 1, dtype=np.float32)
        
        for p in target.open_ports:
            if p < max_ports:
                vec[p] = 1.0
        
        vec[max_ports] = len(target.open_ports)
        
        for p in target.banners:
            if p < max_ports:
                vec[max_ports + 1 + p] = 1.0
                
        return vec
