"""
ML Anomaly Detection Module
Uses Isolation Forest to detect anomalous system and network behavior.
"""

import logging
import pickle
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    np = None
    IsolationForest = None
    StandardScaler = None

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    AI-powered Anomaly Detector using Isolation Forest.
    Detects outliers in system metrics (CPU, Memory, Network I/O).
    """

    def __init__(self, model_path: str = "models/anomaly_detector.pkl"):
        self.model_path = model_path
        self.is_trained = False
        self.training_data: List[List[float]] = []
        
        self.model = None
        self.scaler = None
        
        if HAS_ML:
            self.model = IsolationForest(contamination=0.01, random_state=42)
            self.scaler = StandardScaler()
            # Load existing model if available
            self._load_model()
        else:
            logger.warning("ML libraries not found. AnomalyDetector running in Heuristic Mode.")

    def _load_model(self):
        """Load trained model from disk"""
        if not HAS_ML:
             return

        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.scaler = saved_data['scaler']
                    self.is_trained = True
                logger.info("Anomaly detection model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load anomaly detection model: {e}")

    def save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler
                }, f)
            logger.info(f"Anomaly detection model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save anomaly detection model: {e}")

    def add_training_sample(self, metrics: Dict[str, float]):
        """
        Add a data sample for training.
        Expected keys: cpu_percent, memory_percent, bytes_sent, bytes_recv, packet_count
        """
        features = self._extract_features(metrics)
        self.training_data.append(features)
        
        # Auto-train if we have enough data and not trained yet
        if not self.is_trained and len(self.training_data) >= 100:
            self.train()

    def train(self):
        """Train the Isolation Forest model"""
        if not HAS_ML:
            return

        if not self.training_data:
            logger.warning("No training data available")
            return

        try:
            X = np.array(self.training_data)
            X_scaled = self.scaler.fit_transform(X)
            
            self.model.fit(X_scaled)
            self.is_trained = True
            logger.info(f"Anomaly detection model trained on {len(self.training_data)} samples")
            self.save_model()
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {e}")

    def detect(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect anomalies in the provided metrics.
        Returns score (-1 for anomaly, 1 for normal) and anomaly probability.
        """
        if not self.is_trained or not HAS_ML:
            # Fallback heuristic if not trained
            return self._heuristic_check(metrics)

        try:
            features = self._extract_features(metrics)
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0] # -1 for anomaly, 1 for normal
            score = self.model.decision_function(X_scaled)[0] # lower is more anomalous
            
            is_anomaly = prediction == -1
            
            # Normalize score to 0-1 probability of anomaly
            # Decision function roughly ranges from -0.5 to 0.5 depending on data
            # -0.5 -> very anomalous, 0.5 -> very normal
            # We want to map this to 0.0 (normal) to 1.0 (anomaly)
            # This is an approximation
            anomaly_prob = 1.0 / (1.0 + np.exp(10 * score)) 

            return {
                "is_anomaly": is_anomaly,
                "score": float(score),
                "probability": float(anomaly_prob),
                "details": "ML-detected anomaly" if is_anomaly else "Normal behavior"
            }
            
        except Exception as e:
            logger.error(f"Error during anomaly detection inference: {e}")
            return self._heuristic_check(metrics)

    def _extract_features(self, metrics: Dict[str, float]) -> List[float]:
        """Extract feature vector from metrics dict"""
        return [
            metrics.get('cpu_percent', 0.0),
            metrics.get('memory_percent', 0.0),
            metrics.get('network_bytes_in', 0.0),
            metrics.get('network_bytes_out', 0.0),
            metrics.get('packet_count', 0.0)
        ]

    def _heuristic_check(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Fallback heuristic check when ML model is not ready"""
        is_anomaly = False
        reasons = []
        
        if metrics.get('cpu_percent', 0) > 95:
            is_anomaly = True
            reasons.append("CPU critical")
            
        if metrics.get('memory_percent', 0) > 95:
            is_anomaly = True
            reasons.append("Memory critical")
            
        return {
            "is_anomaly": is_anomaly,
            "score": -1.0 if is_anomaly else 1.0,
            "probability": 1.0 if is_anomaly else 0.0,
            "details": ", ".join(reasons) if reasons else "Normal (Heuristic)"
        }

# Global instance
anomaly_detector = AnomalyDetector()
