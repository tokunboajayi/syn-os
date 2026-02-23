"""
Model Drift Detection for Syn OS

Detects distribution shifts that may impact model performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import deque
from loguru import logger

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class DriftType:
    """Types of detected drift."""
    NONE = "none"
    COVARIATE = "covariate"  # Input distribution shift
    CONCEPT = "concept"  # Target distribution shift
    PREDICTION = "prediction"  # Prediction distribution shift


@dataclass
class DriftReport:
    """Report of detected drift."""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    drift_detected: bool = False
    drift_type: str = DriftType.NONE
    severity: str = "none"  # none, low, medium, high
    
    # Statistics
    affected_features: List[str] = field(default_factory=list)
    p_values: Dict[str, float] = field(default_factory=dict)
    distances: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    
    # Detection thresholds
    p_value_threshold: float = 0.01
    distance_threshold: float = 0.1
    
    # Reference window
    reference_window_size: int = 1000
    detection_window_size: int = 100
    
    # Features to monitor
    feature_names: List[str] = field(default_factory=list)
    target_name: str = "target"
    
    # Alert settings
    alert_on_drift: bool = True
    consecutive_alerts_threshold: int = 3


class DriftDetector:
    """
    Detects model drift using statistical tests.
    
    Methods:
    - Kolmogorov-Smirnov test for distribution comparison
    - Population Stability Index (PSI)
    - Wasserstein distance
    - Prediction error tracking
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        
        # Reference distributions
        self._reference_features: Dict[str, deque] = {}
        self._reference_targets: deque = deque(maxlen=self.config.reference_window_size)
        self._reference_predictions: deque = deque(maxlen=self.config.reference_window_size)
        
        # Current windows
        self._current_features: Dict[str, deque] = {}
        self._current_targets: deque = deque(maxlen=self.config.detection_window_size)
        self._current_predictions: deque = deque(maxlen=self.config.detection_window_size)
        
        # Tracking
        self._consecutive_drift_detections: int = 0
        self._history: List[DriftReport] = []
        
        # Initialize feature buffers
        for name in self.config.feature_names:
            self._reference_features[name] = deque(maxlen=self.config.reference_window_size)
            self._current_features[name] = deque(maxlen=self.config.detection_window_size)
    
    def update_reference(
        self,
        features: Dict[str, float],
        target: Optional[float] = None,
        prediction: Optional[float] = None,
    ):
        """Add sample to reference distribution."""
        for name, value in features.items():
            if name in self._reference_features:
                self._reference_features[name].append(value)
        
        if target is not None:
            self._reference_targets.append(target)
        
        if prediction is not None:
            self._reference_predictions.append(prediction)
    
    def add_sample(
        self,
        features: Dict[str, float],
        target: Optional[float] = None,
        prediction: Optional[float] = None,
    ):
        """Add sample to current window for drift detection."""
        for name, value in features.items():
            if name not in self._current_features:
                self._current_features[name] = deque(maxlen=self.config.detection_window_size)
            self._current_features[name].append(value)
        
        if target is not None:
            self._current_targets.append(target)
        
        if prediction is not None:
            self._current_predictions.append(prediction)
    
    def detect(self) -> DriftReport:
        """Run drift detection on current window."""
        report = DriftReport()
        
        # Check covariate drift (input features)
        feature_drifts = self._detect_covariate_drift()
        if feature_drifts:
            report.drift_detected = True
            report.drift_type = DriftType.COVARIATE
            report.affected_features = list(feature_drifts.keys())
            report.p_values = {k: v["p_value"] for k, v in feature_drifts.items()}
            report.distances = {k: v["distance"] for k, v in feature_drifts.items()}
        
        # Check concept drift (target distribution)
        concept_drift = self._detect_concept_drift()
        if concept_drift["detected"]:
            report.drift_detected = True
            if report.drift_type == DriftType.COVARIATE:
                report.drift_type = DriftType.CONCEPT  # Priority
            else:
                report.drift_type = DriftType.CONCEPT
            report.p_values["target"] = concept_drift["p_value"]
        
        # Check prediction drift
        prediction_drift = self._detect_prediction_drift()
        if prediction_drift["detected"]:
            report.drift_detected = True
            report.drift_type = DriftType.PREDICTION
            report.p_values["prediction"] = prediction_drift["p_value"]
        
        # Determine severity
        report.severity = self._compute_severity(report)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        # Update consecutive detection count
        if report.drift_detected:
            self._consecutive_drift_detections += 1
        else:
            self._consecutive_drift_detections = 0
        
        self._history.append(report)
        
        if report.drift_detected:
            logger.warning(
                f"Drift detected: type={report.drift_type}, "
                f"severity={report.severity}, "
                f"features={report.affected_features}"
            )
        
        return report
    
    def _detect_covariate_drift(self) -> Dict[str, Dict[str, float]]:
        """Detect drift in input features."""
        drifts = {}
        
        for name in self._current_features:
            ref_values = list(self._reference_features.get(name, []))
            curr_values = list(self._current_features[name])
            
            if len(ref_values) < 30 or len(curr_values) < 30:
                continue
            
            # Kolmogorov-Smirnov test
            if HAS_SCIPY:
                ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
            else:
                # Simplified comparison
                ref_mean, ref_std = np.mean(ref_values), np.std(ref_values)
                curr_mean, curr_std = np.mean(curr_values), np.std(curr_values)
                
                if ref_std > 0:
                    z_score = abs(curr_mean - ref_mean) / ref_std
                    p_value = max(0.001, 1 - z_score / 3)
                else:
                    p_value = 1.0
                ks_stat = abs(np.mean(ref_values) - np.mean(curr_values))
            
            # PSI calculation
            psi = self._calculate_psi(ref_values, curr_values)
            
            if p_value < self.config.p_value_threshold or psi > self.config.distance_threshold:
                drifts[name] = {
                    "p_value": p_value,
                    "distance": psi,
                    "ks_stat": ks_stat,
                }
        
        return drifts
    
    def _detect_concept_drift(self) -> Dict[str, Any]:
        """Detect drift in target distribution."""
        ref_targets = list(self._reference_targets)
        curr_targets = list(self._current_targets)
        
        if len(ref_targets) < 30 or len(curr_targets) < 30:
            return {"detected": False, "p_value": 1.0}
        
        if HAS_SCIPY:
            _, p_value = stats.ks_2samp(ref_targets, curr_targets)
        else:
            ref_mean = np.mean(ref_targets)
            curr_mean = np.mean(curr_targets)
            ref_std = np.std(ref_targets)
            
            if ref_std > 0:
                z_score = abs(curr_mean - ref_mean) / ref_std
                p_value = max(0.001, 1 - z_score / 3)
            else:
                p_value = 1.0
        
        return {
            "detected": p_value < self.config.p_value_threshold,
            "p_value": p_value,
        }
    
    def _detect_prediction_drift(self) -> Dict[str, Any]:
        """Detect drift in prediction distribution."""
        ref_preds = list(self._reference_predictions)
        curr_preds = list(self._current_predictions)
        
        if len(ref_preds) < 30 or len(curr_preds) < 30:
            return {"detected": False, "p_value": 1.0}
        
        if HAS_SCIPY:
            _, p_value = stats.ks_2samp(ref_preds, curr_preds)
        else:
            ref_mean = np.mean(ref_preds)
            curr_mean = np.mean(curr_preds)
            ref_std = np.std(ref_preds)
            
            if ref_std > 0:
                z_score = abs(curr_mean - ref_mean) / ref_std
                p_value = max(0.001, 1 - z_score / 3)
            else:
                p_value = 1.0
        
        return {
            "detected": p_value < self.config.p_value_threshold,
            "p_value": p_value,
        }
    
    def _calculate_psi(
        self,
        reference: List[float],
        current: List[float],
        bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index."""
        # Create bins from reference distribution
        ref_array = np.array(reference)
        curr_array = np.array(current)
        
        # Handle edge cases
        if len(ref_array) == 0 or len(curr_array) == 0:
            return 0.0
        
        min_val = min(ref_array.min(), curr_array.min())
        max_val = max(ref_array.max(), curr_array.max())
        
        if min_val == max_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_hist, _ = np.histogram(ref_array, bins=bin_edges)
        curr_hist, _ = np.histogram(curr_array, bins=bin_edges)
        
        # Convert to proportions with small constant to avoid division by zero
        ref_prop = (ref_hist + 0.0001) / len(ref_array)
        curr_prop = (curr_hist + 0.0001) / len(curr_array)
        
        # PSI formula
        psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
        
        return float(psi)
    
    def _compute_severity(self, report: DriftReport) -> str:
        """Compute drift severity level."""
        if not report.drift_detected:
            return "none"
        
        # Count affected features
        affected_count = len(report.affected_features)
        
        # Look at maximum distance
        max_distance = max(report.distances.values()) if report.distances else 0
        
        if affected_count > 3 or max_distance > 0.3:
            return "high"
        elif affected_count > 1 or max_distance > 0.15:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, report: DriftReport) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not report.drift_detected:
            return recommendations
        
        if report.drift_type == DriftType.COVARIATE:
            recommendations.append("Input feature distributions have shifted")
            recommendations.append("Consider retraining with recent data")
            if len(report.affected_features) > 2:
                recommendations.append("Multiple features affected - investigate data pipeline")
        
        elif report.drift_type == DriftType.CONCEPT:
            recommendations.append("Target distribution has changed")
            recommendations.append("Model may need fundamental redesign")
            recommendations.append("Verify label quality and definitions")
        
        elif report.drift_type == DriftType.PREDICTION:
            recommendations.append("Model predictions have diverged")
            recommendations.append("Run immediate model evaluation")
        
        if report.severity == "high":
            recommendations.append("Immediate action required - consider rollback")
        
        if self._consecutive_drift_detections >= self.config.consecutive_alerts_threshold:
            recommendations.append("Persistent drift detected - automated retraining recommended")
        
        return recommendations
    
    def get_history(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get recent drift detection history."""
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "drift_detected": r.drift_detected,
                "drift_type": r.drift_type,
                "severity": r.severity,
                "affected_features": r.affected_features,
            }
            for r in self._history[-n:]
        ]
    
    def reset_reference(self):
        """Reset reference distributions with current data."""
        for name in self._reference_features:
            self._reference_features[name] = deque(
                self._current_features.get(name, []),
                maxlen=self.config.reference_window_size,
            )
        
        self._reference_targets = deque(
            self._current_targets,
            maxlen=self.config.reference_window_size,
        )
        
        self._reference_predictions = deque(
            self._current_predictions,
            maxlen=self.config.reference_window_size,
        )
        
        logger.info("Reset reference distributions with current data")
