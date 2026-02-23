"""
Feature Engineering Pipeline for Syn OS

Transforms raw execution records into ML-ready features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    
    # Time features
    include_time_features: bool = True
    time_granularity: str = "hour"  # hour, minute
    
    # Rolling statistics
    rolling_windows: List[int] = None  # [5, 15, 60] for 5min, 15min, 1hr
    
    # Lag features
    lag_periods: List[int] = None  # [1, 2, 3] for previous tasks
    
    # Aggregation features
    aggregation_windows: List[int] = None  # [60, 300, 3600] seconds
    
    # Normalization
    normalize: bool = True
    normalize_method: str = "minmax"  # minmax, zscore, robust
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [5, 15, 60]
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3]
        if self.aggregation_windows is None:
            self.aggregation_windows = [60, 300, 3600]


class FeatureExtractor:
    """
    Extracts ML features from execution records.
    
    Features extracted:
    - Task characteristics (CPU, memory, priority)
    - Temporal features (hour, day of week, is_peak_hour)
    - System state features (current load, queue depth)
    - Historical features (rolling averages, lags)
    - Derived features (resource ratios, efficiency scores)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._fitted = False
        self._stats: Dict[str, Dict[str, float]] = {}
    
    def fit(self, records: List[Dict[str, Any]]) -> "FeatureExtractor":
        """Compute normalization statistics from training data."""
        if not HAS_PANDAS:
            logger.warning("Pandas not available, skipping fit")
            return self
        
        df = pd.DataFrame(records)
        
        # Compute statistics for normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self._stats[col] = {
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "std": df[col].std(),
                "median": df[col].median(),
                "q25": df[col].quantile(0.25),
                "q75": df[col].quantile(0.75),
            }
        
        self._fitted = True
        logger.info(f"Fitted feature extractor on {len(df)} records")
        return self
    
    def transform(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Transform records to feature matrix.
        
        Returns:
            features: numpy array of shape (n_records, n_features)
            feature_names: list of feature names
        """
        if not HAS_PANDAS:
            return self._transform_simple(records)
        
        df = pd.DataFrame(records)
        features = []
        feature_names = []
        
        # Task characteristics
        task_features, task_names = self._extract_task_features(df)
        features.append(task_features)
        feature_names.extend(task_names)
        
        # Temporal features
        if self.config.include_time_features:
            time_features, time_names = self._extract_time_features(df)
            features.append(time_features)
            feature_names.extend(time_names)
        
        # System state features
        system_features, system_names = self._extract_system_features(df)
        features.append(system_features)
        feature_names.extend(system_names)
        
        # Combine all features
        X = np.hstack(features)
        
        # Normalize if configured
        if self.config.normalize and self._fitted:
            X = self._normalize(X, feature_names)
        
        return X, feature_names
    
    def _extract_task_features(
        self, df: "pd.DataFrame"
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract task characteristic features."""
        feature_names = [
            "requested_cpu_cores",
            "requested_memory_mb",
            "requested_gpu",
            "priority",
            "dependency_count",
            "cpu_memory_ratio",
            "priority_normalized",
        ]
        
        features = np.zeros((len(df), len(feature_names)))
        
        features[:, 0] = df.get("requested_cpu_cores", 1).fillna(1).values
        features[:, 1] = df.get("requested_memory_mb", 1024).fillna(1024).values
        features[:, 2] = df.get("requested_gpu", False).astype(int).values
        features[:, 3] = df.get("priority", 5).fillna(5).values
        features[:, 4] = df.get("dependency_count", 0).fillna(0).values
        
        # Derived features
        features[:, 5] = features[:, 0] / (features[:, 1] / 1024 + 1)  # CPU/GB ratio
        features[:, 6] = features[:, 3] / 10.0  # Normalized priority
        
        return features, feature_names
    
    def _extract_time_features(
        self, df: "pd.DataFrame"
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract temporal features."""
        feature_names = [
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "is_business_hours",
            "is_weekend",
        ]
        
        features = np.zeros((len(df), len(feature_names)))
        
        if "submitted_at" in df.columns:
            timestamps = pd.to_datetime(df["submitted_at"])
            
            # Cyclical encoding for hour
            hours = timestamps.dt.hour
            features[:, 0] = np.sin(2 * np.pi * hours / 24)
            features[:, 1] = np.cos(2 * np.pi * hours / 24)
            
            # Cyclical encoding for day of week
            days = timestamps.dt.dayofweek
            features[:, 2] = np.sin(2 * np.pi * days / 7)
            features[:, 3] = np.cos(2 * np.pi * days / 7)
            
            # Binary features
            features[:, 4] = ((hours >= 9) & (hours <= 17)).astype(float)
            features[:, 5] = (days >= 5).astype(float)
        
        return features, feature_names
    
    def _extract_system_features(
        self, df: "pd.DataFrame"
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract system state features."""
        feature_names = [
            "system_cpu_util",
            "system_memory_util",
            "active_tasks_count",
            "queue_depth_at_submit",
            "system_load_score",
        ]
        
        features = np.zeros((len(df), len(feature_names)))
        
        features[:, 0] = df.get("system_cpu_util", 0).fillna(0).values
        features[:, 1] = df.get("system_memory_util", 0).fillna(0).values
        features[:, 2] = df.get("active_tasks_count", 0).fillna(0).values
        features[:, 3] = df.get("queue_depth_at_submit", 0).fillna(0).values
        
        # Composite load score
        features[:, 4] = (
            0.4 * features[:, 0] +
            0.4 * features[:, 1] +
            0.2 * np.minimum(features[:, 2] / 100, 1)
        )
        
        return features, feature_names
    
    def _normalize(
        self, X: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        """Normalize features."""
        X_norm = X.copy()
        
        for i, name in enumerate(feature_names):
            if name in self._stats:
                stats = self._stats[name]
                
                if self.config.normalize_method == "minmax":
                    range_val = stats["max"] - stats["min"]
                    if range_val > 0:
                        X_norm[:, i] = (X[:, i] - stats["min"]) / range_val
                
                elif self.config.normalize_method == "zscore":
                    if stats["std"] > 0:
                        X_norm[:, i] = (X[:, i] - stats["mean"]) / stats["std"]
                
                elif self.config.normalize_method == "robust":
                    iqr = stats["q75"] - stats["q25"]
                    if iqr > 0:
                        X_norm[:, i] = (X[:, i] - stats["median"]) / iqr
        
        return X_norm
    
    def _transform_simple(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Simple transform without pandas."""
        feature_names = [
            "requested_cpu_cores",
            "requested_memory_mb",
            "priority",
            "system_cpu_util",
            "system_memory_util",
        ]
        
        features = np.zeros((len(records), len(feature_names)))
        
        for i, record in enumerate(records):
            features[i, 0] = record.get("requested_cpu_cores", 1)
            features[i, 1] = record.get("requested_memory_mb", 1024)
            features[i, 2] = record.get("priority", 5)
            features[i, 3] = record.get("system_cpu_util", 0)
            features[i, 4] = record.get("system_memory_util", 0)
        
        return features, feature_names
    
    def extract_target(
        self,
        records: List[Dict[str, Any]],
        target: str = "execution_duration_ms",
    ) -> np.ndarray:
        """Extract target variable for training."""
        return np.array([r.get(target, 0) for r in records], dtype=np.float32)


class FeaturePipeline:
    """
    End-to-end feature pipeline.
    
    Combines:
    - Data loading
    - Feature extraction
    - Train/test split
    - Caching
    """
    
    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        self.config = config or FeatureConfig()
        self.extractor = FeatureExtractor(config)
        self.cache_dir = cache_dir
    
    def prepare_training_data(
        self,
        records: List[Dict[str, Any]],
        target: str = "execution_duration_ms",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Prepare training and test data.
        
        Returns dict with:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Fit extractor
        self.extractor.fit(records)
        
        # Transform
        X, feature_names = self.extractor.transform(records)
        y = self.extractor.extract_target(records, target)
        
        # Split
        n = len(X)
        indices = np.arange(n)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        split_idx = int(n * (1 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        return {
            "X_train": X[train_idx],
            "X_test": X[test_idx],
            "y_train": y[train_idx],
            "y_test": y[test_idx],
            "feature_names": feature_names,
        }
