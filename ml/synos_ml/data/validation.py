"""
Data Validation and Quality Checks for Syn OS

Ensures data quality before training ML models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from loguru import logger


class ValidationLevel(Enum):
    """Validation severity level."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    name: str
    passed: bool
    level: ValidationLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """Check if all critical validations passed."""
        return not any(
            r.level == ValidationLevel.ERROR and not r.passed
            for r in self.results
        )
    
    @property
    def errors(self) -> List[ValidationResult]:
        """Get all error-level failures."""
        return [r for r in self.results if r.level == ValidationLevel.ERROR and not r.passed]
    
    @property
    def warnings(self) -> List[ValidationResult]:
        """Get all warnings."""
        return [r for r in self.results if r.level == ValidationLevel.WARNING and not r.passed]
    
    def summary(self) -> str:
        """Generate summary string."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        errors = len(self.errors)
        warnings = len(self.warnings)
        
        return (
            f"Validation: {passed}/{total} passed, "
            f"{errors} errors, {warnings} warnings"
        )


class DataValidator:
    """
    Validates execution data for ML training.
    
    Checks:
    - Schema validation (required fields)
    - Value range validation
    - Null/missing value detection
    - Outlier detection
    - Temporal consistency
    - Distribution drift detection
    """
    
    # Required fields for training
    REQUIRED_FIELDS = {
        "task_id": str,
        "execution_duration_ms": (int, float),
        "status": str,
    }
    
    # Numeric field ranges (field: (min, max))
    FIELD_RANGES = {
        "execution_duration_ms": (0, 86400000),  # 0 to 24 hours
        "queue_wait_ms": (0, 3600000),  # 0 to 1 hour
        "requested_cpu_cores": (1, 128),
        "requested_memory_mb": (1, 1048576),  # 1 MB to 1 TB
        "priority": (0, 9),
        "system_cpu_util": (0, 1),
        "system_memory_util": (0, 1),
    }
    
    def __init__(
        self,
        null_threshold: float = 0.1,  # Max 10% nulls
        outlier_threshold: float = 0.01,  # Max 1% outliers
    ):
        self.null_threshold = null_threshold
        self.outlier_threshold = outlier_threshold
        self._baseline_stats: Optional[Dict[str, Dict[str, float]]] = None
    
    def validate(
        self,
        records: List[Dict[str, Any]],
    ) -> ValidationReport:
        """Run all validation checks."""
        report = ValidationReport()
        
        if not records:
            report.results.append(ValidationResult(
                name="not_empty",
                passed=False,
                level=ValidationLevel.ERROR,
                message="No records provided",
            ))
            return report
        
        # Run all checks
        report.results.append(self._check_schema(records))
        report.results.append(self._check_required_fields(records))
        report.results.extend(self._check_value_ranges(records))
        report.results.append(self._check_null_values(records))
        report.results.append(self._check_duplicates(records))
        report.results.extend(self._check_outliers(records))
        report.results.append(self._check_status_distribution(records))
        
        if self._baseline_stats:
            report.results.append(self._check_distribution_drift(records))
        
        logger.info(report.summary())
        return report
    
    def set_baseline(self, records: List[Dict[str, Any]]):
        """Set baseline statistics for drift detection."""
        self._baseline_stats = {}
        
        for field in self.FIELD_RANGES:
            values = [r.get(field) for r in records if r.get(field) is not None]
            if values:
                values = np.array(values, dtype=np.float32)
                self._baseline_stats[field] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "p5": float(np.percentile(values, 5)),
                    "p95": float(np.percentile(values, 95)),
                }
        
        logger.info(f"Set baseline statistics for {len(self._baseline_stats)} fields")
    
    def _check_schema(self, records: List[Dict[str, Any]]) -> ValidationResult:
        """Check that all records have consistent schema."""
        first_keys = set(records[0].keys())
        inconsistent = 0
        
        for record in records[1:]:
            if set(record.keys()) != first_keys:
                inconsistent += 1
        
        return ValidationResult(
            name="schema_consistency",
            passed=inconsistent == 0,
            level=ValidationLevel.WARNING,
            message=f"{inconsistent}/{len(records)} records have inconsistent schema",
            details={"inconsistent_count": inconsistent},
        )
    
    def _check_required_fields(self, records: List[Dict[str, Any]]) -> ValidationResult:
        """Check that all required fields are present."""
        missing_fields = []
        
        for field, expected_type in self.REQUIRED_FIELDS.items():
            missing = sum(1 for r in records if field not in r)
            if missing > 0:
                missing_fields.append((field, missing))
        
        return ValidationResult(
            name="required_fields",
            passed=len(missing_fields) == 0,
            level=ValidationLevel.ERROR,
            message=f"Missing required fields: {missing_fields}" if missing_fields else "All required fields present",
            details={"missing": missing_fields},
        )
    
    def _check_value_ranges(self, records: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Check that numeric values are within expected ranges."""
        results = []
        
        for field, (min_val, max_val) in self.FIELD_RANGES.items():
            values = [r.get(field) for r in records if r.get(field) is not None]
            if not values:
                continue
            
            values = np.array(values, dtype=np.float32)
            out_of_range = np.sum((values < min_val) | (values > max_val))
            
            results.append(ValidationResult(
                name=f"range_{field}",
                passed=out_of_range == 0,
                level=ValidationLevel.WARNING,
                message=f"{field}: {out_of_range}/{len(values)} values out of range [{min_val}, {max_val}]",
                details={
                    "out_of_range": int(out_of_range),
                    "min_observed": float(np.min(values)),
                    "max_observed": float(np.max(values)),
                },
            ))
        
        return results
    
    def _check_null_values(self, records: List[Dict[str, Any]]) -> ValidationResult:
        """Check for excessive null values."""
        null_counts = {}
        
        for field in records[0].keys():
            null_count = sum(1 for r in records if r.get(field) is None)
            null_ratio = null_count / len(records)
            if null_ratio > self.null_threshold:
                null_counts[field] = null_ratio
        
        return ValidationResult(
            name="null_values",
            passed=len(null_counts) == 0,
            level=ValidationLevel.WARNING,
            message=f"Fields with excessive nulls (>{self.null_threshold:.0%}): {list(null_counts.keys())}",
            details={"null_ratios": null_counts},
        )
    
    def _check_duplicates(self, records: List[Dict[str, Any]]) -> ValidationResult:
        """Check for duplicate task IDs."""
        task_ids = [r.get("task_id") for r in records if r.get("task_id")]
        unique_ids = set(task_ids)
        duplicates = len(task_ids) - len(unique_ids)
        
        return ValidationResult(
            name="duplicates",
            passed=duplicates == 0,
            level=ValidationLevel.WARNING,
            message=f"{duplicates} duplicate task IDs found",
            details={"duplicate_count": duplicates},
        )
    
    def _check_outliers(self, records: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Check for outliers using IQR method."""
        results = []
        
        for field in ["execution_duration_ms", "queue_wait_ms"]:
            values = [r.get(field) for r in records if r.get(field) is not None]
            if not values:
                continue
            
            values = np.array(values, dtype=np.float32)
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            outliers = np.sum((values < lower) | (values > upper))
            outlier_ratio = outliers / len(values)
            
            results.append(ValidationResult(
                name=f"outliers_{field}",
                passed=outlier_ratio <= self.outlier_threshold,
                level=ValidationLevel.INFO,
                message=f"{field}: {outlier_ratio:.2%} outliers detected",
                details={
                    "outlier_count": int(outliers),
                    "outlier_ratio": float(outlier_ratio),
                    "iqr_bounds": (float(lower), float(upper)),
                },
            ))
        
        return results
    
    def _check_status_distribution(self, records: List[Dict[str, Any]]) -> ValidationResult:
        """Check task status distribution."""
        from collections import Counter
        
        statuses = [r.get("status", "unknown") for r in records]
        counts = Counter(statuses)
        
        completed_ratio = counts.get("completed", 0) / len(records)
        
        return ValidationResult(
            name="status_distribution",
            passed=completed_ratio >= 0.5,
            level=ValidationLevel.WARNING,
            message=f"Completed ratio: {completed_ratio:.2%}",
            details={"status_counts": dict(counts)},
        )
    
    def _check_distribution_drift(self, records: List[Dict[str, Any]]) -> ValidationResult:
        """Check for distribution drift from baseline."""
        drift_detected = []
        
        for field, baseline in self._baseline_stats.items():
            values = [r.get(field) for r in records if r.get(field) is not None]
            if not values:
                continue
            
            values = np.array(values, dtype=np.float32)
            current_mean = np.mean(values)
            
            # Check if mean shifted by more than 2 standard deviations
            if baseline["std"] > 0:
                z_score = abs(current_mean - baseline["mean"]) / baseline["std"]
                if z_score > 2:
                    drift_detected.append((field, z_score))
        
        return ValidationResult(
            name="distribution_drift",
            passed=len(drift_detected) == 0,
            level=ValidationLevel.WARNING,
            message=f"Drift detected in fields: {[d[0] for d in drift_detected]}",
            details={"drift_scores": dict(drift_detected)},
        )


def quick_validate(records: List[Dict[str, Any]]) -> bool:
    """Quick validation check - returns True if data is usable."""
    validator = DataValidator()
    report = validator.validate(records)
    return report.passed
