"""
Blue-Green Model Deployment for Syn OS

Zero-downtime model updates with instant rollback capability.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from enum import Enum
from loguru import logger


class DeploymentSlot(Enum):
    """Active deployment slot."""
    BLUE = "blue"
    GREEN = "green"


@dataclass
class DeploymentInfo:
    """Information about a deployed model version."""
    
    slot: DeploymentSlot
    model_path: str
    version: str
    deployed_at: datetime
    is_active: bool = False
    
    # Metrics
    request_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    
    # Health
    health_check_passed: bool = True
    last_health_check: Optional[datetime] = None


@dataclass
class DeploymentConfig:
    """Configuration for blue-green deployment."""
    
    # Paths
    models_dir: str = "./models"
    
    # Health checks
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5
    
    # Traffic management
    gradual_rollout: bool = True
    initial_traffic_fraction: float = 0.1
    traffic_increment: float = 0.1
    increment_interval_seconds: int = 60
    
    # Rollback
    auto_rollback_enabled: bool = True
    error_rate_threshold: float = 0.05  # 5% errors trigger rollback
    latency_threshold_ms: float = 1000  # 1s latency triggers rollback


class BlueGreenDeployer:
    """
    Blue-green deployment manager for ML models.
    
    Features:
    - Zero-downtime deployments
    - Instant rollback
    - Health checking
    - Gradual traffic shifting
    - Automatic rollback on errors
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[DeploymentConfig] = None,
    ):
        self.model_name = model_name
        self.config = config or DeploymentConfig()
        
        self._blue: Optional[DeploymentInfo] = None
        self._green: Optional[DeploymentInfo] = None
        self._active_slot: DeploymentSlot = DeploymentSlot.BLUE
        self._traffic_fraction: float = 1.0  # Fraction going to active
        
        self._health_check_task: Optional[asyncio.Task] = None
        self._rollout_task: Optional[asyncio.Task] = None
        self._deployment_history: List[Dict[str, Any]] = []
        
        # Ensure models directory exists
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
    
    async def deploy_new_version(
        self,
        model: Any,
        model_path: str,
        version: Optional[str] = None,
    ) -> bool:
        """
        Deploy a new model version.
        
        The new version is deployed to the inactive slot, then
        traffic is gradually shifted if configured.
        
        Returns True if deployment was successful.
        """
        if version is None:
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Determine target slot
        if self._active_slot == DeploymentSlot.BLUE:
            target_slot = DeploymentSlot.GREEN
        else:
            target_slot = DeploymentSlot.BLUE
        
        logger.info(f"Deploying {self.model_name} v{version} to {target_slot.value}")
        
        # Create deployment info
        deployment = DeploymentInfo(
            slot=target_slot,
            model_path=model_path,
            version=version,
            deployed_at=datetime.utcnow(),
            is_active=False,
        )
        
        # Load model to inactive slot
        try:
            if target_slot == DeploymentSlot.BLUE:
                self._blue = deployment
            else:
                self._green = deployment
            
            # Run health check on new deployment
            if self.config.health_check_enabled:
                health_ok = await self._run_health_check(target_slot)
                if not health_ok:
                    logger.error(f"Health check failed for {target_slot.value}")
                    return False
            
            # Start gradual rollout or switch immediately
            if self.config.gradual_rollout:
                await self._start_gradual_rollout(target_slot)
            else:
                self._switch_active(target_slot)
            
            # Record deployment
            self._deployment_history.append({
                "version": version,
                "slot": target_slot.value,
                "deployed_at": deployment.deployed_at.isoformat(),
                "path": model_path,
            })
            
            logger.info(f"Deployment of v{version} successful")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def get_active_model(self) -> Optional[str]:
        """Get path to currently active model."""
        active = self._get_active_deployment()
        return active.model_path if active else None
    
    def route_request(self) -> DeploymentSlot:
        """
        Route a request to appropriate slot.
        
        Used during gradual rollout to split traffic.
        """
        import random
        
        if self._traffic_fraction >= 1.0:
            return self._active_slot
        
        # During rollout, some traffic goes to new slot
        if random.random() < self._traffic_fraction:
            return self._active_slot
        else:
            return self._get_inactive_slot()
    
    def record_request(
        self,
        slot: DeploymentSlot,
        latency_ms: float,
        error: bool = False,
    ):
        """Record a request for metrics tracking."""
        deployment = self._get_deployment(slot)
        if deployment is None:
            return
        
        deployment.request_count += 1
        if error:
            deployment.error_count += 1
        
        # Update running average of latency
        n = deployment.request_count
        deployment.avg_latency_ms = (
            (deployment.avg_latency_ms * (n - 1) + latency_ms) / n
        )
        
        # Check for auto-rollback conditions
        if self.config.auto_rollback_enabled:
            self._check_rollback_conditions(slot)
    
    async def rollback(self):
        """Rollback to previous active slot."""
        previous_slot = self._get_inactive_slot()
        previous_deployment = self._get_deployment(previous_slot)
        
        if previous_deployment is None:
            logger.error("No previous deployment to rollback to")
            return
        
        logger.warning(f"Rolling back to {previous_slot.value}")
        
        # Cancel any ongoing rollout
        if self._rollout_task:
            self._rollout_task.cancel()
            try:
                await self._rollout_task
            except asyncio.CancelledError:
                pass
        
        # Switch immediately
        self._switch_active(previous_slot)
        self._traffic_fraction = 1.0
        
        logger.info(f"Rollback complete, now active: {previous_slot.value}")
    
    async def _start_gradual_rollout(self, new_slot: DeploymentSlot):
        """Start gradual traffic shifting to new slot."""
        self._traffic_fraction = 1.0 - self.config.initial_traffic_fraction
        
        async def _rollout_loop():
            while self._traffic_fraction > 0:
                await asyncio.sleep(self.config.increment_interval_seconds)
                
                # Check health before increasing traffic
                if self.config.health_check_enabled:
                    new_deployment = self._get_deployment(new_slot)
                    if new_deployment and new_deployment.error_count > 0:
                        error_rate = new_deployment.error_count / max(new_deployment.request_count, 1)
                        if error_rate > self.config.error_rate_threshold:
                            await self.rollback()
                            return
                
                # Decrease active slot traffic (increase new slot traffic)
                self._traffic_fraction -= self.config.traffic_increment
                
                new_traffic = 1.0 - self._traffic_fraction
                logger.info(f"Rollout progress: {new_traffic:.0%} to {new_slot.value}")
            
            # Rollout complete
            self._switch_active(new_slot)
            logger.info(f"Gradual rollout complete, {new_slot.value} is now active")
        
        self._rollout_task = asyncio.create_task(_rollout_loop())
    
    def _switch_active(self, slot: DeploymentSlot):
        """Switch the active slot."""
        old_active = self._get_active_deployment()
        if old_active:
            old_active.is_active = False
        
        self._active_slot = slot
        new_active = self._get_active_deployment()
        if new_active:
            new_active.is_active = True
        
        self._traffic_fraction = 1.0
    
    async def _run_health_check(self, slot: DeploymentSlot) -> bool:
        """Run health check on a deployment."""
        deployment = self._get_deployment(slot)
        if deployment is None:
            return False
        
        try:
            # Simplified health check - verify model file exists
            model_path = Path(deployment.model_path)
            exists = model_path.exists() if model_path.suffix else True
            
            deployment.health_check_passed = exists
            deployment.last_health_check = datetime.utcnow()
            
            return exists
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            deployment.health_check_passed = False
            return False
    
    def _check_rollback_conditions(self, slot: DeploymentSlot):
        """Check if auto-rollback should be triggered."""
        deployment = self._get_deployment(slot)
        if deployment is None or not deployment.is_active:
            return
        
        # Check error rate
        if deployment.request_count > 100:
            error_rate = deployment.error_count / deployment.request_count
            if error_rate > self.config.error_rate_threshold:
                logger.warning(f"Error rate {error_rate:.2%} exceeds threshold")
                asyncio.create_task(self.rollback())
                return
        
        # Check latency
        if deployment.avg_latency_ms > self.config.latency_threshold_ms:
            logger.warning(f"Latency {deployment.avg_latency_ms:.0f}ms exceeds threshold")
            asyncio.create_task(self.rollback())
    
    def _get_active_deployment(self) -> Optional[DeploymentInfo]:
        """Get the active deployment."""
        return self._get_deployment(self._active_slot)
    
    def _get_deployment(self, slot: DeploymentSlot) -> Optional[DeploymentInfo]:
        """Get deployment by slot."""
        if slot == DeploymentSlot.BLUE:
            return self._blue
        else:
            return self._green
    
    def _get_inactive_slot(self) -> DeploymentSlot:
        """Get the inactive slot."""
        if self._active_slot == DeploymentSlot.BLUE:
            return DeploymentSlot.GREEN
        else:
            return DeploymentSlot.BLUE
    
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status."""
        return {
            "model_name": self.model_name,
            "active_slot": self._active_slot.value,
            "traffic_fraction": self._traffic_fraction,
            "blue": self._deployment_status(self._blue),
            "green": self._deployment_status(self._green),
        }
    
    def _deployment_status(self, deployment: Optional[DeploymentInfo]) -> Optional[Dict[str, Any]]:
        """Get status dict for a deployment."""
        if deployment is None:
            return None
        
        return {
            "version": deployment.version,
            "deployed_at": deployment.deployed_at.isoformat(),
            "is_active": deployment.is_active,
            "requests": deployment.request_count,
            "errors": deployment.error_count,
            "avg_latency_ms": deployment.avg_latency_ms,
            "health": deployment.health_check_passed,
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return self._deployment_history.copy()
