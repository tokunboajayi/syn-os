"""
Core Scheduler Module
Integrates ML models to make intelligent scheduling decisions.
"""

import logging
import random
from typing import Dict, List, Optional
from datetime import datetime

# Import ML models (mocked or real)
try:
    from api.security.ml.anomaly import anomaly_detector
    from ml.synos_ml.models.predictor import ExecutionTimePredictor, TaskPredictionFeatures
    from ml.synos_ml.scheduler.ppo import PPOScheduler
    from ml.synos_ml.models.gnn import TaskGNN
    from ml.synos_ml.models.forecaster import TransformerLSTMHybrid
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML modules not found, using heuristics: {e}")
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class SynOSScheduler:
    """
    AI-Powered Scheduler.
    Uses:
    1. ExecutionTimePredictor -> Estimates task duration
    2. PPOScheduler -> Selects optimal node/resource
    3. TaskGNN -> Optimizes dependency execution order
    """
    
    def __init__(self):
        self.predictor = None
        self.ppo_scheduler = None
        self.forecaster = None
        
        if ML_AVAILABLE:
            try:
                self.predictor = ExecutionTimePredictor()
                self.ppo_scheduler = PPOScheduler()
                self.forecaster = TransformerLSTMHybrid()
                logger.info("Scheduler initialized with ML models")
            except Exception as e:
                logger.error(f"Failed to initialize ML models: {e}")
    
    def forecast_demand(self, hours_ahead: int) -> Dict:
        """
        Forecast system resource demand.
        """
        if not self.forecaster:
            return {
                "cpu": {"mean": 0.5, "std": 0.1},
                "memory": {"mean": 0.6, "std": 0.1},
                "io": {"mean": 0.3, "std": 0.1},
                "confidence_level": 0.5
            }
            
        try:
            # Mock input sequence (last 60 minutes)
            # In real system, fetch from Prometheus
            import torch
            mock_input = torch.randn(1, 60, 4) # Batch, seq, features
            
            with torch.no_grad():
                cpu_pred, mem_pred = self.forecaster(mock_input)
                
            # Average predictions for the horizon
            return {
                "cpu": {"mean": float(cpu_pred.mean()), "std": float(cpu_pred.std())},
                "memory": {"mean": float(mem_pred.mean()), "std": float(mem_pred.std())},
                "io": {"mean": 0.45, "std": 0.15}, # Not yet in model
                "confidence_level": 0.85
            }
        except Exception as e:
            logger.error(f"Forecasting error: {e}")
            return {
                "cpu": {"mean": 0.5, "std": 0.1},
                "memory": {"mean": 0.6, "std": 0.1},
                "io": {"mean": 0.3, "std": 0.1},
                "confidence_level": 0.5
            }

    def predict_duration(self, task_data: Dict) -> int:
        """
        Predict execution time in milliseconds.
        """
        if not self.predictor:
            return 1000 # Default fallback
            
        try:
            # Extract features from task_data
            resources = task_data.get('resources', {})
            features = TaskPredictionFeatures(
                cpu_cores=resources.get('cpu_cores', 1),
                memory_mb=resources.get('memory_mb', 1024),
                priority=task_data.get('priority', 5),
                hour_of_day=datetime.now().hour,
                day_of_week=datetime.now().weekday()
            )
            
            prediction = self.predictor.predict(features)
            return int(prediction)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 1000

    def assign_node(self, task_data: Dict, available_nodes: List[Dict]) -> str:
        """
        Assign task to the best available node using PPO.
        """
        if not available_nodes:
            return "queued"
            
        if not self.ppo_scheduler:
            # Simple round-robin or random fallback
            return random.choice(available_nodes)['id']
            
        try:
            # Construct state vector for PPO
            # (Simplified for now - real implementation would aggregate cluster state)
            import numpy as np
            state = np.zeros(32) # Placeholder state dim
            
            # Action is index of node
            action, _, _ = self.ppo_scheduler.choose_action(state, deterministic=True)
            
            # Map action index to valid node
            node_idx = action % len(available_nodes)
            return available_nodes[node_idx]['id']
            
        except Exception as e:
            logger.error(f"PPO scheduling error: {e}")
            return available_nodes[0]['id']

    def optimize_queue(self, queued_tasks: List[Dict]) -> List[Dict]:
        """
        Re-order queued tasks based on GNN dependency analysis.
        Calculates a 'critical path' score for each task.
        """
        if not queued_tasks or len(queued_tasks) < 2:
            return queued_tasks
            
        # If no ML, just sort by priority (0 is highest)
        if not ML_AVAILABLE:
            return sorted(queued_tasks, key=lambda t: t.get('priority', 5))
            
        try:
            # In a real implementation:
            # 1. Build adjacency matrix from dependencies
            # 2. Construct graph data (node features, edge index)
            # 3. Pass to self.task_gnn(x, edge_index)
            # 4. Use output scores to sort tasks
            
            # Mock GNN logic for prototype:
            # Boost priority of tasks that have many dependents
            
            # Map task_id -> list of dependent tasks
            dependents = {t['id']: [] for t in queued_tasks}
            for t in queued_tasks:
                for dep in t.get('dependencies', []):
                    dep_id = dep.get('task_id')
                    if dep_id in dependents:
                        dependents[dep_id].append(t['id'])
            
            # Calculate score based on number of dependents (criticality)
            for t in queued_tasks:
                num_dependents = len(dependents.get(t['id'], []))
                # GNN would predict this score complexity
                t['gnn_score'] = num_dependents * 10 + (10 - t.get('priority', 5))
                
            # Sort by GNN score descending
            return sorted(queued_tasks, key=lambda t: t.get('gnn_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"GNN optimization error: {e}")
            return sorted(queued_tasks, key=lambda t: t.get('priority', 5))


# Global instance
scheduler = SynOSScheduler()
