"""
Tests for ML models.
"""

import pytest
import numpy as np
import torch


class TestTaskGNN:
    """Tests for TaskGNN model."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        from synos_ml.models.gnn import TaskGNN
        
        model = TaskGNN(node_features=16, hidden_dim=32, num_heads=2, num_layers=2)
        
        # Create dummy input
        x = torch.randn(10, 16)  # 10 nodes, 16 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        priority, cluster, duration = model(x, edge_index)
        
        assert priority.shape == (10, 1)
        assert cluster.shape == (10, 32)  # output_dim
        assert duration.shape == (10, 1)
        
    def test_execution_order(self):
        """Test execution order computation."""
        from synos_ml.models.gnn import TaskGNN
        
        model = TaskGNN()
        
        # Simple linear DAG: 0 -> 1 -> 2
        priority_scores = torch.tensor([[0.9], [0.5], [0.1]])
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        order = model.get_execution_order(priority_scores, edge_index, num_nodes=3)
        
        # Must respect dependencies
        assert order.index(0) < order.index(1)
        assert order.index(1) < order.index(2)


class TestForecaster:
    """Tests for Transformer-LSTM forecaster."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        from synos_ml.models.forecaster import TransformerLSTMHybrid
        
        model = TransformerLSTMHybrid(
            input_features=4,
            lstm_hidden=32,
            d_model=64,
            n_heads=4,
            transformer_layers=2,
        )
        
        # Batch of sequences
        x = torch.randn(8, 24, 4)  # batch=8, seq_len=24, features=4
        
        predictions, _ = model(x)
        
        assert 6 in predictions
        assert 24 in predictions
        assert 168 in predictions
        assert "uncertainty" in predictions
        
    def test_forecast(self):
        """Test forecast method."""
        from synos_ml.models.forecaster import TransformerLSTMHybrid
        
        model = TransformerLSTMHybrid(input_features=4)
        
        historical = np.random.randn(48, 4).astype(np.float32)
        
        mean, std = model.forecast(historical, horizon=6)
        
        assert mean.shape == (4,)
        assert std.shape == (4,)


class TestPredictor:
    """Tests for execution time predictor."""

    def test_predict(self):
        """Test prediction."""
        from synos_ml.models.predictor import ExecutionTimePredictor, TaskPredictionFeatures
        
        predictor = ExecutionTimePredictor()
        
        features = TaskPredictionFeatures(
            cpu_cores=4,
            memory_mb=8192,
            current_cpu_util=0.5,
            current_memory_util=0.6,
        )
        
        prediction = predictor.predict(features)
        
        assert isinstance(prediction, float)
        assert prediction > 0
        
    def test_batch_predict(self):
        """Test batch prediction."""
        from synos_ml.models.predictor import ExecutionTimePredictor
        
        predictor = ExecutionTimePredictor()
        
        X = np.random.randn(10, 12).astype(np.float32)
        predictions = predictor.predict(X)
        
        assert predictions.shape == (10,)


class TestAnomalyDetector:
    """Tests for anomaly detector."""

    def test_detect(self):
        """Test anomaly detection."""
        from synos_ml.models.anomaly import AnomalyDetector, AnomalyFeatures
        
        detector = AnomalyDetector(input_dim=15)
        
        # Fit on normal data
        normal_data = np.random.randn(100, 15).astype(np.float32)
        detector.fit(normal_data, epochs=5)
        
        # Test detection
        features = AnomalyFeatures(
            execution_time_ms=1000,
            memory_used_mb=512,
            cpu_time_ms=500,
            exit_code=0,
            system_cpu_util=0.5,
            system_memory_util=0.6,
            system_io_util=0.3,
            queue_depth=10,
            active_tasks=5,
            task_arrival_rate=10,
            task_completion_rate=9,
            error_rate=0.01,
        )
        
        result = detector.detect(features)
        
        assert hasattr(result, "is_anomaly")
        assert hasattr(result, "anomaly_score")


class TestPPOScheduler:
    """Tests for PPO scheduler."""

    def test_choose_action(self):
        """Test action selection."""
        from synos_ml.scheduler.ppo import PPOScheduler, PPOConfig
        
        config = PPOConfig(state_dim=16, num_resources=4)
        scheduler = PPOScheduler(config)
        
        state = np.random.randn(16).astype(np.float32)
        action, log_prob, value = scheduler.choose_action(state)
        
        assert 0 <= action < 4
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        
    def test_train_step(self):
        """Test training step."""
        from synos_ml.scheduler.ppo import PPOScheduler, PPOConfig, Experience
        
        config = PPOConfig(state_dim=16, num_resources=4, batch_size=32)
        scheduler = PPOScheduler(config)
        
        # Add experiences
        for _ in range(64):
            state = np.random.randn(16).astype(np.float32)
            action, log_prob, value = scheduler.choose_action(state)
            
            scheduler.store_experience(Experience(
                state=state,
                action=action,
                reward=np.random.uniform(-1, 1),
                next_state=np.random.randn(16).astype(np.float32),
                done=False,
                log_prob=log_prob,
                value=value,
            ))
        
        metrics = scheduler.train_step()
        
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
