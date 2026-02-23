"""
Hyperparameter Auto-Tuning for Syn OS

Implements Bayesian optimization with Optuna for:
- Model hyperparameters (learning rate, hidden dims, etc.)
- Training hyperparameters (batch size, epochs, etc.)
- Architecture search (layer count, attention heads, etc.)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

try:
    import optuna
    from optuna import Trial
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    Trial = Any


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    
    # Study configuration
    study_name: str = "synos_tuning"
    direction: str = "minimize"  # minimize or maximize
    n_trials: int = 100
    timeout_seconds: Optional[int] = 3600  # 1 hour default
    
    # Pruning
    enable_pruning: bool = True
    pruning_warmup_steps: int = 10
    pruning_interval: int = 5
    
    # Parallelization
    n_jobs: int = 1
    
    # Storage
    storage_path: Optional[str] = None  # SQLite path for persistence
    
    # Sampling
    sampler_type: str = "tpe"  # tpe, cmaes, random


@dataclass
class SearchSpace:
    """Defines the hyperparameter search space."""
    
    # Learning rate
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    lr_log: bool = True
    
    # Hidden dimensions
    hidden_dim_choices: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Number of layers
    num_layers_min: int = 1
    num_layers_max: int = 6
    
    # Dropout
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    
    # Batch size
    batch_size_choices: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    
    # Activation function
    activation_choices: List[str] = field(default_factory=lambda: ["relu", "gelu", "swish"])
    
    # Attention heads
    attention_heads_choices: List[int] = field(default_factory=lambda: [2, 4, 8])
    
    # Optimizer
    optimizer_choices: List[str] = field(default_factory=lambda: ["adam", "adamw", "sgd"])
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


class HyperparameterTuner:
    """
    Bayesian hyperparameter optimization using Optuna.
    
    Features:
    - TPE (Tree-structured Parzen Estimator) sampling
    - Automated pruning of unpromising trials
    - Multi-objective optimization support
    - Persistent storage for resumable studies
    """
    
    def __init__(
        self,
        config: TuningConfig,
        search_space: SearchSpace,
    ):
        self.config = config
        self.search_space = search_space
        self._study: Optional["optuna.Study"] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._history: List[Dict[str, Any]] = []
    
    def _create_study(self) -> "optuna.Study":
        """Create or load Optuna study."""
        if not HAS_OPTUNA:
            raise RuntimeError("Optuna not installed. pip install optuna")
        
        # Set up sampler
        if self.config.sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif self.config.sampler_type == "cmaes":
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.RandomSampler()
        
        # Set up pruner
        if self.config.enable_pruning:
            pruner = optuna.pruners.MedianPruner(
                n_warmup_steps=self.config.pruning_warmup_steps,
                interval_steps=self.config.pruning_interval,
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Storage
        storage = None
        if self.config.storage_path:
            storage = f"sqlite:///{self.config.storage_path}"
        
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
        
        return study
    
    def suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dict of suggested hyperparameters
        """
        space = self.search_space
        
        params = {
            # Learning rate
            "learning_rate": trial.suggest_float(
                "learning_rate",
                space.lr_min,
                space.lr_max,
                log=space.lr_log,
            ),
            
            # Hidden dimension
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim",
                space.hidden_dim_choices,
            ),
            
            # Number of layers
            "num_layers": trial.suggest_int(
                "num_layers",
                space.num_layers_min,
                space.num_layers_max,
            ),
            
            # Dropout
            "dropout": trial.suggest_float(
                "dropout",
                space.dropout_min,
                space.dropout_max,
            ),
            
            # Batch size
            "batch_size": trial.suggest_categorical(
                "batch_size",
                space.batch_size_choices,
            ),
            
            # Activation
            "activation": trial.suggest_categorical(
                "activation",
                space.activation_choices,
            ),
            
            # Attention heads
            "attention_heads": trial.suggest_categorical(
                "attention_heads",
                space.attention_heads_choices,
            ),
            
            # Optimizer
            "optimizer": trial.suggest_categorical(
                "optimizer",
                space.optimizer_choices,
            ),
        }
        
        # Add custom parameters
        for name, param_config in space.custom_params.items():
            if param_config.get("type") == "float":
                params[name] = trial.suggest_float(
                    name,
                    param_config["min"],
                    param_config["max"],
                    log=param_config.get("log", False),
                )
            elif param_config.get("type") == "int":
                params[name] = trial.suggest_int(
                    name,
                    param_config["min"],
                    param_config["max"],
                )
            elif param_config.get("type") == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    param_config["choices"],
                )
        
        return params
    
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_fn: Function that takes params dict and returns objective value
            callbacks: Optional list of callback functions
            
        Returns:
            Best hyperparameters found
        """
        if not HAS_OPTUNA:
            logger.warning("Optuna not available, returning default params")
            return self._get_default_params()
        
        self._study = self._create_study()
        
        def wrapped_objective(trial: Trial) -> float:
            params = self.suggest_hyperparameters(trial)
            
            try:
                objective = objective_fn(params)
                
                # Record in history
                self._history.append({
                    "trial_number": trial.number,
                    "params": params,
                    "objective": objective,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
                return objective
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned()
        
        # Run optimization
        logger.info(f"Starting optimization study: {self.config.study_name}")
        
        self._study.optimize(
            wrapped_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs,
            callbacks=callbacks,
        )
        
        # Store best parameters
        self._best_params = self._study.best_params
        
        logger.info(f"Optimization complete. Best value: {self._study.best_value}")
        logger.info(f"Best params: {self._best_params}")
        
        return self._best_params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters (fallback when Optuna unavailable)."""
        space = self.search_space
        return {
            "learning_rate": 1e-3,
            "hidden_dim": space.hidden_dim_choices[1],
            "num_layers": 3,
            "dropout": 0.1,
            "batch_size": space.batch_size_choices[2],
            "activation": "relu",
            "attention_heads": 4,
            "optimizer": "adam",
        }
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found."""
        return self._best_params
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self._history.copy()
    
    def save_results(self, path: str):
        """Save optimization results."""
        results = {
            "study_name": self.config.study_name,
            "best_params": self._best_params,
            "best_value": self._study.best_value if self._study else None,
            "n_trials": len(self._history),
            "history": self._history,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved tuning results to {path}")


class ModelTuner:
    """
    High-level model tuning interface for Syn OS models.
    """
    
    def __init__(
        self,
        model_name: str,
        search_space: Optional[SearchSpace] = None,
        config: Optional[TuningConfig] = None,
    ):
        self.model_name = model_name
        self.search_space = search_space or SearchSpace()
        self.config = config or TuningConfig(study_name=f"synos_{model_name}")
        self._tuner = HyperparameterTuner(self.config, self.search_space)
    
    def tune_execution_predictor(
        self,
        train_data: Tuple[Any, Any],
        val_data: Tuple[Any, Any],
    ) -> Dict[str, Any]:
        """Tune execution time predictor model."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        def objective(params: Dict[str, Any]) -> float:
            from synos_ml.models.predictor import ExecutionTimePredictor
            
            predictor = ExecutionTimePredictor(
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
            )
            
            # Train
            predictor.train_neural(
                X_train, y_train,
                epochs=50,  # Reduced for tuning
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate"],
            )
            
            # Evaluate
            metrics = predictor.evaluate(X_val, y_val)
            return metrics.get("mse", float("inf"))
        
        return self._tuner.optimize(objective)
    
    def tune_forecaster(
        self,
        train_data: Tuple[Any, Any],
        val_data: Tuple[Any, Any],
    ) -> Dict[str, Any]:
        """Tune demand forecaster model."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Add forecaster-specific search space
        self.search_space.custom_params.update({
            "transformer_layers": {
                "type": "int",
                "min": 1,
                "max": 4,
            },
            "lstm_hidden": {
                "type": "categorical",
                "choices": [64, 128, 256],
            },
        })
        
        def objective(params: Dict[str, Any]) -> float:
            from synos_ml.models.forecaster import TransformerLSTMHybrid
            
            model = TransformerLSTMHybrid(
                input_features=X_train.shape[-1],
                lstm_hidden=params.get("lstm_hidden", 128),
                d_model=params["hidden_dim"],
                n_heads=params["attention_heads"],
                transformer_layers=params.get("transformer_layers", 2),
                dropout=params["dropout"],
            )
            
            # Simplified training for tuning
            import torch
            import torch.nn.functional as F
            
            optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
            
            X_t = torch.tensor(X_train, dtype=torch.float32)
            y_t = torch.tensor(y_train, dtype=torch.float32)
            
            model.train()
            for epoch in range(20):
                optimizer.zero_grad()
                predictions, _ = model(X_t)
                # Use first horizon for validation
                loss = F.mse_loss(predictions[6][:, :y_t.shape[1]], y_t)
                loss.backward()
                optimizer.step()
            
            # Validation loss
            model.eval()
            X_v = torch.tensor(X_val, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.float32)
            
            with torch.no_grad():
                preds, _ = model(X_v)
                val_loss = F.mse_loss(preds[6][:, :y_v.shape[1]], y_v)
            
            return val_loss.item()
        
        return self._tuner.optimize(objective)
    
    def tune_ppo_scheduler(
        self,
        env,  # Gym-like environment
        n_eval_episodes: int = 10,
    ) -> Dict[str, Any]:
        """Tune PPO scheduler hyperparameters."""
        # PPO-specific search space
        ppo_space = SearchSpace(
            lr_min=1e-5,
            lr_max=1e-3,
            hidden_dim_choices=[64, 128, 256],
            custom_params={
                "gamma": {"type": "float", "min": 0.9, "max": 0.999},
                "gae_lambda": {"type": "float", "min": 0.9, "max": 0.99},
                "clip_ratio": {"type": "float", "min": 0.1, "max": 0.3},
                "value_coef": {"type": "float", "min": 0.1, "max": 1.0},
                "entropy_coef": {"type": "float", "min": 0.0, "max": 0.1},
            }
        )
        self._tuner = HyperparameterTuner(self.config, ppo_space)
        
        def objective(params: Dict[str, Any]) -> float:
            from synos_ml.scheduler.ppo import PPOScheduler, PPOConfig
            
            config = PPOConfig(
                state_dim=env.observation_space.shape[0],
                num_resources=env.action_space.n,
                hidden_dim=params["hidden_dim"],
                lr=params["learning_rate"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                clip_ratio=params["clip_ratio"],
                value_coef=params["value_coef"],
                entropy_coef=params["entropy_coef"],
            )
            
            scheduler = PPOScheduler(config)
            
            # Training episodes
            total_reward = 0
            for episode in range(n_eval_episodes):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _, _ = scheduler.choose_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                    
                    if truncated:
                        break
                
                total_reward += episode_reward
            
            # Negative because we minimize
            return -total_reward / n_eval_episodes
        
        return self._tuner.optimize(objective)


def quick_tune(
    model_name: str,
    train_fn: Callable,
    eval_fn: Callable,
    n_trials: int = 20,
) -> Dict[str, Any]:
    """
    Quick hyperparameter tuning helper.
    
    Args:
        model_name: Name of the model
        train_fn: Function that trains model given params
        eval_fn: Function that evaluates trained model
        n_trials: Number of trials
        
    Returns:
        Best parameters
    """
    config = TuningConfig(
        study_name=f"quick_tune_{model_name}",
        n_trials=n_trials,
        timeout_seconds=600,  # 10 minutes
    )
    
    tuner = HyperparameterTuner(config, SearchSpace())
    
    def objective(params: Dict[str, Any]) -> float:
        model = train_fn(params)
        return eval_fn(model)
    
    return tuner.optimize(objective)
