"""
PPO (Proximal Policy Optimization) Scheduler

Learns optimal task-to-resource assignment policies through
reinforcement learning.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from loguru import logger


@dataclass
class Experience:
    """Single experience tuple for RL training."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class ExperienceBuffer:
    """Experience replay buffer for PPO."""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, exp: Experience):
        """Add an experience to the buffer."""
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def get_all(self) -> List[Experience]:
        """Get all experiences."""
        return list(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Actor: Outputs action probabilities (which resource to assign)
    Critic: Estimates state value (expected future reward)
    """

    def __init__(
        self,
        state_dim: int = 32,  # Task + system state features
        action_dim: int = 16,  # Number of resources/nodes
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        # Actor output layer with smaller initialization
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

        # Critic output layer
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor [batch, state_dim]

        Returns:
            action_logits: [batch, action_dim]
            value: [batch, 1]
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, float]:
        """
        Sample action from policy.

        Args:
            state: Current state
            deterministic: Whether to use greedy action selection
            action_mask: Optional mask for valid actions (1 = valid, 0 = invalid)

        Returns:
            action: Selected action index
            log_prob: Log probability of action
            value: State value estimate
        """
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = self.forward(state_t)

            # Apply action mask if provided
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.bool)
                logits = logits.masked_fill(~mask_t.unsqueeze(0), float("-inf"))

            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = probs.argmax().item()
            else:
                action = dist.sample().item()

            log_prob = dist.log_prob(torch.tensor(action)).item()

            return action, log_prob, value.item()

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.

        Args:
            states: Batch of states [batch, state_dim]
            actions: Batch of actions [batch]

        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


@dataclass
class PPOConfig:
    """PPO training configuration."""

    state_dim: int = 32
    num_resources: int = 16
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    buffer_size: int = 10000


class PPOScheduler:
    """
    PPO-based scheduler that learns optimal task placement.

    State space:
    - Task features: cpu, memory, priority, estimated_duration
    - System state: resource utilization, queue depths, load per node

    Action space:
    - Which node/resource to assign the task to

    Reward:
    - +1.0 for on-time completion
    - -0.5 for SLA violation
    - +0.2 * efficiency_score (resource utilization improvement)
    - -0.1 for each queue minute
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        if config is None:
            config = PPOConfig()

        self.config = config
        self.network = PPOActorCritic(
            state_dim=config.state_dim,
            action_dim=config.num_resources,
            hidden_dim=config.hidden_dim,
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=config.lr,
            eps=1e-5,
        )
        self.buffer = ExperienceBuffer(config.buffer_size)

        # Training metrics
        self.training_step = 0
        self.episode_rewards: List[float] = []

    def choose_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, float]:
        """Select which resource to assign task to."""
        return self.network.get_action(state, deterministic, action_mask)

    def store_experience(self, exp: Experience):
        """Store experience for training."""
        self.buffer.add(exp)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            next_value: Bootstrap value for last state

        Returns:
            returns: Computed returns
            advantages: Computed advantages
        """
        advantages = []
        returns = []
        gae = 0.0

        # Extend values with bootstrap
        values = values + [next_value]

        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_val = 0.0
                gae = 0.0
            else:
                next_val = values[i + 1]

            delta = rewards[i] + self.config.gamma * next_val - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        return returns, advantages

    def train_step(self) -> Dict[str, float]:
        """Perform one PPO training step."""
        if len(self.buffer) < self.config.batch_size:
            return {"loss": 0.0}

        experiences = self.buffer.get_all()

        # Extract data
        states = torch.tensor(
            np.array([e.state for e in experiences]), dtype=torch.float32
        )
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        old_log_probs = torch.tensor(
            [e.log_prob for e in experiences], dtype=torch.float32
        )
        rewards = [e.reward for e in experiences]
        values = [e.value for e in experiences]
        dones = [e.done for e in experiences]

        # Compute returns and advantages
        returns, advantages = self.compute_gae(rewards, values, dones)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update with multiple epochs
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        n_samples = len(experiences)
        indices = np.arange(n_samples)

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate actions
                new_log_probs, new_values, entropy = self.network.evaluate_actions(
                    batch_states, batch_actions
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.buffer.clear()
        self.training_step += 1

        return {
            "loss": total_loss / max(num_updates, 1),
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "training_step": self.training_step,
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved PPO checkpoint to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        logger.info(f"Loaded PPO checkpoint from {path}")


# Reward functions
def compute_scheduling_reward(
    task_completed: bool,
    sla_met: bool,
    queue_wait_minutes: float,
    resource_efficiency: float,
    task_priority: int,
) -> float:
    """
    Compute reward for a scheduling decision.

    Args:
        task_completed: Did the task complete successfully?
        sla_met: Was the SLA deadline met?
        queue_wait_minutes: Time spent waiting in queue
        resource_efficiency: Resource utilization during execution (0-1)
        task_priority: Task priority (0-9, lower = higher priority)

    Returns:
        Computed reward value
    """
    reward = 0.0

    # Completion reward
    if task_completed:
        reward += 1.0
    else:
        reward -= 0.5

    # SLA reward/penalty
    if sla_met:
        reward += 0.5
    else:
        reward -= 1.0

    # Queue wait penalty (scaled by priority)
    priority_weight = 1.0 + (9 - task_priority) / 9  # Higher weight for high priority
    reward -= 0.05 * queue_wait_minutes * priority_weight

    # Efficiency bonus
    reward += 0.2 * resource_efficiency

    return reward
