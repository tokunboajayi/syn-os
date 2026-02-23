"""
Multi-Agent Reinforcement Learning Coordinator for Syn OS

Enables distributed coordination of multiple PPO agents for global optimization.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    
    agent_id: str
    state_dim: int = 32
    action_dim: int = 8
    hidden_dim: int = 128
    lr: float = 3e-4
    
    # Communication
    message_dim: int = 16
    num_neighbors: int = 4
    
    # Coordination
    coordination_weight: float = 0.3  # Weight for coordination reward


@dataclass
class CoordinatorConfig:
    """Configuration for multi-agent coordinator."""
    
    num_agents: int = 4
    coordination_mode: str = "mean_field"  # mean_field, attention, graph
    global_reward_weight: float = 0.5
    local_reward_weight: float = 0.5
    communication_rounds: int = 2
    
    # Training
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005  # Soft update rate
    
    # Agent config template
    agent_config: AgentConfig = field(default_factory=AgentConfig)


class CommunicationModule(nn.Module):
    """
    Communication module for agents to share information.
    
    Implements mean-field and attention-based communication.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        message_dim: int,
        num_heads: int = 4,
        mode: str = "attention",
    ):
        super().__init__()
        
        self.mode = mode
        
        # Message encoder/decoder
        self.message_encoder = nn.Linear(hidden_dim, message_dim)
        self.message_decoder = nn.Linear(message_dim, hidden_dim)
        
        if mode == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=message_dim,
                num_heads=num_heads,
                batch_first=True,
            )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        agent_states: torch.Tensor,  # [batch, num_agents, hidden_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform communication and return updated states.
        
        Args:
            agent_states: Current hidden states of all agents
            mask: Optional mask for selective communication
            
        Returns:
            Updated agent states incorporating information from others
        """
        batch_size, num_agents, hidden_dim = agent_states.shape
        
        # Encode messages
        messages = self.message_encoder(agent_states)  # [batch, num_agents, message_dim]
        
        if self.mode == "mean_field":
            # Average message from all other agents
            mean_message = messages.mean(dim=1, keepdim=True)  # [batch, 1, message_dim]
            mean_message = mean_message.expand(-1, num_agents, -1)
            aggregated = mean_message
            
        elif self.mode == "attention":
            # Attention-based message aggregation
            aggregated, _ = self.attention(
                messages, messages, messages,
                key_padding_mask=mask,
            )
        else:
            aggregated = messages
        
        # Decode and add to states
        decoded = self.message_decoder(aggregated)
        updated_states = self.layer_norm(agent_states + decoded)
        
        return updated_states


class AgentNetwork(nn.Module):
    """
    Neural network for a single agent in multi-agent setting.
    
    Includes local policy and value estimation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        message_dim: int = 16,
    ):
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Message output for coordination
        self.message_out = nn.Linear(hidden_dim, message_dim)
    
    def forward(
        self,
        state: torch.Tensor,
        received_messages: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_logits: Action probability logits
            value: State value estimate
            hidden: Hidden state for communication
            message: Outgoing message for other agents
        """
        hidden = self.state_encoder(state)
        message = self.message_out(hidden)
        
        # Incorporate received messages
        if received_messages is not None:
            combined = torch.cat([hidden, received_messages], dim=-1)
        else:
            combined = torch.cat([hidden, torch.zeros_like(message)], dim=-1)
        
        action_logits = self.policy(combined)
        value = self.value(combined)
        
        return action_logits, value, hidden, message


class MultiAgentCoordinator:
    """
    Coordinates multiple RL agents for distributed scheduling.
    
    Features:
    - Mean-field or attention-based communication
    - Global reward sharing with local optimization
    - Centralized training, decentralized execution
    - Emergent coordination behaviors
    """
    
    def __init__(self, config: CoordinatorConfig):
        self.config = config
        self.agents: Dict[str, AgentNetwork] = {}
        self.communication: Optional[CommunicationModule] = None
        self._step_count = 0
        
        if HAS_TORCH:
            self._init_networks()
    
    def _init_networks(self):
        """Initialize agent networks and communication module."""
        # Create agents
        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = AgentNetwork(
                state_dim=self.config.agent_config.state_dim,
                action_dim=self.config.agent_config.action_dim,
                hidden_dim=self.config.agent_config.hidden_dim,
                message_dim=self.config.agent_config.message_dim,
            )
        
        # Create communication module
        self.communication = CommunicationModule(
            hidden_dim=self.config.agent_config.hidden_dim,
            message_dim=self.config.agent_config.message_dim,
            mode=self.config.coordination_mode,
        )
        
        # Create optimizers
        all_params = []
        for agent in self.agents.values():
            all_params.extend(agent.parameters())
        all_params.extend(self.communication.parameters())
        
        self.optimizer = torch.optim.Adam(all_params, lr=self.config.agent_config.lr)
        
        logger.info(f"Initialized {self.config.num_agents} agents with {self.config.coordination_mode} communication")
    
    def select_actions(
        self,
        agent_states: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Dict[str, Tuple[int, float]]:
        """
        Select actions for all agents with coordination.
        
        Args:
            agent_states: Dict mapping agent_id to state array
            deterministic: If True, select most likely action
            
        Returns:
            Dict mapping agent_id to (action, log_prob)
        """
        if not HAS_TORCH:
            # Fallback: random actions
            return {
                agent_id: (np.random.randint(0, self.config.agent_config.action_dim), 0.0)
                for agent_id in agent_states
            }
        
        # Convert states to tensors
        states = []
        agent_ids = list(agent_states.keys())
        for agent_id in agent_ids:
            states.append(torch.tensor(agent_states[agent_id], dtype=torch.float32))
        
        states_tensor = torch.stack(states).unsqueeze(0)  # [1, num_agents, state_dim]
        
        # Get initial hidden states and messages
        hiddens = []
        messages = []
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                _, _, hidden, message = agent(states_tensor[0, i:i+1])
                hiddens.append(hidden)
                messages.append(message)
        
        if not hiddens:
            return {}
        
        hidden_tensor = torch.stack(hiddens, dim=1)  # [1, num_agents, hidden_dim]
        
        # Communication rounds
        for _ in range(self.config.communication_rounds):
            hidden_tensor = self.communication(hidden_tensor)
        
        # Get actions with coordinated hidden states
        actions = {}
        
        for i, agent_id in enumerate(agent_ids):
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            
            # Get coordinated message
            coord_message = hidden_tensor[0, i:i+1, :16]  # Take message_dim from hidden
            
            # Forward pass with coordination
            action_logits, value, _, _ = agent(states_tensor[0, i:i+1], coord_message)
            
            # Select action
            probs = F.softmax(action_logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1).item()
                log_prob = torch.log(probs[0, action]).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action_tensor = dist.sample()
                action = action_tensor.item()
                log_prob = dist.log_prob(action_tensor).item()
            
            actions[agent_id] = (action, log_prob)
        
        return actions
    
    def compute_coordinated_reward(
        self,
        local_rewards: Dict[str, float],
        global_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute coordinated rewards combining local and global components.
        
        Args:
            local_rewards: Per-agent local rewards
            global_metrics: System-wide metrics (throughput, utilization, etc.)
            
        Returns:
            Coordinated rewards for each agent
        """
        # Compute global reward component
        global_reward = (
            global_metrics.get("throughput", 0) * 0.4 +
            global_metrics.get("utilization", 0) * 0.3 +
            (1 - global_metrics.get("latency_ratio", 0)) * 0.3
        )
        
        # Blend local and global rewards
        coordinated_rewards = {}
        for agent_id, local_reward in local_rewards.items():
            coordinated_rewards[agent_id] = (
                self.config.local_reward_weight * local_reward +
                self.config.global_reward_weight * global_reward
            )
        
        return coordinated_rewards
    
    def train_step(
        self,
        experiences: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Perform a coordinated training step.
        
        Args:
            experiences: List of experience dicts with states, actions, rewards
            
        Returns:
            Training metrics
        """
        if not HAS_TORCH or not experiences:
            return {}
        
        # Parse experiences
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for exp in experiences:
            states_batch.append(exp.get("states", {}))
            actions_batch.append(exp.get("actions", {}))
            rewards_batch.append(exp.get("rewards", {}))
            next_states_batch.append(exp.get("next_states", {}))
            dones_batch.append(exp.get("done", False))
        
        # Convert to tensors (simplified)
        losses = []
        
        for agent_id, agent in self.agents.items():
            agent_losses = []
            
            for i, exp in enumerate(experiences):
                if agent_id not in states_batch[i]:
                    continue
                
                state = torch.tensor(states_batch[i][agent_id], dtype=torch.float32)
                action = actions_batch[i].get(agent_id, (0, 0))[0]
                reward = rewards_batch[i].get(agent_id, 0)
                
                # Forward pass
                action_logits, value, _, _ = agent(state.unsqueeze(0))
                
                # Policy loss (advantage actor-critic style)
                log_prob = F.log_softmax(action_logits, dim=-1)[0, action]
                advantage = reward - value.squeeze()
                policy_loss = -log_prob * advantage.detach()
                
                # Value loss
                value_loss = F.mse_loss(value.squeeze(), torch.tensor(reward, dtype=torch.float32))
                
                agent_losses.append(policy_loss + 0.5 * value_loss)
            
            if agent_losses:
                losses.append(torch.stack(agent_losses).mean())
        
        if not losses:
            return {}
        
        # Backprop
        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_all_parameters(), 0.5)
        self.optimizer.step()
        
        self._step_count += 1
        
        return {
            "total_loss": total_loss.item(),
            "step": self._step_count,
        }
    
    def get_all_parameters(self):
        """Get all parameters for gradient clipping."""
        params = []
        for agent in self.agents.values():
            params.extend(agent.parameters())
        if self.communication:
            params.extend(self.communication.parameters())
        return params
    
    def save(self, path: str):
        """Save coordinator state."""
        if not HAS_TORCH:
            return
        
        import torch
        state = {
            "agents": {
                agent_id: agent.state_dict()
                for agent_id, agent in self.agents.items()
            },
            "communication": self.communication.state_dict() if self.communication else None,
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "config": self.config,
        }
        torch.save(state, path)
        logger.info(f"Saved multi-agent coordinator to {path}")
    
    def load(self, path: str):
        """Load coordinator state."""
        if not HAS_TORCH:
            return
        
        import torch
        state = torch.load(path)
        
        for agent_id, agent_state in state["agents"].items():
            if agent_id in self.agents:
                self.agents[agent_id].load_state_dict(agent_state)
        
        if self.communication and state["communication"]:
            self.communication.load_state_dict(state["communication"])
        
        self.optimizer.load_state_dict(state["optimizer"])
        self._step_count = state["step_count"]
        
        logger.info(f"Loaded multi-agent coordinator from {path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "num_agents": len(self.agents),
            "coordination_mode": self.config.coordination_mode,
            "step_count": self._step_count,
            "agents": list(self.agents.keys()),
        }
