//! Pluggable scheduler interface for Syn OS
//!
//! Defines the Scheduler trait and provides multiple implementations:
//! - RoundRobinScheduler: Simple fallback scheduler
//! - WeightedScheduler: Priority-weighted selection
//! - MLScheduler: Machine learning powered (calls Python)

use crate::task::{Task, TaskId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, instrument};

/// Resource assignment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAssignment {
    /// Node ID where task will run
    pub node_id: String,
    /// Assigned CPU cores
    pub cpu_cores: Vec<u8>,
    /// Memory region (start, size)
    pub memory_region: (u64, u64),
    /// GPU ID if assigned
    pub gpu_id: Option<u8>,
}

/// Scheduling decision with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingDecision {
    /// Resource assignment
    pub assignment: ResourceAssignment,
    /// Estimated time until task starts (ms)
    pub estimated_start_ms: u64,
    /// Estimated time until task completes (ms)
    pub estimated_completion_ms: u64,
    /// Confidence in this decision (0.0 - 1.0)
    pub confidence: f32,
    /// Optional reason for this decision
    pub reason: Option<String>,
}

/// Scheduling feedback for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingFeedback {
    /// Actual execution duration
    pub actual_duration_ms: u64,
    /// Whether task succeeded
    pub success: bool,
    /// Resource utilization during execution
    pub resource_utilization: f32,
    /// Queue wait time before execution
    pub wait_time_ms: u64,
}

/// Scheduler trait - pluggable implementations
#[async_trait]
pub trait Scheduler: Send + Sync {
    /// Schedule a single task
    async fn schedule(&self, task: &Task) -> Option<SchedulingDecision>;

    /// Batch schedule multiple tasks (can optimize globally)
    async fn schedule_batch(&self, tasks: &[Task]) -> Vec<Option<SchedulingDecision>> {
        let mut results = Vec::with_capacity(tasks.len());
        for task in tasks {
            results.push(self.schedule(task).await);
        }
        results
    }

    /// Update scheduler with execution feedback
    async fn feedback(&self, task_id: &TaskId, feedback: SchedulingFeedback);

    /// Get scheduler name
    fn name(&self) -> &str;

    /// Get scheduler statistics
    fn stats(&self) -> SchedulerStats;
}

/// Scheduler statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Total scheduling decisions made
    pub total_decisions: u64,
    /// Total feedback received
    pub total_feedback: u64,
    /// Average decision time (microseconds)
    pub avg_decision_time_us: f64,
    /// Average prediction accuracy (0.0 - 1.0)
    pub accuracy: f32,
}

// ============ Round Robin Scheduler ============

/// Simple round-robin scheduler (fallback)
pub struct RoundRobinScheduler {
    nodes: Vec<String>,
    current: AtomicUsize,
    stats: parking_lot::RwLock<SchedulerStats>,
}

impl RoundRobinScheduler {
    /// Create a new round-robin scheduler
    pub fn new(nodes: Vec<String>) -> Self {
        Self {
            nodes,
            current: AtomicUsize::new(0),
            stats: parking_lot::RwLock::new(SchedulerStats::default()),
        }
    }

    /// Add a node to the scheduler
    pub fn add_node(&mut self, node: String) {
        self.nodes.push(node);
    }

    /// Remove a node from the scheduler
    pub fn remove_node(&mut self, node: &str) {
        self.nodes.retain(|n| n != node);
    }
}

#[async_trait]
impl Scheduler for RoundRobinScheduler {
    #[instrument(skip(self, task), fields(task_id = %task.id))]
    async fn schedule(&self, task: &Task) -> Option<SchedulingDecision> {
        if self.nodes.is_empty() {
            return None;
        }

        let idx = self.current.fetch_add(1, Ordering::Relaxed) % self.nodes.len();
        let node_id = self.nodes[idx].clone();

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_decisions += 1;
        }

        debug!("Round-robin assigned task {} to node {}", task.id, node_id);

        Some(SchedulingDecision {
            assignment: ResourceAssignment {
                node_id,
                cpu_cores: (0..task.resources.cpu_cores).collect(),
                memory_region: (0, task.resources.memory_mb * 1024 * 1024),
                gpu_id: None,
            },
            estimated_start_ms: 0,
            estimated_completion_ms: task.predicted_duration_ms.unwrap_or(1000),
            confidence: 0.5, // Low confidence for simple scheduler
            reason: Some("Round-robin selection".to_string()),
        })
    }

    async fn feedback(&self, _task_id: &TaskId, _feedback: SchedulingFeedback) {
        // Round-robin doesn't learn, just update stats
        let mut stats = self.stats.write();
        stats.total_feedback += 1;
    }

    fn name(&self) -> &str {
        "round-robin"
    }

    fn stats(&self) -> SchedulerStats {
        self.stats.read().clone()
    }
}

// ============ Weighted Scheduler ============

/// Weighted scheduler that considers node load
pub struct WeightedScheduler {
    nodes: Arc<dashmap::DashMap<String, NodeState>>,
    stats: parking_lot::RwLock<SchedulerStats>,
}

/// State of a compute node
#[derive(Debug, Clone, Default)]
pub struct NodeState {
    /// Current CPU utilization (0.0 - 1.0)
    pub cpu_util: f32,
    /// Current memory utilization (0.0 - 1.0)
    pub memory_util: f32,
    /// Number of tasks currently running
    pub running_tasks: usize,
    /// Total CPU cores
    pub total_cpu: u8,
    /// Total memory MB
    pub total_memory_mb: u64,
    /// Is the node healthy?
    pub healthy: bool,
}

impl NodeState {
    /// Calculate node weight (lower = better candidate)
    pub fn weight(&self) -> f32 {
        if !self.healthy {
            return f32::MAX;
        }
        // Combine utilization metrics
        (self.cpu_util * 0.5 + self.memory_util * 0.3 + (self.running_tasks as f32 * 0.1))
    }

    /// Check if node can accept task with given requirements
    pub fn can_accept(&self, cpu_cores: u8, memory_mb: u64) -> bool {
        if !self.healthy {
            return false;
        }
        let available_cpu = ((1.0_f32 - self.cpu_util) * self.total_cpu as f32) as u8;
        let available_mem = ((1.0_f32 - self.memory_util) * self.total_memory_mb as f32) as u64;
        available_cpu >= cpu_cores && available_mem >= memory_mb
    }
}

impl WeightedScheduler {
    /// Create a new weighted scheduler
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(dashmap::DashMap::new()),
            stats: parking_lot::RwLock::new(SchedulerStats::default()),
        }
    }

    /// Register a node
    pub fn register_node(&self, node_id: String, total_cpu: u8, total_memory_mb: u64) {
        self.nodes.insert(
            node_id,
            NodeState {
                total_cpu,
                total_memory_mb,
                healthy: true,
                ..Default::default()
            },
        );
    }

    /// Update node state
    pub fn update_node(&self, node_id: &str, cpu_util: f32, memory_util: f32, running: usize) {
        if let Some(mut node) = self.nodes.get_mut(node_id) {
            node.cpu_util = cpu_util;
            node.memory_util = memory_util;
            node.running_tasks = running;
        }
    }

    /// Mark node as unhealthy
    pub fn mark_unhealthy(&self, node_id: &str) {
        if let Some(mut node) = self.nodes.get_mut(node_id) {
            node.healthy = false;
        }
    }
}

impl Default for WeightedScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Scheduler for WeightedScheduler {
    #[instrument(skip(self, task), fields(task_id = %task.id))]
    async fn schedule(&self, task: &Task) -> Option<SchedulingDecision> {
        // Find best node based on weight
        let mut best_node: Option<(String, f32)> = None;

        for entry in self.nodes.iter() {
            let node_id = entry.key();
            let state = entry.value();

            if !state.can_accept(task.resources.cpu_cores, task.resources.memory_mb) {
                continue;
            }

            let weight = state.weight();
            if best_node.is_none() || weight < best_node.as_ref().unwrap().1 {
                best_node = Some((node_id.clone(), weight));
            }
        }

        let (node_id, weight) = best_node?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_decisions += 1;
        }

        debug!(
            "Weighted scheduler assigned task {} to node {} (weight: {:.2})",
            task.id, node_id, weight
        );

        Some(SchedulingDecision {
            assignment: ResourceAssignment {
                node_id,
                cpu_cores: (0..task.resources.cpu_cores).collect(),
                memory_region: (0, task.resources.memory_mb * 1024 * 1024),
                gpu_id: None,
            },
            estimated_start_ms: 0,
            estimated_completion_ms: task.predicted_duration_ms.unwrap_or(1000),
            confidence: 0.7,
            reason: Some(format!("Weighted selection (weight: {:.2})", weight)),
        })
    }

    async fn feedback(&self, _task_id: &TaskId, feedback: SchedulingFeedback) {
        let mut stats = self.stats.write();
        stats.total_feedback += 1;
        // Update accuracy
        if feedback.success {
            stats.accuracy = stats.accuracy * 0.99 + 0.01; // Smooth update
        } else {
            stats.accuracy = stats.accuracy * 0.99;
        }
    }

    fn name(&self) -> &str {
        "weighted"
    }

    fn stats(&self) -> SchedulerStats {
        self.stats.read().clone()
    }
}

/// Type alias for shared scheduler
pub type SharedScheduler = Arc<dyn Scheduler>;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_round_robin_scheduler() {
        let scheduler = RoundRobinScheduler::new(vec!["node1".to_string(), "node2".to_string()]);

        let task1 = Task::new("t1", vec![]);
        let task2 = Task::new("t2", vec![]);
        let task3 = Task::new("t3", vec![]);

        let d1 = scheduler.schedule(&task1).await.unwrap();
        let d2 = scheduler.schedule(&task2).await.unwrap();
        let d3 = scheduler.schedule(&task3).await.unwrap();

        // Should alternate between nodes
        assert_eq!(d1.assignment.node_id, "node1");
        assert_eq!(d2.assignment.node_id, "node2");
        assert_eq!(d3.assignment.node_id, "node1");
    }

    #[tokio::test]
    async fn test_weighted_scheduler() {
        let scheduler = WeightedScheduler::new();
        scheduler.register_node("node1".to_string(), 8, 16384);
        scheduler.register_node("node2".to_string(), 8, 16384);

        // Make node1 busy
        scheduler.update_node("node1", 0.8, 0.7, 5);
        scheduler.update_node("node2", 0.2, 0.3, 1);

        let task = Task::new("t1", vec![]);
        let decision = scheduler.schedule(&task).await.unwrap();

        // Should prefer node2 (less loaded)
        assert_eq!(decision.assignment.node_id, "node2");
    }
}
