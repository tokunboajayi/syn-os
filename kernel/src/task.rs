//! Task definitions and structures for Syn OS
//!
//! This module defines the core Task type and related structures
//! used throughout the kernel.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// Unique task identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub Uuid);

impl TaskId {
    /// Create a new unique task ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from existing UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Task execution status with detailed states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Task is in the queue waiting to be scheduled
    Queued,
    /// Task has been assigned to a resource
    Scheduled,
    /// Task is currently executing
    Running,
    /// Task completed successfully
    Completed,
    /// Task execution failed
    Failed,
    /// Task exceeded its deadline
    Timeout,
    /// Task was cancelled by user
    Cancelled,
    /// Task failed but will be retried
    RetryPending,
}

impl TaskStatus {
    /// Check if task is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Timeout | TaskStatus::Cancelled
        )
    }

    /// Check if task is active (running or scheduled)
    pub fn is_active(&self) -> bool {
        matches!(self, TaskStatus::Scheduled | TaskStatus::Running)
    }
}

/// Task priority levels (0 = highest, 9 = lowest)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Priority(pub u8);

impl Priority {
    pub const CRITICAL: Self = Self(0);
    pub const HIGH: Self = Self(2);
    pub const NORMAL: Self = Self(5);
    pub const LOW: Self = Self(7);
    pub const BACKGROUND: Self = Self(9);

    /// Create a new priority, clamped to valid range
    pub fn new(value: u8) -> Self {
        Self(value.min(9))
    }

    /// Get the priority value
    pub fn value(&self) -> u8 {
        self.0
    }
}

impl Default for Priority {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// Resource requirements for a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Number of CPU cores required
    pub cpu_cores: u8,
    /// Memory required in megabytes
    pub memory_mb: u64,
    /// GPU memory required in megabytes (optional)
    pub gpu_memory_mb: Option<u64>,
    /// Disk I/O throughput required in MB/s (optional)
    pub disk_io_mbps: Option<u32>,
    /// Network bandwidth required in Mbps (optional)
    pub network_bandwidth_mbps: Option<u32>,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            memory_mb: 256,
            gpu_memory_mb: None,
            disk_io_mbps: None,
            network_bandwidth_mbps: None,
        }
    }
}

impl ResourceRequirements {
    /// Create a new resource requirement
    pub fn new(cpu_cores: u8, memory_mb: u64) -> Self {
        Self {
            cpu_cores,
            memory_mb,
            ..Default::default()
        }
    }

    /// Builder method to add GPU memory requirement
    pub fn with_gpu(mut self, gpu_memory_mb: u64) -> Self {
        self.gpu_memory_mb = Some(gpu_memory_mb);
        self
    }

    /// Calculate a resource "weight" for scheduling decisions
    pub fn weight(&self) -> f64 {
        let cpu_weight = self.cpu_cores as f64 * 10.0;
        let mem_weight = (self.memory_mb as f64 / 1024.0) * 5.0;
        let gpu_weight = self.gpu_memory_mb.map_or(0.0, |g| (g as f64 / 1024.0) * 20.0);
        cpu_weight + mem_weight + gpu_weight
    }
}

/// Task dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDependency {
    /// ID of the task this depends on
    pub task_id: TaskId,
    /// Type of dependency
    pub dependency_type: DependencyType,
}

/// Types of task dependencies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DependencyType {
    /// Must complete successfully before this task starts
    Hard,
    /// Preferred but not required
    Soft,
    /// Data output from this task is needed
    Data,
}

/// Retry policy for failed tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_retries: u8,
    /// Initial backoff duration in milliseconds
    pub backoff_ms: u64,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f32,
    /// Current retry count
    #[serde(default)]
    pub current_retries: u8,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_ms: 1000,
            backoff_multiplier: 2.0,
            current_retries: 0,
        }
    }
}

impl RetryPolicy {
    /// Calculate the next backoff duration
    pub fn next_backoff(&self) -> Duration {
        let backoff = self.backoff_ms as f64
            * self.backoff_multiplier.powi(self.current_retries as i32) as f64;
        Duration::from_millis(backoff as u64)
    }

    /// Check if more retries are available
    pub fn can_retry(&self) -> bool {
        self.current_retries < self.max_retries
    }
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Exit code (0 = success)
    pub exit_code: i32,
    /// Standard output (truncated)
    pub stdout: Option<String>,
    /// Standard error (truncated)
    pub stderr: Option<String>,
    /// Actual execution duration
    pub duration_ms: u64,
    /// Peak memory usage in MB
    pub peak_memory_mb: u64,
    /// CPU time used in milliseconds
    pub cpu_time_ms: u64,
}

/// Core task structure with all metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: TaskId,
    /// Human-readable task name
    pub name: String,
    /// Task priority
    pub priority: Priority,
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Optional deadline (duration from creation)
    pub deadline: Option<Duration>,
    /// Command to execute
    pub command: Vec<String>,
    /// Environment variables
    pub env: HashMap<String, String>,
    /// Task dependencies
    pub dependencies: Vec<TaskDependency>,
    /// Retry policy
    pub retry_policy: RetryPolicy,
    /// Task creation timestamp
    pub created_at: DateTime<Utc>,
    /// Current task status
    pub status: TaskStatus,
    /// Assigned node ID (if scheduled)
    pub assigned_node: Option<String>,

    // === ML-generated predictions ===
    /// Predicted execution duration in milliseconds
    pub predicted_duration_ms: Option<u64>,
    /// Predicted peak memory usage in MB
    pub predicted_memory_peak_mb: Option<u64>,
    /// Neural network computed priority score
    pub priority_score: Option<f32>,

    // === Execution tracking ===
    /// When the task started executing
    pub started_at: Option<DateTime<Utc>>,
    /// When the task completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Execution result
    pub result: Option<TaskResult>,
    /// Error message if failed
    pub error: Option<String>,
}

impl Task {
    /// Create a new task with default values
    pub fn new(name: impl Into<String>, command: Vec<String>) -> Self {
        Self {
            id: TaskId::new(),
            name: name.into(),
            priority: Priority::default(),
            resources: ResourceRequirements::default(),
            deadline: None,
            command,
            env: HashMap::new(),
            dependencies: Vec::new(),
            retry_policy: RetryPolicy::default(),
            created_at: Utc::now(),
            status: TaskStatus::Queued,
            assigned_node: None,
            predicted_duration_ms: None,
            predicted_memory_peak_mb: None,
            priority_score: None,
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
        }
    }

    /// Builder method to set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Builder method to set resources
    pub fn with_resources(mut self, resources: ResourceRequirements) -> Self {
        self.resources = resources;
        self
    }

    /// Builder method to set deadline
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add an environment variable
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Add a dependency
    pub fn with_dependency(mut self, task_id: TaskId, dep_type: DependencyType) -> Self {
        self.dependencies.push(TaskDependency {
            task_id,
            dependency_type: dep_type,
        });
        self
    }

    /// Check if all dependencies are satisfied
    pub fn dependencies_satisfied(&self, completed_tasks: &[TaskId]) -> bool {
        self.dependencies.iter().all(|dep| {
            if dep.dependency_type == DependencyType::Soft {
                true // Soft dependencies are always "satisfied"
            } else {
                completed_tasks.contains(&dep.task_id)
            }
        })
    }

    /// Calculate time since creation
    pub fn age(&self) -> Duration {
        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.created_at);
        Duration::from_millis(elapsed.num_milliseconds().max(0) as u64)
    }

    /// Check if task has exceeded deadline
    pub fn is_overdue(&self) -> bool {
        if let Some(deadline) = self.deadline {
            self.age() > deadline
        } else {
            false
        }
    }

    /// Mark task as started
    pub fn start(&mut self, node_id: String) {
        self.status = TaskStatus::Running;
        self.started_at = Some(Utc::now());
        self.assigned_node = Some(node_id);
    }

    /// Mark task as completed
    pub fn complete(&mut self, result: TaskResult) {
        self.status = TaskStatus::Completed;
        self.completed_at = Some(Utc::now());
        self.result = Some(result);
    }

    /// Mark task as failed
    pub fn fail(&mut self, error: String) {
        if self.retry_policy.can_retry() {
            self.status = TaskStatus::RetryPending;
            self.retry_policy.current_retries += 1;
        } else {
            self.status = TaskStatus::Failed;
        }
        self.error = Some(error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_creation() {
        let task = Task::new("test-task", vec!["echo".to_string(), "hello".to_string()]);
        assert_eq!(task.name, "test-task");
        assert_eq!(task.status, TaskStatus::Queued);
        assert_eq!(task.priority, Priority::NORMAL);
    }

    #[test]
    fn test_task_builder() {
        let task = Task::new("test", vec!["ls".to_string()])
            .with_priority(Priority::HIGH)
            .with_resources(ResourceRequirements::new(4, 2048))
            .with_env("PATH", "/usr/bin");

        assert_eq!(task.priority, Priority::HIGH);
        assert_eq!(task.resources.cpu_cores, 4);
        assert_eq!(task.resources.memory_mb, 2048);
        assert_eq!(task.env.get("PATH"), Some(&"/usr/bin".to_string()));
    }

    #[test]
    fn test_retry_policy() {
        let mut policy = RetryPolicy::default();
        assert!(policy.can_retry());

        policy.current_retries = 3;
        assert!(!policy.can_retry());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::CRITICAL < Priority::HIGH);
        assert!(Priority::HIGH < Priority::NORMAL);
        assert!(Priority::NORMAL < Priority::LOW);
    }
}
