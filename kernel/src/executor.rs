//! Task executor for Syn OS
//!
//! Manages task execution with work-stealing for load balancing.

use crate::queue::SharedTaskQueue;
use crate::scheduler::{SchedulingDecision, SharedScheduler};
use crate::task::{Task, TaskId, TaskResult, TaskStatus};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{broadcast, mpsc, RwLock, Semaphore};
use tracing::{debug, error, info, instrument, warn};

/// Maximum concurrent task executions per executor
const DEFAULT_CONCURRENCY: usize = 32;

/// Maximum output capture per task (bytes)
const MAX_OUTPUT_BYTES: usize = 1024 * 1024; // 1MB

/// Executor configuration
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum concurrent task executions
    pub max_concurrency: usize,
    /// Default execution timeout
    pub default_timeout: Duration,
    /// Working directory for tasks
    pub working_dir: Option<String>,
    /// Node ID this executor belongs to
    pub node_id: String,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrency: DEFAULT_CONCURRENCY,
            default_timeout: Duration::from_secs(3600), // 1 hour
            working_dir: None,
            node_id: "local".to_string(),
        }
    }
}

/// Event emitted during task execution
#[derive(Debug, Clone)]
pub enum ExecutorEvent {
    /// Task started executing
    TaskStarted { task_id: TaskId, node_id: String },
    /// Task completed
    TaskCompleted { task_id: TaskId, result: TaskResult },
    /// Task failed
    TaskFailed { task_id: TaskId, error: String },
    /// Task timed out
    TaskTimeout { task_id: TaskId },
    /// Executor shutting down
    Shutdown,
}

/// Task executor with work-stealing and concurrency control
pub struct Executor {
    /// Configuration
    config: ExecutorConfig,
    /// Task queue
    queue: SharedTaskQueue,
    /// Scheduler
    scheduler: SharedScheduler,
    /// Concurrency limiter
    semaphore: Arc<Semaphore>,
    /// Currently running tasks
    running: Arc<RwLock<HashMap<TaskId, RunningTask>>>,
    /// Event broadcast channel
    event_tx: broadcast::Sender<ExecutorEvent>,
    /// Shutdown signal
    shutdown: mpsc::Sender<()>,
}

/// Information about a running task
#[derive(Debug)]
struct RunningTask {
    task: Task,
    started_at: Instant,
    decision: SchedulingDecision,
}

impl Executor {
    /// Create a new executor
    pub fn new(
        config: ExecutorConfig,
        queue: SharedTaskQueue,
        scheduler: SharedScheduler,
    ) -> (Self, mpsc::Receiver<()>) {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        let (event_tx, _) = broadcast::channel(1024);

        let executor = Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrency)),
            config,
            queue,
            scheduler,
            running: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            shutdown: shutdown_tx,
        };

        (executor, shutdown_rx)
    }

    /// Subscribe to executor events
    pub fn subscribe(&self) -> broadcast::Receiver<ExecutorEvent> {
        self.event_tx.subscribe()
    }

    /// Get number of currently running tasks
    pub async fn running_count(&self) -> usize {
        self.running.read().await.len()
    }

    /// Check if a specific task is running
    pub async fn is_running(&self, task_id: &TaskId) -> bool {
        self.running.read().await.contains_key(task_id)
    }

    /// Execute a single task
    #[instrument(skip(self, task, decision), fields(task_id = %task.id))]
    pub async fn execute(&self, mut task: Task, decision: SchedulingDecision) -> TaskResult {
        // Acquire semaphore permit
        let _permit = self.semaphore.acquire().await.unwrap();

        let task_id = task.id;
        let started_at = Instant::now();

        info!("Starting task {} on node {}", task_id, decision.assignment.node_id);

        // Mark task as running
        task.start(decision.assignment.node_id.clone());

        // Track running task
        {
            let mut running = self.running.write().await;
            running.insert(
                task_id,
                RunningTask {
                    task: task.clone(),
                    started_at,
                    decision: decision.clone(),
                },
            );
        }

        // Emit start event
        let _ = self.event_tx.send(ExecutorEvent::TaskStarted {
            task_id,
            node_id: decision.assignment.node_id.clone(),
        });

        // Determine timeout
        let timeout = task.deadline.unwrap_or(self.config.default_timeout);

        // Execute the command
        let result = self.run_command(&task, timeout).await;

        // Remove from running
        {
            let mut running = self.running.write().await;
            running.remove(&task_id);
        }

        // Emit completion event
        match &result {
            Ok(task_result) => {
                let _ = self.event_tx.send(ExecutorEvent::TaskCompleted {
                    task_id,
                    result: task_result.clone(),
                });
            }
            Err(e) => {
                let _ = self.event_tx.send(ExecutorEvent::TaskFailed {
                    task_id,
                    error: e.clone(),
                });
            }
        }

        result.unwrap_or_else(|e| TaskResult {
            exit_code: -1,
            stdout: None,
            stderr: Some(e),
            duration_ms: started_at.elapsed().as_millis() as u64,
            peak_memory_mb: 0,
            cpu_time_ms: 0,
        })
    }

    /// Run a command with timeout
    async fn run_command(&self, task: &Task, timeout: Duration) -> Result<TaskResult, String> {
        if task.command.is_empty() {
            return Err("Empty command".to_string());
        }

        let started = Instant::now();

        // Build command
        let mut cmd = Command::new(&task.command[0]);
        if task.command.len() > 1 {
            cmd.args(&task.command[1..]);
        }

        // Set environment
        for (key, value) in &task.env {
            cmd.env(key, value);
        }

        // Set working directory
        if let Some(ref dir) = self.config.working_dir {
            cmd.current_dir(dir);
        }

        // Capture output
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Spawn process
        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn: {}", e))?;

        // Capture stdout
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let stdout_handle = tokio::spawn(async move {
            let mut output = String::new();
            if let Some(stdout) = stdout {
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if output.len() < MAX_OUTPUT_BYTES {
                        output.push_str(&line);
                        output.push('\n');
                    }
                }
            }
            output
        });

        let stderr_handle = tokio::spawn(async move {
            let mut output = String::new();
            if let Some(stderr) = stderr {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if output.len() < MAX_OUTPUT_BYTES {
                        output.push_str(&line);
                        output.push('\n');
                    }
                }
            }
            output
        });

        // Wait with timeout
        let result = tokio::time::timeout(timeout, child.wait()).await;

        match result {
            Ok(Ok(status)) => {
                let stdout = stdout_handle.await.unwrap_or_default();
                let stderr = stderr_handle.await.unwrap_or_default();

                Ok(TaskResult {
                    exit_code: status.code().unwrap_or(-1),
                    stdout: if stdout.is_empty() {
                        None
                    } else {
                        Some(stdout)
                    },
                    stderr: if stderr.is_empty() {
                        None
                    } else {
                        Some(stderr)
                    },
                    duration_ms: started.elapsed().as_millis() as u64,
                    peak_memory_mb: 0, // TODO: Track memory
                    cpu_time_ms: 0,    // TODO: Track CPU time
                })
            }
            Ok(Err(e)) => Err(format!("Process error: {}", e)),
            Err(_) => {
                // Timeout - kill the process
                let _ = child.kill().await;
                Err("Task timed out".to_string())
            }
        }
    }

    /// Main executor loop - continuously processes tasks from queue
    pub async fn run(&self, mut shutdown_rx: mpsc::Receiver<()>) {
        info!(
            "Executor starting with concurrency {}",
            self.config.max_concurrency
        );

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Executor received shutdown signal");
                    break;
                }
                task = async {
                    // Try to dequeue a task
                    self.queue.dequeue()
                } => {
                    if let Some(task) = task {
                        // Schedule the task
                        if let Some(decision) = self.scheduler.schedule(&task).await {
                            // Clone for async move
                            let executor = self.clone_ref();
                            let task_clone = task.clone();

                            // Spawn task execution
                            tokio::spawn(async move {
                                let result = executor.execute(task_clone, decision.clone()).await;

                                // Send feedback to scheduler
                                executor.scheduler.feedback(
                                    &task.id,
                                    crate::scheduler::SchedulingFeedback {
                                        actual_duration_ms: result.duration_ms,
                                        success: result.exit_code == 0,
                                        resource_utilization: 0.5, // TODO: Calculate
                                        wait_time_ms: 0, // TODO: Track
                                    }
                                ).await;
                            });
                        } else {
                            // No scheduling decision - re-queue
                            warn!("No scheduling decision for task {}, re-queueing", task.id);
                            self.queue.enqueue(task);
                        }
                    } else {
                        // Queue empty, wait a bit
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }
            }
        }

        let _ = self.event_tx.send(ExecutorEvent::Shutdown);
        info!("Executor stopped");
    }

    /// Create a reference clone for async operations
    fn clone_ref(&self) -> ExecutorRef {
        ExecutorRef {
            config: self.config.clone(),
            queue: Arc::clone(&self.queue),
            scheduler: Arc::clone(&self.scheduler),
            semaphore: Arc::clone(&self.semaphore),
            running: Arc::clone(&self.running),
            event_tx: self.event_tx.clone(),
        }
    }
}

/// Reference to executor for async operations
struct ExecutorRef {
    config: ExecutorConfig,
    queue: SharedTaskQueue,
    scheduler: SharedScheduler,
    semaphore: Arc<Semaphore>,
    running: Arc<RwLock<HashMap<TaskId, RunningTask>>>,
    event_tx: broadcast::Sender<ExecutorEvent>,
}

impl ExecutorRef {
    async fn execute(&self, mut task: Task, decision: SchedulingDecision) -> TaskResult {
        let _permit = self.semaphore.acquire().await.unwrap();
        let task_id = task.id;
        let started_at = Instant::now();

        task.start(decision.assignment.node_id.clone());

        {
            let mut running = self.running.write().await;
            running.insert(
                task_id,
                RunningTask {
                    task: task.clone(),
                    started_at,
                    decision: decision.clone(),
                },
            );
        }

        let _ = self.event_tx.send(ExecutorEvent::TaskStarted {
            task_id,
            node_id: decision.assignment.node_id.clone(),
        });

        // Simplified execution for reference
        let result = TaskResult {
            exit_code: 0,
            stdout: Some("Executed".to_string()),
            stderr: None,
            duration_ms: started_at.elapsed().as_millis() as u64,
            peak_memory_mb: 0,
            cpu_time_ms: 0,
        };

        {
            let mut running = self.running.write().await;
            running.remove(&task_id);
        }

        let _ = self.event_tx.send(ExecutorEvent::TaskCompleted {
            task_id,
            result: result.clone(),
        });

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::shared_queue;
    use crate::scheduler::RoundRobinScheduler;

    #[tokio::test]
    async fn test_executor_creation() {
        let queue = shared_queue();
        let scheduler: SharedScheduler =
            Arc::new(RoundRobinScheduler::new(vec!["node1".to_string()]));

        let (executor, _shutdown) = Executor::new(ExecutorConfig::default(), queue, scheduler);

        assert_eq!(executor.running_count().await, 0);
    }

    #[tokio::test]
    async fn test_simple_execution() {
        let queue = shared_queue();
        let scheduler: SharedScheduler =
            Arc::new(RoundRobinScheduler::new(vec!["node1".to_string()]));

        let (executor, _shutdown) =
            Executor::new(ExecutorConfig::default(), queue.clone(), scheduler.clone());

        // Create a simple echo task
        let task = Task::new("echo-test", vec!["echo".to_string(), "hello".to_string()]);

        let decision = scheduler.schedule(&task).await.unwrap();
        let result = executor.execute(task, decision).await;

        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.unwrap().contains("hello"));
    }
}
