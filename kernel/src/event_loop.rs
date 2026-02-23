//! Main event loop for Syn OS kernel
//!
//! Coordinates task queue, scheduler, and executor in an async event-driven architecture.

use crate::executor::{Executor, ExecutorConfig, ExecutorEvent};
use crate::queue::{shared_queue, SharedTaskQueue};
use crate::scheduler::{RoundRobinScheduler, SharedScheduler, WeightedScheduler};
use crate::task::Task;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{error, info, warn};

/// Event loop configuration
#[derive(Debug, Clone)]
pub struct EventLoopConfig {
    /// Executor configuration
    pub executor: ExecutorConfig,
    /// Batch size for dequeuing tasks
    pub batch_size: usize,
    /// Poll interval when queue is empty
    pub poll_interval_ms: u64,
    /// Enable metrics collection
    pub metrics_enabled: bool,
}

impl Default for EventLoopConfig {
    fn default() -> Self {
        Self {
            executor: ExecutorConfig::default(),
            batch_size: 16,
            poll_interval_ms: 10,
            metrics_enabled: true,
        }
    }
}

/// Main event loop state
pub struct EventLoop {
    /// Configuration
    config: EventLoopConfig,
    /// Task queue
    queue: SharedTaskQueue,
    /// Scheduler
    scheduler: SharedScheduler,
    /// Running state
    running: Arc<AtomicBool>,
    /// Start time for uptime tracking
    start_time: Instant,
    /// Metrics
    metrics: Arc<EventLoopMetrics>,
}

/// Event loop metrics
#[derive(Debug, Default)]
pub struct EventLoopMetrics {
    /// Total tasks submitted
    pub tasks_submitted: AtomicU64,
    /// Total tasks completed
    pub tasks_completed: AtomicU64,
    /// Total tasks failed
    pub tasks_failed: AtomicU64,
    /// Total loop iterations
    pub loop_iterations: AtomicU64,
}

impl EventLoopMetrics {
    /// Get a snapshot of metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            tasks_submitted: self.tasks_submitted.load(Ordering::Relaxed),
            tasks_completed: self.tasks_completed.load(Ordering::Relaxed),
            tasks_failed: self.tasks_failed.load(Ordering::Relaxed),
            loop_iterations: self.loop_iterations.load(Ordering::Relaxed),
        }
    }
}

/// Metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub loop_iterations: u64,
}

impl EventLoop {
    /// Create a new event loop with default scheduler
    pub fn new(config: EventLoopConfig) -> Self {
        let scheduler: SharedScheduler =
            Arc::new(RoundRobinScheduler::new(vec!["local".to_string()]));
        Self::with_scheduler(config, scheduler)
    }

    /// Create a new event loop with custom scheduler
    pub fn with_scheduler(config: EventLoopConfig, scheduler: SharedScheduler) -> Self {
        Self {
            config,
            queue: shared_queue(),
            scheduler,
            running: Arc::new(AtomicBool::new(false)),
            start_time: Instant::now(),
            metrics: Arc::new(EventLoopMetrics::default()),
        }
    }

    /// Create with weighted scheduler
    pub fn with_weighted_scheduler(config: EventLoopConfig) -> Self {
        let scheduler = WeightedScheduler::new();
        scheduler.register_node("local".to_string(), 8, 16384);
        Self::with_scheduler(config, Arc::new(scheduler))
    }

    /// Get the task queue
    pub fn queue(&self) -> SharedTaskQueue {
        Arc::clone(&self.queue)
    }

    /// Get the scheduler
    pub fn scheduler(&self) -> SharedScheduler {
        Arc::clone(&self.scheduler)
    }

    /// Submit a task
    pub fn submit(&self, task: Task) {
        self.queue.enqueue(task);
        self.metrics.tasks_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Submit multiple tasks
    pub fn submit_batch(&self, tasks: Vec<Task>) {
        for task in tasks {
            self.submit(task);
        }
    }

    /// Check if event loop is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get metrics
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Get queue depth
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Run the event loop
    pub async fn run(&self) -> Result<(), String> {
        if self
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err("Event loop already running".to_string());
        }

        info!("Syn OS Event Loop starting...");

        // Create executor
        let (executor, shutdown_rx) = Executor::new(
            self.config.executor.clone(),
            Arc::clone(&self.queue),
            Arc::clone(&self.scheduler),
        );

        // Subscribe to executor events
        let mut event_rx = executor.subscribe();
        let metrics = Arc::clone(&self.metrics);

        // Spawn event handler
        let event_handler = tokio::spawn(async move {
            while let Ok(event) = event_rx.recv().await {
                match event {
                    ExecutorEvent::TaskCompleted { .. } => {
                        metrics.tasks_completed.fetch_add(1, Ordering::Relaxed);
                    }
                    ExecutorEvent::TaskFailed { .. } | ExecutorEvent::TaskTimeout { .. } => {
                        metrics.tasks_failed.fetch_add(1, Ordering::Relaxed);
                    }
                    ExecutorEvent::Shutdown => {
                        break;
                    }
                    _ => {}
                }
            }
        });

        // Run executor
        executor.run(shutdown_rx).await;

        // Wait for event handler
        let _ = event_handler.await;

        self.running.store(false, Ordering::SeqCst);
        info!("Syn OS Event Loop stopped");

        Ok(())
    }

    /// Stop the event loop gracefully
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

/// Builder for EventLoop
pub struct EventLoopBuilder {
    config: EventLoopConfig,
    scheduler: Option<SharedScheduler>,
}

impl EventLoopBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: EventLoopConfig::default(),
            scheduler: None,
        }
    }

    /// Set executor concurrency
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.config.executor.max_concurrency = concurrency;
        self
    }

    /// Set node ID
    pub fn node_id(mut self, node_id: impl Into<String>) -> Self {
        self.config.executor.node_id = node_id.into();
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set poll interval
    pub fn poll_interval(mut self, ms: u64) -> Self {
        self.config.poll_interval_ms = ms;
        self
    }

    /// Set custom scheduler
    pub fn scheduler(mut self, scheduler: SharedScheduler) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Build the event loop
    pub fn build(self) -> EventLoop {
        match self.scheduler {
            Some(scheduler) => EventLoop::with_scheduler(self.config, scheduler),
            None => EventLoop::new(self.config),
        }
    }
}

impl Default for EventLoopBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::Priority;

    #[tokio::test]
    async fn test_event_loop_creation() {
        let event_loop = EventLoop::new(EventLoopConfig::default());
        assert!(!event_loop.is_running());
        assert_eq!(event_loop.queue_depth(), 0);
    }

    #[tokio::test]
    async fn test_task_submission() {
        let event_loop = EventLoop::new(EventLoopConfig::default());

        event_loop.submit(Task::new("task1", vec!["echo".to_string()]));
        event_loop.submit(Task::new("task2", vec!["ls".to_string()]));

        assert_eq!(event_loop.queue_depth(), 2);
        assert_eq!(event_loop.metrics().tasks_submitted, 2);
    }

    #[test]
    fn test_builder() {
        let event_loop = EventLoopBuilder::new()
            .concurrency(16)
            .node_id("my-node")
            .batch_size(32)
            .poll_interval(5)
            .build();

        assert_eq!(event_loop.config.executor.max_concurrency, 16);
        assert_eq!(event_loop.config.executor.node_id, "my-node");
        assert_eq!(event_loop.config.batch_size, 32);
    }
}
