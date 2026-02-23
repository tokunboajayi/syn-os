//! Syn OS Kernel - Core Module
//!
//! This is the main entry point for the Syn OS kernel library.
//! It provides a high-performance, ML-optimized task execution environment.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           Event Loop                     │
//! │  ┌─────────┐ ┌──────────┐ ┌──────────┐  │
//! │  │ Queue   │→│ Scheduler│→│ Executor │  │
//! │  └─────────┘ └──────────┘ └──────────┘  │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use synos_kernel::{EventLoop, Task, Priority};
//!
//! #[tokio::main]
//! async fn main() {
//!     let event_loop = EventLoop::new(Default::default());
//!     
//!     // Submit a task
//!     event_loop.submit(
//!         Task::new("my-task", vec!["echo".to_string(), "hello".to_string()])
//!             .with_priority(Priority::HIGH)
//!     );
//!     
//!     // Run the event loop
//!     event_loop.run().await.unwrap();
//! }
//! ```

// Re-export main types
pub mod event_loop;
pub mod executor;
pub mod queue;
pub mod scheduler;
pub mod scanner;
pub mod python;
pub mod task;

// Convenience re-exports
pub use event_loop::{EventLoop, EventLoopBuilder, EventLoopConfig};
pub use executor::{Executor, ExecutorConfig, ExecutorEvent};
pub use queue::{shared_queue, QueueStats, SharedTaskQueue, TaskQueue};
pub use scheduler::{
    ResourceAssignment, RoundRobinScheduler, Scheduler, SchedulerStats, SchedulingDecision,
    SchedulingFeedback, SharedScheduler, WeightedScheduler,
};
pub use scanner::{NetworkScanner, ScanConfig, ScanResult};
pub use task::{
    DependencyType, Priority, ResourceRequirements, RetryPolicy, Task, TaskDependency, TaskId,
    TaskResult, TaskStatus,
};

/// Kernel version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Kernel name
pub const NAME: &str = "Syn OS Kernel";

/// Get kernel info
pub fn info() -> KernelInfo {
    KernelInfo {
        name: NAME.to_string(),
        version: VERSION.to_string(),
    }
}

/// Kernel information
#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub name: String,
    pub version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let info = info();
        assert_eq!(info.name, "Syn OS Kernel");
        assert!(!info.version.is_empty());
    }

    #[test]
    fn test_public_exports() {
        // Verify all public types are accessible
        let _task = Task::new("test", vec![]);
        let _priority = Priority::NORMAL;
        let _status = TaskStatus::Queued;
        let _queue = TaskQueue::new();
    }
}
