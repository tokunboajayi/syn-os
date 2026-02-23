//! Lock-free priority task queue for Syn OS
//!
//! This module provides a high-performance, lock-free priority queue
//! implementation using crossbeam for concurrent operations.

use crate::task::{Priority, Task, TaskId, TaskStatus};
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, instrument, trace};

const PRIORITY_LEVELS: usize = 10;

/// Lock-free priority task queue with O(1) enqueue/dequeue
///
/// Uses separate queues for each priority level (0-9) to achieve
/// priority-based scheduling without lock contention.
pub struct TaskQueue {
    /// Separate queues for each priority level
    queues: [SegQueue<TaskId>; PRIORITY_LEVELS],
    /// Task storage for O(1) lookup
    tasks: DashMap<TaskId, Task>,
    /// Count of tasks at each priority level
    counts: [AtomicUsize; PRIORITY_LEVELS],
    /// Total task count
    total: AtomicUsize,
    /// Total tasks ever enqueued (for metrics)
    total_enqueued: AtomicU64,
    /// Total tasks ever dequeued (for metrics)
    total_dequeued: AtomicU64,
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskQueue {
    /// Create a new empty task queue
    pub fn new() -> Self {
        Self {
            queues: Default::default(),
            tasks: DashMap::new(),
            counts: Default::default(),
            total: AtomicUsize::new(0),
            total_enqueued: AtomicU64::new(0),
            total_dequeued: AtomicU64::new(0),
        }
    }

    /// Enqueue a task - O(1) lock-free operation
    ///
    /// # Arguments
    /// * `task` - The task to enqueue
    ///
    /// # Returns
    /// The task ID of the enqueued task
    #[instrument(skip(self, task), fields(task_id = %task.id, priority = task.priority.0))]
    pub fn enqueue(&self, task: Task) -> TaskId {
        let priority = task.priority.0 as usize;
        let id = task.id;

        trace!("Enqueueing task {} at priority {}", id, priority);

        // Store the task
        self.tasks.insert(id, task);

        // Add to priority queue
        self.queues[priority].push(id);

        // Update counts
        self.counts[priority].fetch_add(1, Ordering::Relaxed);
        self.total.fetch_add(1, Ordering::Relaxed);
        self.total_enqueued.fetch_add(1, Ordering::Relaxed);

        debug!("Task {} enqueued, queue depth: {}", id, self.len());
        id
    }

    /// Dequeue highest priority task - O(1) amortized
    ///
    /// Returns the highest priority task available, or None if the queue is empty.
    #[instrument(skip(self))]
    pub fn dequeue(&self) -> Option<Task> {
        // Try each priority level from highest (0) to lowest (9)
        for (priority, queue) in self.queues.iter().enumerate() {
            if let Some(id) = queue.pop() {
                self.counts[priority].fetch_sub(1, Ordering::Relaxed);
                self.total.fetch_sub(1, Ordering::Relaxed);
                self.total_dequeued.fetch_add(1, Ordering::Relaxed);

                if let Some((_, task)) = self.tasks.remove(&id) {
                    trace!("Dequeued task {} from priority {}", id, priority);
                    return Some(task);
                }
            }
        }
        None
    }

    /// Dequeue a batch of tasks for efficient processing
    ///
    /// # Arguments
    /// * `max_count` - Maximum number of tasks to dequeue
    ///
    /// # Returns
    /// Vector of dequeued tasks (may be less than max_count)
    pub fn dequeue_batch(&self, max_count: usize) -> Vec<Task> {
        let mut tasks = Vec::with_capacity(max_count);

        for _ in 0..max_count {
            if let Some(task) = self.dequeue() {
                tasks.push(task);
            } else {
                break;
            }
        }

        tasks
    }

    /// Get task by ID without removing
    ///
    /// # Arguments
    /// * `id` - The task ID to look up
    ///
    /// # Returns
    /// A cloned copy of the task if found
    pub fn get(&self, id: &TaskId) -> Option<Task> {
        self.tasks.get(id).map(|r| r.clone())
    }

    /// Update a task in place
    ///
    /// # Arguments
    /// * `id` - The task ID to update
    /// * `f` - A function that modifies the task
    ///
    /// # Returns
    /// true if the task was found and updated, false otherwise
    pub fn update<F>(&self, id: &TaskId, f: F) -> bool
    where
        F: FnOnce(&mut Task),
    {
        if let Some(mut task) = self.tasks.get_mut(id) {
            f(&mut task);
            true
        } else {
            false
        }
    }

    /// Update task status
    pub fn update_status(&self, id: &TaskId, status: TaskStatus) -> bool {
        self.update(id, |task| {
            task.status = status;
        })
    }

    /// Remove a task from the queue
    ///
    /// Note: This removes from storage but the ID may still be in a priority queue.
    /// The dequeue operation handles missing tasks gracefully.
    pub fn remove(&self, id: &TaskId) -> Option<Task> {
        self.tasks.remove(id).map(|(_, task)| task)
    }

    /// Check if a task exists
    pub fn contains(&self, id: &TaskId) -> bool {
        self.tasks.contains_key(id)
    }

    /// Current queue depth (total across all priorities)
    pub fn len(&self) -> usize {
        self.total.load(Ordering::Relaxed)
    }

    /// Queue depth by priority level
    pub fn len_by_priority(&self, priority: u8) -> usize {
        if priority as usize >= PRIORITY_LEVELS {
            return 0;
        }
        self.counts[priority as usize].load(Ordering::Relaxed)
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get total tasks ever enqueued
    pub fn total_enqueued(&self) -> u64 {
        self.total_enqueued.load(Ordering::Relaxed)
    }

    /// Get total tasks ever dequeued
    pub fn total_dequeued(&self) -> u64 {
        self.total_dequeued.load(Ordering::Relaxed)
    }

    /// Get queue statistics
    pub fn stats(&self) -> QueueStats {
        QueueStats {
            total_depth: self.len(),
            by_priority: std::array::from_fn(|i| self.counts[i].load(Ordering::Relaxed)),
            total_enqueued: self.total_enqueued(),
            total_dequeued: self.total_dequeued(),
        }
    }

    /// Get all tasks with a specific status
    pub fn tasks_by_status(&self, status: TaskStatus) -> Vec<Task> {
        self.tasks
            .iter()
            .filter(|r| r.status == status)
            .map(|r| r.clone())
            .collect()
    }

    /// Get tasks that are ready to execute (dependencies satisfied)
    pub fn ready_tasks(&self, completed_task_ids: &[TaskId]) -> Vec<Task> {
        self.tasks
            .iter()
            .filter(|r| r.status == TaskStatus::Queued)
            .filter(|r| r.dependencies_satisfied(completed_task_ids))
            .map(|r| r.clone())
            .collect()
    }

    /// Clear all tasks from the queue
    pub fn clear(&self) {
        self.tasks.clear();
        for queue in &self.queues {
            while queue.pop().is_some() {}
        }
        for count in &self.counts {
            count.store(0, Ordering::Relaxed);
        }
        self.total.store(0, Ordering::Relaxed);
    }
}

/// Queue statistics snapshot
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Total queue depth
    pub total_depth: usize,
    /// Depth by priority level
    pub by_priority: [usize; PRIORITY_LEVELS],
    /// Total tasks ever enqueued
    pub total_enqueued: u64,
    /// Total tasks ever dequeued
    pub total_dequeued: u64,
}

impl QueueStats {
    /// Calculate throughput (requires time delta)
    pub fn throughput(&self, elapsed_secs: f64) -> f64 {
        if elapsed_secs > 0.0 {
            self.total_dequeued as f64 / elapsed_secs
        } else {
            0.0
        }
    }
}

/// Thread-safe wrapper for shared queue access
pub type SharedTaskQueue = Arc<TaskQueue>;

/// Create a new shared task queue
pub fn shared_queue() -> SharedTaskQueue {
    Arc::new(TaskQueue::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_dequeue() {
        let queue = TaskQueue::new();

        let task1 = Task::new("task1", vec!["echo".to_string()]);
        let task2 = Task::new("task2", vec!["ls".to_string()]);

        queue.enqueue(task1);
        queue.enqueue(task2);

        assert_eq!(queue.len(), 2);

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(queue.len(), 1);
        assert!(dequeued.name == "task1" || dequeued.name == "task2");
    }

    #[test]
    fn test_priority_ordering() {
        let queue = TaskQueue::new();

        // Enqueue low priority first
        let low = Task::new("low", vec![]).with_priority(Priority::LOW);
        let high = Task::new("high", vec![]).with_priority(Priority::HIGH);
        let critical = Task::new("critical", vec![]).with_priority(Priority::CRITICAL);

        queue.enqueue(low);
        queue.enqueue(high);
        queue.enqueue(critical);

        // Should dequeue in priority order (critical first)
        assert_eq!(queue.dequeue().unwrap().name, "critical");
        assert_eq!(queue.dequeue().unwrap().name, "high");
        assert_eq!(queue.dequeue().unwrap().name, "low");
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let queue = Arc::new(TaskQueue::new());
        let mut handles = vec![];

        // Spawn multiple producer threads
        for i in 0..10 {
            let q = Arc::clone(&queue);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let task = Task::new(format!("task-{}-{}", i, j), vec![]);
                    q.enqueue(task);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(queue.len(), 1000);
        assert_eq!(queue.total_enqueued(), 1000);
    }

    #[test]
    fn test_batch_dequeue() {
        let queue = TaskQueue::new();

        for i in 0..100 {
            queue.enqueue(Task::new(format!("task-{}", i), vec![]));
        }

        let batch = queue.dequeue_batch(30);
        assert_eq!(batch.len(), 30);
        assert_eq!(queue.len(), 70);
    }

    #[test]
    fn test_queue_stats() {
        let queue = TaskQueue::new();

        queue.enqueue(Task::new("t1", vec![]).with_priority(Priority::HIGH));
        queue.enqueue(Task::new("t2", vec![]).with_priority(Priority::HIGH));
        queue.enqueue(Task::new("t3", vec![]).with_priority(Priority::LOW));

        let stats = queue.stats();
        assert_eq!(stats.total_depth, 3);
        assert_eq!(stats.by_priority[Priority::HIGH.0 as usize], 2);
        assert_eq!(stats.by_priority[Priority::LOW.0 as usize], 1);
    }
}
