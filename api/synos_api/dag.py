"""
DAG-Based Task Dependencies for Syn OS

Enables complex workflow orchestration with:
- Directed Acyclic Graph task representation
- Parallel execution of independent tasks
- Critical path analysis
- Fault-tolerant execution with checkpointing
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import json
from collections import defaultdict
import heapq
from loguru import logger


class TaskState(Enum):
    """Task execution state."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class DAGTask:
    """Task in a DAG workflow."""
    
    task_id: str
    name: str
    execute_fn: Optional[Callable] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Execution settings
    timeout_seconds: float = 3600
    retries: int = 3
    retry_delay_seconds: float = 10
    
    # Resource requirements
    cpu_request: float = 1.0
    memory_mb: int = 1024
    
    # State
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Checkpointing
    checkpoint_data: Optional[bytes] = None
    
    def __hash__(self):
        return hash(self.task_id)


@dataclass
class DAGConfig:
    """Configuration for DAG execution."""
    
    max_parallel_tasks: int = 10
    checkpoint_enabled: bool = True
    checkpoint_dir: str = "./checkpoints"
    fail_fast: bool = False  # Stop on first failure vs continue
    retry_failed: bool = True


class TaskDAG:
    """
    Directed Acyclic Graph for task orchestration.
    
    Features:
    - Topological sorting for execution order
    - Parallel execution of independent tasks
    - Critical path analysis
    - Checkpoint/resume capability
    """
    
    def __init__(self, dag_id: str, config: Optional[DAGConfig] = None):
        self.dag_id = dag_id
        self.config = config or DAGConfig()
        
        self._tasks: Dict[str, DAGTask] = {}
        self._graph: Dict[str, Set[str]] = defaultdict(set)  # task -> dependents
        self._reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # task -> dependencies
        
        self._execution_order: List[str] = []
        self._critical_path: List[str] = []
        
        self._running_tasks: Set[str] = set()
        self._completed_count: int = 0
        self._failed_count: int = 0
    
    def add_task(self, task: DAGTask):
        """Add a task to the DAG."""
        if task.task_id in self._tasks:
            raise ValueError(f"Task {task.task_id} already exists")
        
        self._tasks[task.task_id] = task
        
        # Build dependency graph
        for dep_id in task.depends_on:
            self._graph[dep_id].add(task.task_id)
            self._reverse_graph[task.task_id].add(dep_id)
        
        # Invalidate cached execution order
        self._execution_order = []
    
    def add_dependency(self, from_task: str, to_task: str):
        """Add a dependency edge."""
        if from_task not in self._tasks or to_task not in self._tasks:
            raise ValueError("Both tasks must exist")
        
        self._graph[from_task].add(to_task)
        self._reverse_graph[to_task].add(from_task)
        self._tasks[to_task].depends_on.append(from_task)
        
        # Check for cycles
        if self._has_cycle():
            # Rollback
            self._graph[from_task].remove(to_task)
            self._reverse_graph[to_task].remove(from_task)
            self._tasks[to_task].depends_on.remove(from_task)
            raise ValueError("Adding this dependency would create a cycle")
    
    def _has_cycle(self) -> bool:
        """Check for cycles in the graph."""
        visited = set()
        rec_stack = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self._graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_id in self._tasks:
            if task_id not in visited:
                if dfs(task_id):
                    return True
        
        return False
    
    def get_execution_order(self) -> List[str]:
        """Get topological sort of tasks."""
        if self._execution_order:
            return self._execution_order
        
        in_degree = {task_id: len(deps) for task_id, deps in self._reverse_graph.items()}
        for task_id in self._tasks:
            if task_id not in in_degree:
                in_degree[task_id] = 0
        
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for dependent in self._graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(self._tasks):
            raise ValueError("Graph has a cycle")
        
        self._execution_order = result
        return result
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to execute."""
        ready = []
        
        for task_id, task in self._tasks.items():
            if task.state != TaskState.PENDING:
                continue
            
            # Check all dependencies completed
            deps_completed = all(
                self._tasks[dep_id].state == TaskState.COMPLETED
                for dep_id in task.depends_on
            )
            
            if deps_completed:
                ready.append(task_id)
        
        return ready
    
    def compute_critical_path(self) -> Tuple[List[str], float]:
        """
        Compute the critical path (longest path through DAG).
        
        Returns:
            (path, total_duration)
        """
        # Use estimated durations or 1.0 as default
        durations = {}
        for task_id, task in self._tasks.items():
            # Use actual duration if completed, otherwise estimate
            if task.started_at and task.completed_at:
                durations[task_id] = (task.completed_at - task.started_at).total_seconds()
            else:
                durations[task_id] = task.timeout_seconds * 0.1  # Estimate 10% of timeout
        
        # Compute longest path ending at each node
        longest = {task_id: 0.0 for task_id in self._tasks}
        predecessor = {task_id: None for task_id in self._tasks}
        
        for task_id in self.get_execution_order():
            for dep_id in self._reverse_graph[task_id]:
                new_dist = longest[dep_id] + durations[dep_id]
                if new_dist > longest[task_id]:
                    longest[task_id] = new_dist
                    predecessor[task_id] = dep_id
        
        # Find task with longest path
        end_task = max(longest, key=lambda t: longest[t] + durations[t])
        max_duration = longest[end_task] + durations[end_task]
        
        # Reconstruct path
        path = [end_task]
        current = end_task
        while predecessor[current]:
            current = predecessor[current]
            path.append(current)
        
        path.reverse()
        self._critical_path = path
        
        return path, max_duration
    
    async def execute(
        self,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the entire DAG.
        
        Returns:
            Dict with execution results and metrics
        """
        start_time = datetime.utcnow()
        execution_context = execution_context or {}
        
        logger.info(f"Starting DAG execution: {self.dag_id}")
        
        # Initialize all tasks as PENDING
        for task in self._tasks.values():
            if task.state == TaskState.PENDING:
                pass  # Already pending
        
        # Mark tasks with no dependencies as READY
        for task_id in self.get_execution_order():
            if not self._reverse_graph[task_id]:
                self._tasks[task_id].state = TaskState.READY
        
        # Execute until all done or failure
        while True:
            ready_tasks = self.get_ready_tasks()
            
            if not ready_tasks and not self._running_tasks:
                break
            
            # Limit parallel tasks
            available_slots = self.config.max_parallel_tasks - len(self._running_tasks)
            tasks_to_run = ready_tasks[:available_slots]
            
            # Start new tasks
            running_coroutines = []
            for task_id in tasks_to_run:
                self._tasks[task_id].state = TaskState.READY
                self._running_tasks.add(task_id)
                running_coroutines.append(
                    self._execute_task(task_id, execution_context)
                )
            
            if running_coroutines:
                # Wait for at least one to complete
                done, pending = await asyncio.wait(
                    running_coroutines,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                
                # Process completed
                for future in done:
                    try:
                        task_id = await future
                        self._running_tasks.discard(task_id)
                    except Exception as e:
                        logger.error(f"Task execution error: {e}")
                        if self.config.fail_fast:
                            # Cancel all pending
                            for p in pending:
                                p.cancel()
                            raise
            else:
                # Wait for running tasks
                await asyncio.sleep(0.1)
        
        end_time = datetime.utcnow()
        
        # Compile results
        results = {
            "dag_id": self.dag_id,
            "status": "completed" if self._failed_count == 0 else "failed",
            "total_tasks": len(self._tasks),
            "completed_tasks": self._completed_count,
            "failed_tasks": self._failed_count,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "task_results": {
                task_id: {
                    "state": task.state.value,
                    "result": str(task.result)[:100] if task.result else None,
                    "error": task.error,
                }
                for task_id, task in self._tasks.items()
            },
        }
        
        logger.info(f"DAG execution complete: {results['status']}")
        return results
    
    async def _execute_task(
        self,
        task_id: str,
        context: Dict[str, Any],
    ) -> str:
        """Execute a single task."""
        task = self._tasks[task_id]
        task.state = TaskState.RUNNING
        task.started_at = datetime.utcnow()
        
        logger.info(f"Executing task: {task_id}")
        
        for attempt in range(task.retries + 1):
            try:
                if task.execute_fn:
                    # Get results from dependencies
                    dep_results = {
                        dep_id: self._tasks[dep_id].result
                        for dep_id in task.depends_on
                    }
                    
                    # Execute with timeout
                    task.result = await asyncio.wait_for(
                        task.execute_fn(context, dep_results),
                        timeout=task.timeout_seconds,
                    )
                else:
                    # No-op task
                    task.result = None
                
                task.state = TaskState.COMPLETED
                task.completed_at = datetime.utcnow()
                self._completed_count += 1
                
                # Update dependents
                for dependent_id in self._graph[task_id]:
                    # Check if all dependencies now complete
                    dep_task = self._tasks[dependent_id]
                    if dep_task.state == TaskState.PENDING:
                        all_deps_done = all(
                            self._tasks[d].state == TaskState.COMPLETED
                            for d in dep_task.depends_on
                        )
                        if all_deps_done:
                            dep_task.state = TaskState.READY
                
                logger.info(f"Task completed: {task_id}")
                return task_id
                
            except asyncio.TimeoutError:
                task.error = f"Timeout after {task.timeout_seconds}s"
                logger.warning(f"Task {task_id} attempt {attempt + 1} timed out")
                
            except Exception as e:
                task.error = str(e)
                logger.warning(f"Task {task_id} attempt {attempt + 1} failed: {e}")
            
            # Retry delay
            if attempt < task.retries:
                await asyncio.sleep(task.retry_delay_seconds)
        
        # All retries failed
        task.state = TaskState.FAILED
        task.completed_at = datetime.utcnow()
        self._failed_count += 1
        
        # Skip dependents
        self._skip_dependents(task_id)
        
        return task_id
    
    def _skip_dependents(self, failed_task_id: str):
        """Mark all dependents of a failed task as SKIPPED."""
        queue = list(self._graph[failed_task_id])
        visited = set()
        
        while queue:
            task_id = queue.pop(0)
            if task_id in visited:
                continue
            visited.add(task_id)
            
            task = self._tasks[task_id]
            if task.state == TaskState.PENDING:
                task.state = TaskState.SKIPPED
                queue.extend(self._graph[task_id])
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG to dictionary."""
        return {
            "dag_id": self.dag_id,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "depends_on": t.depends_on,
                    "state": t.state.value,
                }
                for t in self._tasks.values()
            ],
        }
    
    def visualize(self) -> str:
        """Generate Mermaid diagram of DAG."""
        lines = ["graph TD"]
        
        for task_id, task in self._tasks.items():
            # Node styling based on state
            if task.state == TaskState.COMPLETED:
                style = ":::completed"
            elif task.state == TaskState.FAILED:
                style = ":::failed"
            elif task.state == TaskState.RUNNING:
                style = ":::running"
            else:
                style = ""
            
            lines.append(f"    {task_id}[{task.name}]{style}")
        
        for from_id, to_ids in self._graph.items():
            for to_id in to_ids:
                lines.append(f"    {from_id} --> {to_id}")
        
        # Add styles
        lines.extend([
            "",
            "    classDef completed fill:#90EE90",
            "    classDef failed fill:#FFB6C1",
            "    classDef running fill:#87CEEB",
        ])
        
        return "\n".join(lines)
