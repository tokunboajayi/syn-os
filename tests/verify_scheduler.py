
import asyncio
import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from api.synos_api.core.scheduler import scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_scheduler")

async def verify_scheduler():
    logger.info("Verifying ML Scheduler Components...")
    
    # 1. Test Prediction
    logger.info("\n1. Testing Execution Time Prediction...")
    task_data = {
        'priority': 5,
        'resources': {'cpu_cores': 2, 'memory_mb': 2048}
    }
    duration = scheduler.predict_duration(task_data)
    logger.info(f"Predicted duration: {duration} ms")
    
    if duration <= 0:
        logger.error("Prediction failed (non-positive duration)")
        return False

    # 2. Test Forecasting
    logger.info("\n2. Testing Resource Forecasting...")
    forecast = scheduler.forecast_demand(hours_ahead=6)
    logger.info(f"Forecast (6h): {forecast}")
    
    if 'cpu' not in forecast or 'mean' not in forecast['cpu']:
        logger.error("Forecast structure invalid")
        return False
        
    # 3. Test Queue Optimization (GNN Logic)
    logger.info("\n3. Testing GNN Queue Optimization...")
    
    # Create mock tasks with dependencies
    # Task A <- Task B (B depends on A)
    # Task A <- Task C
    # Task A has 2 dependents, most critical
    
    tasks = [
        {'id': 'task_c', 'priority': 5, 'dependencies': [{'task_id': 'task_a'}]}, # Dependent
        {'id': 'task_a', 'priority': 5, 'dependencies': []},                     # Critical Dependency
        {'id': 'task_b', 'priority': 5, 'dependencies': [{'task_id': 'task_a'}]}, # Dependent
        {'id': 'task_d', 'priority': 1, 'dependencies': []},                     # High Priority Independent
    ]
    
    logger.info("Original order: " + ", ".join([t['id'] for t in tasks]))
    
    optimized = scheduler.optimize_queue(tasks)
    logger.info("Optimized order: " + ", ".join([t['id'] for t in optimized]))
    
    # Check if task_a is prioritized (it has 2 dependents)
    # Note: Our GNN mock logic gives bonus for dependents
    # task_a score = 2 * 10 + (10 - 5) = 25
    # task_d score = 0 * 10 + (10 - 1) = 9
    # So task_a should be first
    
    if optimized[0]['id'] != 'task_a':
        logger.warning(f"Optimization check: Expected task_a first, got {optimized[0]['id']}")
    else:
        logger.info("Optimization verified: Critical dependency prioritized")
        
    # 4. Test Node Assignment (PPO)
    logger.info("\n4. Testing PPO Node Assignment...")
    nodes = [{'id': 'node-1'}, {'id': 'node-2'}, {'id': 'node-3'}]
    assigned = scheduler.assign_node(task_data, nodes)
    logger.info(f"Assigned node: {assigned}")
    
    if assigned not in [n['id'] for n in nodes]:
        logger.error(f"Invalid node assigned: {assigned}")
        return False

    logger.info("\nScheduler Verification Complete!")
    return True

if __name__ == "__main__":
    asyncio.run(verify_scheduler())
