import asyncio
import aiohttp
import time
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WAR_GAMES")

API_URL = "http://localhost:8000"
CONCURRENT_USERS = 50
TOTAL_REQUESTS = 200

async def simulate_user(session, user_id):
    for i in range(5):
        try:
            # 1. Health Check (Fast) - Should hit rate limit eventually
            async with session.get(f"{API_URL}/health") as resp:
                status = resp.status
                if status == 429:
                    logger.warning(f"User {user_id}: Rate Limited (429)")
                elif status == 200:
                    pass # Normal
                else:
                    logger.error(f"User {user_id}: Unexpected status {status}")

            # 2. Trigger Scan (Heavy) - Should trigger Circuit Breaker if we flood it
            # We'll use a fake target to fail fast or timeout
            payload = {
                "target": f"192.168.1.{random.randint(1, 255)}",
                "ports": [80, 443],
                "timeout_ms": 100
            }
            async with session.post(f"{API_URL}/api/v1/scanner/scan", json=payload) as resp:
                if resp.status == 503:
                    logger.critical(f"User {user_id}: CIRCUIT BREAKER OPEN (503)")
                elif resp.status == 200:
                    logger.info(f"User {user_id}: Scan successful")
            
            await asyncio.sleep(random.random() * 2) # Random delay
            
        except Exception as e:
            logger.error(f"User {user_id}: Connection failed - {e}")

async def wait_for_api(session):
    """Wait for API to be healthy."""
    logger.info("Waiting for API to be ready...")
    for i in range(30):
        try:
            async with session.get(f"{API_URL}/health") as resp:
                if resp.status == 200:
                    logger.info("API is READY!")
                    return True
        except:
            pass
        await asyncio.sleep(1)
        if i % 5 == 0:
            logger.info(f"Still waiting... ({i}s)")
    logger.error("API failed to become ready.")
    return False

async def run_war_games():
    logger.info("INITIATING WAR GAMES SIMULATION...")
    logger.info(f"Targets: {API_URL}")
    logger.info(f"Agents: {CONCURRENT_USERS}")
    
    async with aiohttp.ClientSession() as session:
        if not await wait_for_api(session):
            return

        start_time = time.time()
        tasks = [simulate_user(session, i) for i in range(CONCURRENT_USERS)]
        await asyncio.gather(*tasks)
        
    duration = time.time() - start_time
    logger.info(f"SIMULATION COMPLETE in {duration:.2f}s")

if __name__ == "__main__":
    asyncio.run(run_war_games())
