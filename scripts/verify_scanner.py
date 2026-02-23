import sys
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier")

async def test_scanner():
    try:
        import synos_kernel
        logger.info(f"Successfully imported synos_kernel: {synos_kernel}")
    except ImportError as e:
        logger.error(f"Failed to import synos_kernel: {e}")
        return

    target = "127.0.0.1" # Localhost
    ports = [80, 443, 8000, 22] # Common ports, 8000 might be our API if running
    timeout = 500
    concurrency = 10

    logger.info(f"Starting scan on {target} ports {ports}")
    
    # scan_network is blocking, mimicking how API calls it (but API uses threadpool)
    # We can call it directly here since this is a script
    try:
        results = synos_kernel.scan_network(target, ports, timeout, concurrency)
        logger.info(f"Scan complete. Found {len(results)} results.")
        for r in results:
            status = "OPEN" if r.is_open else "CLOSED"
            logger.info(f"Port {r.port}: {status} ({r.latency_ms}ms)")
            
    except Exception as e:
        logger.error(f"Scan failed: {e}")

if __name__ == "__main__":
    # synos_kernel.scan_network uses tokio runtime internally, so we don't need asyncio.run for it specifically
    # but let's look at the signature again.
    # fn scan_network(...) -> PyResult<Vec<PyScanResult>>
    # It creates its own runtime. So we can call it synchronously.
    
    # We used 'async def' in the API route only to use `run_in_threadpool`.
    # Here we can just call it.
    
    import synos_kernel
    results = synos_kernel.scan_network("127.0.0.1", [80, 443, 8000, 135], 500, 10)
    print("Direct call results:", results)
    for r in results:
        print(f"IP: {r.ip}, Port: {r.port}, Open: {r.is_open}")
