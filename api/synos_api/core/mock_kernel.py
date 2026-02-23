import random
import time
from typing import List

class MockScanResult:
    def __init__(self, ip: str, port: int, is_open: bool, latency_ms: int):
        self.ip = ip
        self.port = port
        self.is_open = is_open
        self.latency_ms = latency_ms

def scan_network(target: str, ports: List[int], timeout_ms: int, concurrency: int) -> List[MockScanResult]:
    """Mock implementation of synos_kernel.scan_network"""
    time.sleep(0.1) # Simulate work
    results = []
    
    # Handle CIDR or single IP
    ips = [target] if '/' not in target else [target.split('/')[0]] # Simplified
    
    for ip in ips:
        for port in ports:
            is_open = random.choice([True, False])
            latency = random.randint(10, 100)
            results.append(MockScanResult(ip, port, is_open, latency))
            
    return results
