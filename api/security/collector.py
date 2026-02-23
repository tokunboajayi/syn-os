"""
Security Data Collector Module
Collects network traffic, system logs, and authentication events
for threat analysis and intrusion detection.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import psutil
import socket

from api.security.nids import nids
from api.security.ml.anomaly import anomaly_detector

logger = logging.getLogger(__name__)


class SecurityDataCollector:
    """Collects security-relevant data from various sources"""
    
    def __init__(self):
        self.is_collecting = False
        self.network_stats = {}
        self.auth_events = []
        self.system_logs = []
        self.tasks = []
        
    async def start_collection(self):
        """Start collecting security data"""
        self.is_collecting = True
        logger.info("Security data collection started")
        
        # Start parallel collection tasks in background
        self.tasks = [
            asyncio.create_task(self.collect_network_traffic()),
            asyncio.create_task(self.collect_system_metrics()),
            asyncio.create_task(self.collect_auth_events()),
            asyncio.create_task(self.analyze_system_state())
        ]
    
    async def stop_collection(self):
        """Stop collecting security data"""
        self.is_collecting = False
        for task in self.tasks:
            task.cancel()
        self.tasks = []
        logger.info("Security data collection stopped")

    async def analyze_system_state(self):
        """
        Periodically analyze the aggregate system state using ML models.
        Combines network and system metrics for holistic anomaly detection.
        """
        while self.is_collecting:
            try:
                # Gather current metrics
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                
                # Get latest network stats (or defaults if not yet collected)
                net_sent = self.network_stats.get('bytes_sent', 0)
                net_recv = self.network_stats.get('bytes_recv', 0)
                packets = self.network_stats.get('packets_recv', 0) + self.network_stats.get('packets_sent', 0)
                
                features = {
                    'cpu_percent': cpu,
                    'memory_percent': mem,
                    'network_bytes_in': net_recv,
                    'network_bytes_out': net_sent,
                    'packet_count': packets
                }
                
                # 1. Online Training (Continuous Learning)
                # Add current normal-looking state to training data
                # In production, you'd filter out known bad states first
                anomaly_detector.add_training_sample(features)
                
                # 2. Run ML Analysis via NIDS
                await nids.analyze_with_ml(features)
                
            except Exception as e:
                logger.error(f"Error in ML analysis loop: {e}")
                
            await asyncio.sleep(5)  # Analyze every 5 seconds

    
    async def collect_network_traffic(self):
        """Monitor network connections and traffic patterns"""
        while self.is_collecting:
            try:
                connections = psutil.net_connections(kind='inet')
                
                # Analyze connections for suspicious patterns
                suspicious_connections = []
                for conn in connections:
                    if conn.status == 'ESTABLISHED':
                        # Check for connections to unusual ports
                        if conn.raddr and conn.raddr.port in [4444, 5555, 6666, 31337]:
                            suspicious_connections.append({
                                'local_addr': f"{conn.laddr.ip}:{conn.laddr.port}",
                                'remote_addr': f"{conn.raddr.ip}:{conn.raddr.port}",
                                'status': conn.status,
                                'timestamp': datetime.utcnow().isoformat()
                            })
                
                if suspicious_connections:
                    logger.warning(f"Found {len(suspicious_connections)} suspicious connections")
                    
                # Store network statistics
                net_io = psutil.net_io_counters()
                self.network_stats = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errors_in': net_io.errin,
                    'errors_out': net_io.errout,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error collecting network traffic: {e}")
            
            await asyncio.sleep(5)  # Collect every 5 seconds
    
    async def collect_system_metrics(self):
        """Collect system-level security metrics"""
        while self.is_collecting:
            try:
                # CPU usage spikes can indicate malicious activity
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Memory usage
                memory = psutil.virtual_memory()
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                
                # Check for anomalies
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage detected: {cpu_percent}%")
                
                if memory.percent > 90:
                    logger.warning(f"High memory usage detected: {memory.percent}%")
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            await asyncio.sleep(10)  # Collect every 10 seconds
    
    async def collect_auth_events(self):
        """Monitor authentication-related events"""
        while self.is_collecting:
            try:
                # In a real implementation, this would parse auth logs
                # For now, we'll track basic process information
                
                processes = psutil.process_iter(['pid', 'name', 'username', 'create_time'])
                new_processes = []
                
                for proc in processes:
                    try:
                        info = proc.info
                        # Track processes created in the last 10 seconds
                        if datetime.utcnow().timestamp() - info['create_time'] < 10:
                            new_processes.append({
                                'pid': info['pid'],
                                'name': info['name'],
                                'user': info['username'],
                                'timestamp': datetime.fromtimestamp(info['create_time']).isoformat()
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if new_processes:
                    self.auth_events.extend(new_processes)
                    # Keep only last 1000 events
                    self.auth_events = self.auth_events[-1000:]
                
            except Exception as e:
                logger.error(f"Error collecting auth events: {e}")
            
            await asyncio.sleep(10)
    
    def get_latest_stats(self) -> Dict:
        """Get the latest collected statistics"""
        return {
            'network': self.network_stats,
            'auth_events_count': len(self.auth_events),
            'recent_auth_events': self.auth_events[-10:] if self.auth_events else []
        }


# Global collector instance
collector = SecurityDataCollector()
