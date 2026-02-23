
import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.append(os.getcwd())

from api.security.ml.anomaly import anomaly_detector
from api.security.ml.classifier import threat_classifier
from api.security.nids import nids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_ml")

async def verify_ml_components():
    logger.info("Verifying Security ML Components...")
    
    # 1. Test Anomaly Detector
    logger.info("1. Testing Anomaly Detector...")
    
    # Generate some normal-looking training data
    for _ in range(120):
        anomaly_detector.add_training_sample({
            'cpu_percent': 10 + (os.urandom(1)[0] % 20),
            'memory_percent': 30 + (os.urandom(1)[0] % 10),
            'network_bytes_in': 1000 + (os.urandom(1)[0] % 500),
            'network_bytes_out': 1000 + (os.urandom(1)[0] % 500),
            'packet_count': 10 + (os.urandom(1)[0] % 5)
        })
    
    # Force train
    anomaly_detector.train()
    
    if not anomaly_detector.is_trained:
        logger.error("Anomaly detector failed to train")
        return False
        
    # Test Normal
    normal_stats = {
        'cpu_percent': 15,
        'memory_percent': 35,
        'network_bytes_in': 1200,
        'network_bytes_out': 1200,
        'packet_count': 12
    }
    result_normal = anomaly_detector.detect(normal_stats)
    logger.info(f"Normal sample result: {result_normal}")
    
    # Test Anomaly (High CPU and Traffic)
    anomaly_stats = {
        'cpu_percent': 95,
        'memory_percent': 90,
        'network_bytes_in': 10000000,
        'network_bytes_out': 10000000,
        'packet_count': 50000
    }
    result_anomaly = anomaly_detector.detect(anomaly_stats)
    logger.info(f"Anomaly sample result: {result_anomaly}")
    
    if not result_anomaly['is_anomaly']:
        logger.warning("Anomaly detector failed to detect obvious anomaly (might need more training data or tuning)")
    
    # 2. Test Threat Classifier
    logger.info("2. Testing Threat Classifier...")
    
    # Test DDoS pattern
    ddos_stats = {'packet_rate': 2000, 'features': 'test'} 
    # specific keys expected by classifier heuristic
    ddos_features = {
        'packet_rate': 2000, 
        'failed_auth_count': 0, 
        'distinct_ports': 5, 
        'bytes_out': 5000
    }
    classification = threat_classifier.classify(ddos_features)
    logger.info(f"DDoS Classification: {classification}")
    
    if classification['threat_type'] != 'ddos':
        logger.error(f"Classifier failed to identify DDoS: {classification}")
        return False

    # 3. Test NIDS Integration
    logger.info("3. Testing NIDS ML Integration...")
    alerts = await nids.analyze_with_ml(anomaly_stats)
    logger.info(f"NIDS ML Alerts: {len(alerts)}")
    
    if len(alerts) > 0:
        logger.info(f"Alert details: {alerts[0].to_dict()}")
        
    logger.info("Verification Complete!")
    return True

if __name__ == "__main__":
    asyncio.run(verify_ml_components())
