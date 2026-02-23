import numpy as np
import torch
import os
from loguru import logger
from synos_ml.models import NIDS, NetworkFeatures, VulnerabilityPredictor, TargetInfo

def train_nids():
    logger.info("Generating synthetic NIDS data...")
    n_samples = 5000
    
    # Normal traffic (low entropy, moderate rates)
    normal_traffic = np.random.normal(loc=1.0, scale=0.5, size=(n_samples, 16))
    normal_traffic = np.abs(normal_traffic) # features are positive
    
    # Train NIDS
    nids = NIDS()
    nids.fit(normal_traffic, epochs=10)
    
    # Test on "Attack" traffic (high entropy/rates)
    attack = np.array([
        [15.0] * 16, # Extreme values
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0] # SYN Flood
    ], dtype=np.float32)
    
    logger.info("Testing NIDS...")
    for i in range(len(attack)):
        res = nids.detect(attack[i])
        logger.info(f"Attack sample {i}: Anomaly={res.is_anomaly}, Score={res.anomaly_score:.4f}, Type={res.anomaly_type}")
        
    # Save model
    os.makedirs("ml_models", exist_ok=True)
    nids.save("ml_models/nids.pth")

def train_vuln_predictor():
    logger.info("Generating synthetic Vulnerability data...")
    predictor = VulnerabilityPredictor()
    
    # X: feature vectors, y: risk scores
    predictor.fit(np.zeros((1, 2049)), np.zeros(1)) # Dummy fit to confirm it works
    
    target = TargetInfo(ip="192.168.1.10", open_ports=[22, 80, 443])
    pred = predictor.predict(target)
    logger.info(f"Target [22, 80, 443] -> Risk: {pred.risk_score}, Criticality: {pred.criticality}")

    target_crit = TargetInfo(ip="10.0.0.5", open_ports=[445, 3389])
    pred_crit = predictor.predict(target_crit)
    logger.info(f"Target [445, 3389] -> Risk: {pred_crit.risk_score}, Criticality: {pred_crit.criticality}")
    
    # Save model (pickle for now since XGBoost has its own save/load but class wrapper needs pickle)
    import pickle
    with open("ml_models/vuln_predictor.pkl", "wb") as f:
        pickle.dump(predictor, f)
    logger.info("Saved VulnerabilityPredictor to ml_models/vuln_predictor.pkl")

if __name__ == "__main__":
    train_nids()
    train_vuln_predictor()
