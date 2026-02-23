import numpy as np
import logging
import os
from synos_ml.models import TrajectoryPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_paths(num_paths=100, seq_len=50):
    """Generate synthetic random walk paths."""
    paths = []
    for _ in range(num_paths):
        # Start at a random "home" location
        lat = 40.7128 + np.random.normal(0, 0.01)
        lon = -74.0060 + np.random.normal(0, 0.01)
        
        path = []
        for _ in range(seq_len):
            path.append([lat, lon])
            # Random walk with momentum
            lat += np.random.normal(0, 0.0001)
            lon += np.random.normal(0, 0.0001)
        paths.append(np.array(path))
    return paths

def train_trajectory():
    logger.info("Generating synthetic Trajectory data...")
    paths = generate_synthetic_paths()
    
    predictor = TrajectoryPredictor()
    predictor.fit(paths, epochs=20)
    
    # Test prediction
    test_path = paths[0][-10:] # Last 10 points
    pred_lat, pred_lon = predictor.predict(test_path.tolist())
    
    actual_lat, actual_lon = test_path[-1] # The last point we gave it
    logger.info(f"Last Input: {actual_lat:.6f}, {actual_lon:.6f}")
    logger.info(f"Predicted Next: {pred_lat:.6f}, {pred_lon:.6f}")
    
    os.makedirs("ml_models", exist_ok=True)
    predictor.save("ml_models/trajectory_predictor.pkl")

if __name__ == "__main__":
    train_trajectory()
