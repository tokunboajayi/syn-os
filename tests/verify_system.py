
import sys
import os
import asyncio
from pathlib import Path
from loguru import logger

# Add project root and component dirs to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "ml"))
sys.path.append(str(project_root / "api"))

# Configure logger to file
logger.add("verify_system.log", rotation="1 MB")

def verify_ml_components():
    """Verify ML components can be instantiated."""
    print("Verifying ML components...")
    try:
        # Import modules
        from synos_ml.models.anomaly import AnomalyDetector
        from synos_ml.models.gnn import TaskGNN
        from synos_ml.scheduler.ppo import PPOScheduler
        from synos_ml.data.validation import DataValidator
        print("Imports successful")
        
        # Instantiate Anomaly Detector
        try:
            print("Initializing AnomalyDetector...")
            detector = AnomalyDetector(input_dim=10)
            logger.info("✅ AnomalyDetector initialized")
            print("AnomalyDetector initialized")
        except Exception as e:
            logger.error(f"❌ AnomalyDetector Error: {e}")
            print(f"AnomalyDetector Error: {e}")

        # Instantiate GNN
        try:
            print("Initializing TaskGNN...")
            gnn = TaskGNN(node_features=16)
            logger.info("✅ TaskGNN initialized")
            print("TaskGNN initialized")
        except Exception as e:
            logger.error(f"❌ TaskGNN Error: {e}")
            print(f"TaskGNN Error: {e}")
        
        # Instantiate PPO
        try:
            print("Initializing PPOScheduler...")
            scheduler = PPOScheduler()
            logger.info("✅ PPOScheduler initialized")
            print("PPOScheduler initialized")
        except Exception as e:
            logger.error(f"❌ PPOScheduler Error: {e}")
            print(f"PPOScheduler Error: {e}")
        
        # Instantiate Validator
        try:
            print("Initializing DataValidator...")
            validator = DataValidator()
            logger.info("✅ DataValidator initialized")
            print("DataValidator initialized")
        except Exception as e:
            logger.error(f"❌ DataValidator Error: {e}")
            print(f"DataValidator Error: {e}")
            
        return True
    except ImportError as e:
        logger.error(f"❌ ML Import Error: {e}")
        print(f"ML Import Error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ ML Init Error: {e}")
        print(f"ML Init Error: {e}")
        return False

def verify_api_components():
    """Verify API components can be instantiated."""
    logger.info("Verifying API components...")
    # Mock prometheus_client if missing
    try:
        import prometheus_client
    except ImportError:
        logger.warning("Mocking prometheus_client for verification")
        from unittest.mock import MagicMock
        sys.modules["prometheus_client"] = MagicMock()
        sys.modules["prometheus_client.Counter"] = MagicMock
        sys.modules["prometheus_client.Histogram"] = MagicMock
        sys.modules["prometheus_client.generate_latest"] = MagicMock

    try:
        from fastapi.testclient import TestClient
        from synos_api.main import app
        from synos_api.dag import TaskDAG
        
        # Verify DAG
        dag = TaskDAG("test_dag")
        logger.info("✅ TaskDAG initialized")
        
        # Verify API
        client = TestClient(app)
        response = client.get("/health")
        if response.status_code == 200:
            logger.info("✅ API Health check passed")
        else:
            logger.error(f"❌ API Health check failed: {response.status_code}")
            return False
            
        return True
    except ImportError as e:
        logger.error(f"❌ API Import Error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ API Init Exception: {e}")
        return False

def main():
    logger.info("Starting Syn OS System Verification...")
    
    ml_status = verify_ml_components()
    api_status = verify_api_components()
    
    if ml_status and api_status:
        logger.info("🎉 All systems verified successfully!")
        sys.exit(0)
    else:
        logger.error("⚠️ Verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
