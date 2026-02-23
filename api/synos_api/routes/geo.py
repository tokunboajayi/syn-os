from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import uuid
from datetime import datetime
import logging

from api.synos_api.schemas.geo import LocationPoint, FloorPlanCreate, FloorPlanResponse, TrajectoryPrediction
try:
    from synos_ml.models import TrajectoryPredictor
except ImportError:
    class TrajectoryPredictor:
        def __init__(self, path=None): pass
        def predict(self, data): return None

import os
import pickle

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/geo", tags=["geo"])

# In-memory store placeholder (replace with PostGIS)
locations_db = []
floor_plans_db = {}
trajectory_predictor = None

@router.on_event("startup")
async def load_models():
    """Load ML models on startup."""
    global trajectory_predictor
    try:
        paths = ["ml_models/trajectory_predictor.pkl", "syn-os/ml_models/trajectory_predictor.pkl"]
        model_path = next((p for p in paths if os.path.exists(p)), None)
        
        if model_path:
            trajectory_predictor = TrajectoryPredictor(model_path)
            logger.info(f"Loaded TrajectoryPredictor from {model_path}")
        else:
            logger.warning("TrajectoryPredictor model not found, using untrained instance")
            trajectory_predictor = TrajectoryPredictor() 
            
    except Exception as e:
        logger.error(f"Failed to load TrajectoryPredictor: {e}")
        trajectory_predictor = TrajectoryPredictor()

@router.post("/location")
async def ingest_location(point: LocationPoint):
    """Ingest a single location point."""
    locations_db.append(point)
    
    # Emit real-time update
    from api.synos_api.core.socket import sio
    await sio.emit('geo:location_update', point.dict())
    
    # In real implementation:
    # 1. Save to PostGIS
    # 2. Update real-time map (WebSocket)
    # 3. Feed to TrajectoryPredictor
    return {"status": "received", "id": str(uuid.uuid4())}

@router.post("/floorplan", response_model=FloorPlanResponse)
async def create_floor_plan(plan: FloorPlanCreate):
    """Upload a floor plan for Indoor Mode."""
    plan_id = str(uuid.uuid4())
    response = FloorPlanResponse(
        id=plan_id,
        created_at=datetime.utcnow(),
        **plan.dict()
    )
    floor_plans_db[plan_id] = response
    return response

@router.get("/floorplans", response_model=List[FloorPlanResponse])
async def list_floor_plans():
    """List all floor plans."""
    return list(floor_plans_db.values())

@router.get("/predict/{device_id}", response_model=TrajectoryPrediction)
async def predict_path(device_id: str):
    """Predict future path using AI."""
    # Placeholder for LSTM prediction
    return TrajectoryPrediction(
        device_id=device_id,
        future_points=[(0.0, 0.0)],
        confidence=0.0,
        timestamp=datetime.utcnow()
    )
