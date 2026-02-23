from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from datetime import datetime

class LocationPoint(BaseModel):
    device_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: Optional[float] = None
    accuracy: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "gps"

class FloorPlanCreate(BaseModel):
    name: str
    image_url: str
    top_left: Tuple[float, float] # lat, lon
    bottom_right: Tuple[float, float] # lat, lon
    level: int = 0

class FloorPlanResponse(FloorPlanCreate):
    id: str
    created_at: datetime

class TrajectoryPrediction(BaseModel):
    device_id: str
    future_points: List[Tuple[float, float]] # lat, lon
    confidence: float
    timestamp: datetime
