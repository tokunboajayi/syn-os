from pydantic import BaseModel, Field, IPvAnyAddress
from typing import List, Optional, Any
from datetime import datetime

class ScanRequest(BaseModel):
    target: str = Field(..., description="Target IP or CIDR range", example="192.168.1.0/24")
    ports: List[int] = Field(default=[80, 443, 22, 8080], description="List of ports to scan")
    timeout_ms: int = Field(default=1000, ge=10, le=10000)
    concurrency: int = Field(default=100, ge=1, le=1000)

class ScanResultItem(BaseModel):
    ip: str
    port: int
    is_open: bool
    latency_ms: Optional[int]

class VulnReport(BaseModel):
    risk_score: float
    criticality: str
    likely_cves: List[str]
    confidence: float

class ScanResponse(BaseModel):
    scan_id: str
    target: str
    status: str
    results: List[ScanResultItem]
    vulnerability_report: Optional[VulnReport] = None
    timestamp: datetime

class ThreatAlert(BaseModel):
    id: str
    severity: str  # critical, high, medium, low
    source_ip: str
    description: str
    timestamp: datetime
    status: str  # active, resolved

class SecurityStatus(BaseModel):
    score: float
    status: str
    threat_level: str
    active_scans: int
