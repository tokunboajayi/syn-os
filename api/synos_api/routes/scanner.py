from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from typing import List
import uuid
from datetime import datetime
import logging

from api.synos_api.schemas.security import ScanRequest, ScanResponse, ScanResultItem, VulnReport

try:
    import synos_kernel  # The Rust extension
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("synos_kernel not found. Using MOCK kernel.")
    from api.synos_api.core import mock_kernel as synos_kernel

try:
    from synos_ml.models import VulnerabilityPredictor, TargetInfo
except ImportError:
    class TargetInfo:
        def __init__(self, ip, open_ports): pass
        
    class VulnerabilityPredictor:
        def predict(self, info): return None
        
import pickle
import os

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/scanner", tags=["scanner"])

# In-memory store for now
scans_db = {}
vuln_predictor = None

@router.on_event("startup")
async def load_models():
    """Load ML models on startup."""
    global vuln_predictor
    try:
        # PWD is usually root of project in docker, or syn-os locally
        # We try a few paths
        paths = ["ml_models/vuln_predictor.pkl", "syn-os/ml_models/vuln_predictor.pkl"]
        model_path = next((p for p in paths if os.path.exists(p)), None)
        
        if model_path:
            with open(model_path, "rb") as f:
                vuln_predictor = pickle.load(f)
            logger.info(f"Loaded VulnerabilityPredictor from {model_path}")
        else:
            logger.warning("VulnerabilityPredictor model not found, utilizing heuristic mode")
            vuln_predictor = VulnerabilityPredictor() # Default heuristic
            
    except Exception as e:
        logger.error(f"Failed to load VulnerabilityPredictor: {e}")
        vuln_predictor = VulnerabilityPredictor() # Fallback

@router.post("/scan", response_model=ScanResponse)
async def start_scan(request: ScanRequest):
    """
    Start a network scan using the high-performance Rust Kernel.
    """
    scan_id = str(uuid.uuid4())
    logger.info(f"Starting scan {scan_id} for target {request.target}")

    try:
        # Run the blocking Rust call in a threadpool to avoid blocking the event loop
        # Wrapped in Circuit Breaker to prevent cascading failures
        from api.synos_api.core.circuit_breaker import scanner_breaker, CircuitBreakerOpenException
        
        try:
            results = await run_in_threadpool(
                scanner_breaker.call,
                synos_kernel.scan_network,
                request.target,
                request.ports,
                request.timeout_ms,
                100 
            )
        except CircuitBreakerOpenException:
            logger.error("Scanner Circuit Breaker is OPEN. Rejecting request.")
            raise HTTPException(status_code=503, detail="Scanner service temporarily unavailable (Circuit Breaker Open)")
        
        # Convert results to Pydantic models and aggregate ports by IP
        scan_results = []
        ports_by_ip = {}
        
        for r in results:
            scan_results.append(ScanResultItem(
                ip=r.ip,
                port=r.port,
                is_open=r.is_open,
                latency_ms=r.latency_ms
            ))
            if r.is_open:
                if r.ip not in ports_by_ip:
                    ports_by_ip[r.ip] = []
                ports_by_ip[r.ip].append(r.port)
        
        # Predict vulnerabilities (Max risk across all IPs)
        vuln_report = None
        if vuln_predictor and ports_by_ip:
            try:
                max_risk = -1.0
                best_pred = None
                
                for ip, ports in ports_by_ip.items():
                    info = TargetInfo(ip=ip, open_ports=ports)
                    pred = vuln_predictor.predict(info)
                    if pred.risk_score > max_risk:
                        max_risk = pred.risk_score
                        best_pred = pred
                
                if best_pred:
                    vuln_report = VulnReport(
                        risk_score=best_pred.risk_score,
                        criticality=best_pred.criticality,
                        likely_cves=best_pred.likely_cves,
                        confidence=best_pred.confidence
                    )
            except Exception as e:
                logger.error(f"Vuln prediction failed: {e}")

        response = ScanResponse(
            scan_id=scan_id,
            target=request.target,
            status="completed",
            results=scan_results,
            vulnerability_report=vuln_report,
            timestamp=datetime.utcnow()
        )
        
        # Store result
        scans_db[scan_id] = response
        
        return response

    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Emit event regardless of success/fail (if we have a result)
        if 'scan_id' in locals() and scan_id in scans_db:
             from api.synos_api.core.socket import sio
             await sio.emit('security:scan_complete', scans_db[scan_id].dict())

@router.get("/scans", response_model=List[ScanResponse])
async def list_scans():
    """List all past scans."""
    return list(scans_db.values())

@router.get("/scans/{scan_id}", response_model=ScanResponse)
async def get_scan(scan_id: str):
    """Get details of a specific scan."""
    if scan_id not in scans_db:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scans_db[scan_id]
