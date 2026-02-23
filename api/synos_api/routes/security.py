"""
Security API Routes
Endpoints for security monitoring, alerts, and threat analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict
from datetime import datetime
import logging

from api.security.collector import collector
from api.security.nids import nids, ThreatSeverity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/security", tags=["security"])


@router.get("/alerts")
async def get_security_alerts(limit: int = 100):
    """Get recent security alerts"""
    try:
        alerts = nids.get_recent_alerts(limit=limit)
        return {
            "success": True,
            "count": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/unresolved")
async def get_unresolved_alerts():
    """Get unresolved security alerts"""
    try:
        alerts = nids.get_unresolved_alerts()
        return {
            "success": True,
            "count": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        logger.error(f"Error fetching unresolved alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Mark a security alert as resolved"""
    try:
        success = nids.resolve_alert(alert_id)
        if success:
            return {"success": True, "message": f"Alert {alert_id} resolved"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/score")
async def get_security_score():
    """Get overall security health score (0-100)"""
    try:
        score = nids.get_security_score()
        
        # Determine status based on score
        if score >= 90:
            status = "excellent"
        elif score >= 70:
            status = "good"
        elif score >= 50:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "success": True,
            "score": round(score, 2),
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating security score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_security_stats():
    """Get security statistics and metrics"""
    try:
        stats = collector.get_latest_stats()
        alerts = nids.get_recent_alerts(limit=10)
        score = nids.get_security_score()
        
        return {
            "success": True,
            "score": round(score, 2),
            "network_stats": stats.get('network', {}),
            "auth_events_count": stats.get('auth_events_count', 0),
            "recent_alerts_count": len(alerts),
            "unresolved_alerts_count": len(nids.get_unresolved_alerts()),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching security stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan")
async def trigger_security_scan(background_tasks: BackgroundTasks):
    """Trigger a manual security scan"""
    try:
        # In a real implementation, this would trigger a comprehensive scan
        logger.info("Manual security scan triggered")
        
        return {
            "success": True,
            "message": "Security scan initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering security scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collection/start")
async def start_collection(background_tasks: BackgroundTasks):
    """Start security data collection"""
    try:
        if not collector.is_collecting:
            background_tasks.add_task(collector.start_collection)
            return {
                "success": True,
                "message": "Security data collection started"
            }
        else:
            return {
                "success": False,
                "message": "Collection already running"
            }
    except Exception as e:
        logger.error(f"Error starting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collection/stop")
async def stop_collection():
    """Stop security data collection"""
    try:
        await collector.stop_collection()
        return {
            "success": True,
            "message": "Security data collection stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threats/realtime")
async def get_realtime_threats():
    """Get real-time threat feed"""
    try:
        # Get latest alerts and statistics
        recent_alerts = nids.get_recent_alerts(limit=5)
        unresolved = nids.get_unresolved_alerts()
        
        # Categorize by severity
        by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        for alert in unresolved:
            severity = alert.get('severity', 'low')
            by_severity[severity].append(alert)
        
        return {
            "success": True,
            "recent_alerts": recent_alerts,
            "unresolved_count": len(unresolved),
            "by_severity": {
                "critical": len(by_severity["critical"]),
                "high": len(by_severity["high"]),
                "medium": len(by_severity["medium"]),
                "low": len(by_severity["low"])
            },
            "top_threats": unresolved[:5]  # Top 5 unresolved
        }
    except Exception as e:
        logger.error(f"Error fetching realtime threats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
