"""
Network Intrusion Detection System (NIDS)
Analyzes network traffic for suspicious patterns and known attack signatures
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

from api.security.ml.anomaly import anomaly_detector
from api.security.ml.classifier import threat_classifier, ThreatType

logger = logging.getLogger(__name__)


class ThreatSeverity(str, Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatCategory(str, Enum):
    """Threat category types"""
    RECONNAISSANCE = "reconnaissance"
    EXPLOIT = "exploit"
    MALWARE = "malware"
    DOS = "denial_of_service"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALY = "anomaly"


class SecurityAlert:
    """Represents a security alert"""
    
    def __init__(
        self,
        severity: ThreatSeverity,
        category: ThreatCategory,
        description: str,
        source_ip: Optional[str] = None,
        destination_ip: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        self.id = f"alert_{datetime.utcnow().timestamp()}"
        self.timestamp = datetime.utcnow().isoformat()
        self.severity = severity
        self.category = category
        self.description = description
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.metadata = metadata or {}
        self.resolved = False
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'category': self.category.value,
            'description': self.description,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'metadata': self.metadata,
            'resolved': self.resolved
        }


class NetworkIntrusionDetector:
    """Network Intrusion Detection System"""
    
    def __init__(self):
        self.alerts: List[SecurityAlert] = []
        self.signature_db = self._load_signatures()
        
    def _load_signatures(self) -> Dict:
        """Load attack signatures database"""
        return {
            'port_scan': {
                'description': 'Port scanning detected',
                'indicators': ['multiple_connections', 'sequential_ports'],
                'severity': ThreatSeverity.MEDIUM
            },
            'brute_force': {
                'description': 'Brute force attack detected',
                'indicators': ['multiple_auth_failures', 'rapid_succession'],
                'severity': ThreatSeverity.HIGH
            },
            'ddos': {
                'description': 'DDoS attack detected',
                'indicators': ['high_packet_rate', 'multiple_sources'],
                'severity': ThreatSeverity.CRITICAL
            },
            'suspicious_port': {
                'description': 'Connection to suspicious port',
                'indicators': ['known_malware_port'],
                'severity': ThreatSeverity.HIGH
            }
        }
    
    async def analyze_connection(self, connection: Dict) -> Optional[SecurityAlert]:
        """Analyze a single network connection for threats"""
        
        # Check for connections to known malicious ports
        if connection.get('remote_port') in [4444, 5555, 6666, 31337]:
            alert = SecurityAlert(
                severity=ThreatSeverity.HIGH,
                category=ThreatCategory.MALWARE,
                description=f"Connection to suspicious port {connection['remote_port']}",
                source_ip=connection.get('local_addr'),
                destination_ip=connection.get('remote_addr'),
                metadata={'port': connection['remote_port']}
            )
            self.alerts.append(alert)
            logger.warning(f"Suspicious port connection detected: {alert.to_dict()}")
            return alert
        
        return None
    
    async def analyze_traffic_pattern(self, connections: List[Dict]) -> List[SecurityAlert]:
        """Analyze traffic patterns for attacks"""
        new_alerts = []
        
        # Detect port scanning
        if len(connections) > 20:
            # Check if connections are to sequential ports
            ports = [c.get('remote_port') for c in connections if c.get('remote_port')]
            if self._is_sequential_scan(ports):
                alert = SecurityAlert(
                    severity=ThreatSeverity.MEDIUM,
                    category=ThreatCategory.RECONNAISSANCE,
                    description=f"Port scanning detected: {len(connections)} connections",
                    metadata={'connection_count': len(connections), 'ports': ports[:10]}
                )
                new_alerts.append(alert)
                self.alerts.append(alert)
                logger.warning(f"Port scan detected: {alert.to_dict()}")
        
        return new_alerts

    async def analyze_with_ml(self, traffic_stats: Dict[str, float]) -> List[SecurityAlert]:
        """
        Analyze traffic using AI/ML models.
        Input: traffic_stats dict with keys: cpu_percent, memory_percent, bytes_sent, bytes_recv, packet_count
        """
        ml_alerts = []
        
        # 1. Anomaly Detection (Isolation Forest)
        anomaly_result = anomaly_detector.detect(traffic_stats)
        
        if anomaly_result['is_anomaly']:
            # 2. Threat Classification (Classifier)
            # Use same stats as features for now
            classification = threat_classifier.classify(traffic_stats)
            
            description = f"AI Anomaly Detected: {classification['threat_type'].value.upper()} " \
                          f"({classification['confidence']*100:.1f}% confidence)"
            
            # Map ML threat type to NIDS severity
            severity = ThreatSeverity.MEDIUM
            if classification['threat_type'] in [ThreatType.DDoS, ThreatType.DATA_EXFILTRATION]:
                severity = ThreatSeverity.CRITICAL
            elif classification['threat_type'] == ThreatType.BRUTE_FORCE:
                severity = ThreatSeverity.HIGH
                
            alert = SecurityAlert(
                severity=severity,
                category=ThreatCategory.ANOMALY,
                description=description,
                metadata={
                    'anomaly_score': anomaly_result['score'],
                    'classification': classification,
                    'ai_model': 'isolation_forest_v1'
                }
            )
            ml_alerts.append(alert)
            self.alerts.append(alert)
            logger.warning(f"AI Threat Detected: {alert.to_dict()}")
            
        return ml_alerts
    
    def _is_sequential_scan(self, ports: List[int]) -> bool:
        """Check if ports are sequential (indicating a scan)"""
        if len(ports) < 5:
            return False
        
        sorted_ports = sorted(set(ports))
        sequential_count = 0
        
        for i in range(len(sorted_ports) - 1):
            if sorted_ports[i+1] - sorted_ports[i] == 1:
                sequential_count += 1
        
        # If more than 50% of ports are sequential
        return sequential_count > len(sorted_ports) * 0.5
    
    def get_recent_alerts(self, limit: int = 100) -> List[Dict]:
        """Get recent security alerts"""
        return [alert.to_dict() for alert in self.alerts[-limit:]]
    
    def get_unresolved_alerts(self) -> List[Dict]:
        """Get unresolved security alerts"""
        return [alert.to_dict() for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert {alert_id} marked as resolved")
                return True
        return False
    
    def get_security_score(self) -> float:
        """Calculate overall security score (0-100)"""
        if not self.alerts:
            return 100.0
        
        # Count unresolved alerts by severity
        unresolved = [a for a in self.alerts if not a.resolved]
        
        if not unresolved:
            return 95.0
        
        # Weight by severity
        severity_weights = {
            ThreatSeverity.LOW: 1,
            ThreatSeverity.MEDIUM: 3,
            ThreatSeverity.HIGH: 7,
            ThreatSeverity.CRITICAL: 15
        }
        
        total_weight = sum(severity_weights[a.severity] for a in unresolved)
        
        # Calculate score (max penalty is 50 points)
        penalty = min(total_weight, 50)
        score = 100 - penalty
        
        return max(score, 0.0)


# Global NIDS instance
nids = NetworkIntrusionDetector()
