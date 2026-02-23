#!/bin/bash
# Syn OS Kubernetes Deployment Script

set -e

echo "=== Syn OS Kubernetes Deployment ==="

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required but not installed."; exit 1; }

# Configuration
NAMESPACE="synos"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR"

echo "Deploying to namespace: $NAMESPACE"

# Create namespace and base resources
echo "1. Creating namespace and base configuration..."
kubectl apply -f "$K8S_DIR/namespace.yaml"
kubectl apply -f "$K8S_DIR/config.yaml"

# Wait for namespace
kubectl wait --for=jsonpath='{.status.phase}'=Active namespace/$NAMESPACE --timeout=30s

# Deploy databases first
echo "2. Deploying databases..."
kubectl apply -f "$K8S_DIR/databases/databases.yaml"

# Wait for databases to be ready
echo "   Waiting for PostgreSQL..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=synos-postgres -n $NAMESPACE --timeout=120s || true

echo "   Waiting for Redis..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=synos-redis -n $NAMESPACE --timeout=120s || true

# Deploy monitoring
echo "3. Deploying monitoring stack..."
kubectl apply -f "$K8S_DIR/monitoring/monitoring.yaml"

# Deploy ML service
echo "4. Deploying ML service..."
kubectl apply -f "$K8S_DIR/ml-service/deployment.yaml"

# Wait for ML service
echo "   Waiting for ML service..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=synos-ml -n $NAMESPACE --timeout=180s || true

# Deploy API
echo "5. Deploying API service..."
kubectl apply -f "$K8S_DIR/api/deployment.yaml"

# Wait for API
echo "   Waiting for API..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=synos-api -n $NAMESPACE --timeout=120s || true

# Deploy ingress
echo "6. Deploying ingress..."
kubectl apply -f "$K8S_DIR/ingress.yaml"

# Deploy security policies
echo "7. Applying security policies..."
kubectl apply -f "$K8S_DIR/security/rbac.yaml"

# Deploy service mesh (if Istio is installed)
if kubectl get namespace istio-system >/dev/null 2>&1; then
    echo "8. Deploying Istio service mesh configuration..."
    kubectl apply -f "$K8S_DIR/service-mesh/istio.yaml"
else
    echo "8. Skipping Istio (not installed)"
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Services:"
kubectl get svc -n $NAMESPACE

echo ""
echo "Pods:"
kubectl get pods -n $NAMESPACE

echo ""
echo "Access points:"
echo "  - API:     http://api.synos.io (configure DNS/ingress)"
echo "  - Grafana: http://grafana.synos.io"
echo ""
echo "For local access:"
echo "  kubectl port-forward svc/synos-api -n synos 8000:80"
echo "  kubectl port-forward svc/synos-grafana -n synos 3000:3000"
