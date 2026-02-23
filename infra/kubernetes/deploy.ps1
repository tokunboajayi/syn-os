# Syn OS Kubernetes Deployment Script (PowerShell)

Write-Host "=== Syn OS Kubernetes Deployment ===" -ForegroundColor Cyan

# Check prerequisites
$kubectl = Get-Command kubectl -ErrorAction SilentlyContinue
if (-not $kubectl) {
    Write-Host "kubectl required but not installed." -ForegroundColor Red
    exit 1
}

# Configuration
$NAMESPACE = "synos"
$K8S_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Deploying to namespace: $NAMESPACE" -ForegroundColor Yellow

# Create namespace and base resources
Write-Host "1. Creating namespace and base configuration..." -ForegroundColor Green
kubectl apply -f "$K8S_DIR\namespace.yaml"
kubectl apply -f "$K8S_DIR\config.yaml"

# Wait for namespace
kubectl wait --for=jsonpath='{.status.phase}'=Active namespace/$NAMESPACE --timeout=30s

# Deploy databases first
Write-Host "2. Deploying databases..." -ForegroundColor Green
kubectl apply -f "$K8S_DIR\databases\databases.yaml"

# Wait for databases to be ready
Write-Host "   Waiting for PostgreSQL..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=synos-postgres -n $NAMESPACE --timeout=120s 2>$null

Write-Host "   Waiting for Redis..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=synos-redis -n $NAMESPACE --timeout=120s 2>$null

# Deploy monitoring
Write-Host "3. Deploying monitoring stack..." -ForegroundColor Green
kubectl apply -f "$K8S_DIR\monitoring\monitoring.yaml"

# Deploy ML service
Write-Host "4. Deploying ML service..." -ForegroundColor Green
kubectl apply -f "$K8S_DIR\ml-service\deployment.yaml"

# Wait for ML service
Write-Host "   Waiting for ML service..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=synos-ml -n $NAMESPACE --timeout=180s 2>$null

# Deploy API
Write-Host "5. Deploying API service..." -ForegroundColor Green
kubectl apply -f "$K8S_DIR\api\deployment.yaml"

# Wait for API
Write-Host "   Waiting for API..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=synos-api -n $NAMESPACE --timeout=120s 2>$null

# Deploy ingress
Write-Host "6. Deploying ingress..." -ForegroundColor Green
kubectl apply -f "$K8S_DIR\ingress.yaml"

# Deploy security policies
Write-Host "7. Applying security policies..." -ForegroundColor Green
kubectl apply -f "$K8S_DIR\security\rbac.yaml"

# Deploy service mesh (if Istio is installed)
$istio = kubectl get namespace istio-system 2>$null
if ($istio) {
    Write-Host "8. Deploying Istio service mesh configuration..." -ForegroundColor Green
    kubectl apply -f "$K8S_DIR\service-mesh\istio.yaml"
} else {
    Write-Host "8. Skipping Istio (not installed)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services:" -ForegroundColor Yellow
kubectl get svc -n $NAMESPACE

Write-Host ""
Write-Host "Pods:" -ForegroundColor Yellow
kubectl get pods -n $NAMESPACE

Write-Host ""
Write-Host "Access points:" -ForegroundColor Green
Write-Host "  - API:     http://api.synos.io (configure DNS/ingress)"
Write-Host "  - Grafana: http://grafana.synos.io"
Write-Host ""
Write-Host "For local access:" -ForegroundColor Cyan
Write-Host "  kubectl port-forward svc/synos-api -n synos 8000:80"
Write-Host "  kubectl port-forward svc/synos-grafana -n synos 3000:3000"
