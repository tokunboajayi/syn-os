"""
API endpoint tests.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from synos_api.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health endpoints."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Syn OS"
        assert data["status"] == "operational"

    def test_health(self, client):
        """Test health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestTaskEndpoints:
    """Tests for task endpoints."""

    def test_submit_task(self, client):
        """Test task submission."""
        response = client.post(
            "/api/v1/tasks",
            json={
                "name": "test-task",
                "command": ["echo", "hello"],
                "priority": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"

    def test_get_task_status(self, client):
        """Test getting task status."""
        # First submit a task
        response = client.post(
            "/api/v1/tasks",
            json={"name": "test", "command": ["ls"]},
        )
        task_id = response.json()["task_id"]

        # Then get its status
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id

    def test_get_nonexistent_task(self, client):
        """Test getting nonexistent task."""
        response = client.get("/api/v1/tasks/nonexistent-id")
        assert response.status_code == 404

    def test_list_tasks(self, client):
        """Test listing tasks."""
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_cancel_task(self, client):
        """Test cancelling a task."""
        # Submit task
        response = client.post(
            "/api/v1/tasks",
            json={"name": "cancel-test", "command": ["sleep", "10"]},
        )
        task_id = response.json()["task_id"]

        # Cancel it
        response = client.delete(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200


class TestSystemEndpoints:
    """Tests for system endpoints."""

    def test_system_status(self, client):
        """Test system status."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "uptime_seconds" in data
        assert "ml_models" in data

    def test_metrics(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/v1/system/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "throughput_tasks_per_sec" in data
        assert "latency_p50_ms" in data


class TestMLEndpoints:
    """Tests for ML endpoints."""

    def test_forecast(self, client):
        """Test demand forecast."""
        response = client.post(
            "/api/v1/ml/forecast",
            json={"hours_ahead": 6},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["horizon_hours"] == 6
        assert "cpu" in data
        assert "memory" in data

    def test_list_models(self, client):
        """Test listing models."""
        response = client.get("/api/v1/ml/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) >= 4
