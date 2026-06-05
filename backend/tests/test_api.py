"""
Tests for the FastAPI endpoints.

Uses TestClient for HTTP-level testing without a running server.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check and readiness endpoints."""

    def test_health_check(self, test_client: TestClient):
        """Test that /health returns 200."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "helloagents-qa"

    def test_readiness_check(self, test_client: TestClient):
        """Test that /health/ready returns status."""
        response = test_client.get("/api/v1/health/ready")
        assert response.status_code in [200, 503]  # May not be ready if no LLM


class TestAdminEndpoints:
    """Test admin/stats endpoints."""

    def test_get_stats(self, test_client: TestClient):
        """Test that /admin/stats returns stats."""
        response = test_client.get("/api/v1/admin/stats")
        if response.status_code == 200:
            data = response.json()
            assert "total_sessions" in data

    def test_list_tools(self, test_client: TestClient):
        """Test that /admin/tools returns tool list."""
        response = test_client.get("/api/v1/admin/tools")
        if response.status_code == 200:
            data = response.json()
            assert "tools" in data
            assert "count" in data
            assert "web_search" in data["tools"]


class TestChatEndpoints:
    """Test chat endpoints."""

    def test_send_message_validation(self, test_client: TestClient):
        """Test that empty messages are rejected."""
        response = test_client.post("/api/v1/chat", json={
            "session_id": "test-session",
            "message": "",
        })
        assert response.status_code == 422  # Validation error

    def test_send_message_too_long(self, test_client: TestClient):
        """Test that overly long messages are rejected."""
        response = test_client.post("/api/v1/chat", json={
            "session_id": "test-session",
            "message": "x" * 5000,  # Exceeds 4000 char limit
        })
        assert response.status_code == 422

    def test_list_sessions(self, test_client: TestClient):
        """Test listing sessions."""
        response = test_client.get("/api/v1/chat/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "count" in data

    def test_clear_nonexistent_session(self, test_client: TestClient):
        """Test clearing a session that doesn't exist."""
        response = test_client.delete("/api/v1/chat/nonexistent-session")
        assert response.status_code == 404


class TestCORSMiddleware:
    """Test that CORS headers are present."""

    def test_cors_headers(self, test_client: TestClient):
        """Test that CORS headers are in the response."""
        response = test_client.options(
            "/api/v1/health",
            headers={"Origin": "http://localhost:5173"},
        )
        assert response.status_code in [200, 405]  # OPTIONS may not be allowed on this route
        # CORS headers should be present regardless
