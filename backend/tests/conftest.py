"""
Shared test fixtures for the QA backend tests.

Provides:
- Mock LLM client for deterministic testing
- Test FastAPI client
- In-memory settings with dummy API keys
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""
    def _make_response(text: str = "Test response"):
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = text
        return mock
    return _make_response


@pytest.fixture
def mock_llm():
    """Mock HelloAgentsLLM that returns canned responses."""
    with patch("hello_agents.HelloAgentsLLM") as mock:
        instance = mock.return_value
        instance.invoke.return_value = "This is a mock LLM response."
        instance.think.return_value = iter(["This ", "is ", "a ", "mock ", "stream."])
        instance.model = "mock-model"
        yield instance


@pytest.fixture
def test_settings():
    """Settings configured for testing (no real API keys needed)."""
    from app.config import Settings
    return Settings(
        llm_model_id="test-model",
        llm_api_key="test-key",
        llm_base_url="https://test.api.com/v1",
        tavily_api_key="",
        serpapi_api_key="",
        reflection_enabled=False,  # Disable for faster tests
        max_react_steps=3,
        max_reflection_iterations=1,
    )


@pytest.fixture
def test_client():
    """FastAPI TestClient for API integration tests."""
    from app.main import create_app
    app = create_app()
    return TestClient(app)
