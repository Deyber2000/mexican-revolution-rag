#!/usr/bin/env python3
"""
Unit tests for FastAPI endpoints
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app_langchain import app


class TestAPIEndpoints:
    """Test FastAPI endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_rag_system(self):
        """Mock RAG system"""
        mock_system = Mock()
        mock_system.process_query = AsyncMock()
        mock_system.get_config_summary = Mock()
        mock_system.confidence_calibration = {}
        return mock_system

    @patch("src.api.app_langchain.rag_system")
    def test_root_endpoint(self, mock_rag_system, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Mexican Revolution RAG Conversational Agent"
        assert data["version"] == "2.0.0"
        assert data["framework"] == "LangChain"
        assert data["status"] == "running"

    @patch("src.api.app_langchain.rag_system")
    def test_health_check_success(self, mock_rag_system, client):
        """Test health check endpoint when RAG system is initialized"""
        mock_rag_system.configure_mock(**{"is not None": True})

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["rag_system"] == "initialized"

    @patch("src.api.app_langchain.rag_system")
    def test_health_check_failure(self, mock_rag_system, client):
        """Test health check endpoint when RAG system is not initialized"""
        # Set rag_system to None to simulate uninitialized state
        import src.api.app_langchain

        original_rag_system = src.api.app_langchain.rag_system
        src.api.app_langchain.rag_system = None

        try:
            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
            assert "not initialized" in data["detail"]
        finally:
            # Restore the original rag_system
            src.api.app_langchain.rag_system = original_rag_system

    @pytest.mark.asyncio
    @patch("src.api.app_langchain.rag_system")
    @patch("src.api.app_langchain.conversation_history")
    async def test_chat_endpoint_success(self, mock_history, mock_rag_system, client):
        """Test successful chat endpoint"""
        # Mock RAG system
        mock_rag_system.configure_mock(**{"is not None": True})
        mock_rag_system.process_query = AsyncMock(
            return_value=(
                "Test response",
                ["Document Section 1", "Document Section 2"],
                0.85,
            )
        )

        # Mock conversation history
        mock_history.__getitem__.return_value = []

        response = client.post(
            "/chat",
            json={
                "message": "What was the Mexican Revolution?",
                "conversation_id": "test_conv_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["sources"] == ["Document Section 1", "Document Section 2"]
        assert data["confidence"] == 0.85
        assert data["conversation_id"] == "test_conv_123"
        assert "timestamp" in data

    @patch("src.api.app_langchain.rag_system")
    def test_chat_endpoint_rag_not_initialized(self, mock_rag_system, client):
        """Test chat endpoint when RAG system is not initialized"""
        # Set rag_system to None to simulate uninitialized state
        import src.api.app_langchain

        original_rag_system = src.api.app_langchain.rag_system
        src.api.app_langchain.rag_system = None

        try:
            response = client.post(
                "/chat",
                json={
                    "message": "What was the Mexican Revolution?",
                    "conversation_id": "test_conv_123",
                },
            )

            assert response.status_code == 503
            data = response.json()
            assert "not initialized" in data["detail"]
        finally:
            # Restore the original rag_system
            src.api.app_langchain.rag_system = original_rag_system

    @pytest.mark.asyncio
    @patch("src.api.app_langchain.rag_system")
    @patch("src.api.app_langchain.conversation_history")
    async def test_chat_endpoint_error(self, mock_history, mock_rag_system, client):
        """Test chat endpoint with processing error"""
        # Mock RAG system
        mock_rag_system.configure_mock(**{"is not None": True})
        mock_rag_system.process_query = AsyncMock(
            side_effect=Exception("Processing error")
        )

        # Mock conversation history
        mock_history.__getitem__.return_value = []

        response = client.post(
            "/chat",
            json={
                "message": "What was the Mexican Revolution?",
                "conversation_id": "test_conv_123",
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "Error processing request" in data["detail"]

    @pytest.mark.asyncio
    @patch("src.api.app_langchain.rag_system")
    @patch("src.api.app_langchain.conversation_history")
    async def test_summarize_endpoint_success(
        self, mock_history, mock_rag_system, client
    ):
        """Test successful summarize endpoint"""
        # Mock RAG system
        mock_rag_system.configure_mock(**{"is not None": True})
        mock_rag_system.process_query = AsyncMock(
            return_value=(
                "This conversation discussed the Mexican Revolution...",
                [],
                0.8,
            )
        )

        # Mock conversation history
        mock_history.__contains__.return_value = True
        mock_history.__getitem__.return_value = [
            {"role": "user", "content": "What was the Mexican Revolution?"},
            {"role": "assistant", "content": "The Mexican Revolution was..."},
        ]

        response = client.post("/summarize", json={"conversation_id": "test_conv_123"})

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert data["conversation_id"] == "test_conv_123"

    @patch("src.api.app_langchain.rag_system")
    @patch("src.api.app_langchain.conversation_history")
    def test_summarize_endpoint_conversation_not_found(
        self, mock_history, mock_rag_system, client
    ):
        """Test summarize endpoint with non-existent conversation"""
        # Mock RAG system to be initialized
        mock_rag_system.configure_mock(**{"is not None": True})

        # Mock conversation_history to return False for nonexistent conversation
        mock_history.__contains__.return_value = False

        response = client.post(
            "/summarize", json={"conversation_id": "nonexistent_conv"}
        )

        assert response.status_code == 404
        data = response.json()
        assert "Conversation not found" in data["detail"]

    @pytest.mark.asyncio
    @patch("src.api.app_langchain.rag_system")
    @patch("src.api.app_langchain.conversation_history")
    async def test_classify_endpoint_success(
        self, mock_history, mock_rag_system, client
    ):
        """Test successful classify endpoint"""
        # Mock RAG system
        mock_rag_system.configure_mock(**{"is not None": True})
        mock_rag_system.process_query = AsyncMock(
            return_value=("historical_figures", [], 0.85)
        )

        # Mock conversation history
        mock_history.__contains__.return_value = True
        mock_history.__getitem__.return_value = [
            {"role": "user", "content": "Who was Emiliano Zapata?"},
            {"role": "assistant", "content": "Emiliano Zapata was..."},
        ]

        response = client.post("/classify", json={"conversation_id": "test_conv_123"})

        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "historical_figures"
        assert data["confidence"] == 0.85
        assert data["conversation_id"] == "test_conv_123"

    @patch("src.api.app_langchain.rag_system")
    @patch("src.api.app_langchain.conversation_history")
    def test_classify_endpoint_conversation_not_found(
        self, mock_history, mock_rag_system, client
    ):
        """Test classify endpoint with non-existent conversation"""
        # Mock RAG system to be initialized
        mock_rag_system.configure_mock(**{"is not None": True})

        # Mock conversation_history to return False for nonexistent conversation
        mock_history.__contains__.return_value = False

        response = client.post(
            "/classify", json={"conversation_id": "nonexistent_conv"}
        )

        assert response.status_code == 404
        data = response.json()
        assert "Conversation not found" in data["detail"]

    @patch("src.api.app_langchain.rag_system")
    def test_config_endpoint_success(self, mock_rag_system, client):
        """Test successful config endpoint"""
        # Mock RAG system
        mock_rag_system.configure_mock(**{"is not None": True})
        mock_rag_system.get_config_summary = Mock(
            return_value={
                "model_name": "gpt-3.5-turbo",
                "embedding_model": "text-embedding-ada-002",
            }
        )
        # Set the confidence_calibration as a property
        mock_rag_system.confidence_calibration = {"param": 0.5}

        response = client.get("/config")

        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "calibration_params" in data
        assert data["config"]["model_name"] == "gpt-3.5-turbo"

    @patch("src.api.app_langchain.rag_system")
    def test_config_endpoint_rag_not_initialized(self, mock_rag_system, client):
        """Test config endpoint when RAG system is not initialized"""
        # Set rag_system to None to simulate uninitialized state
        import src.api.app_langchain

        original_rag_system = src.api.app_langchain.rag_system
        src.api.app_langchain.rag_system = None

        try:
            response = client.get("/config")

            assert response.status_code == 503
            data = response.json()
            assert "not initialized" in data["detail"]
        finally:
            # Restore the original rag_system
            src.api.app_langchain.rag_system = original_rag_system

    @patch("src.api.app_langchain.conversation_history")
    def test_get_conversation_history_success(self, mock_history, client):
        """Test successful get conversation history endpoint"""
        mock_history.__contains__.return_value = True
        mock_history.__getitem__.return_value = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
        ]

        response = client.get("/conversations/test_conv_123")

        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "test_conv_123"
        assert "history" in data
        assert data["message_count"] == 2

    @patch("src.api.app_langchain.conversation_history")
    def test_get_conversation_history_not_found(self, mock_history, client):
        """Test get conversation history endpoint with non-existent conversation"""
        mock_history.__contains__.return_value = False

        response = client.get("/conversations/nonexistent_conv")

        assert response.status_code == 404
        data = response.json()
        assert "Conversation not found" in data["detail"]

    @patch("src.api.app_langchain.conversation_history")
    def test_list_conversations_success(self, mock_history, client):
        """Test successful list conversations endpoint"""
        mock_history.keys.return_value = ["conv_1", "conv_2", "conv_3"]
        mock_history.__len__.return_value = 3

        response = client.get("/conversations")

        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        assert "total_conversations" in data
        assert data["total_conversations"] == 3
        assert len(data["conversations"]) == 3

    def test_invalid_json_request(self, client):
        """Test endpoints with invalid JSON"""
        response = client.post(
            "/chat",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test endpoints with missing required fields"""
        response = client.post(
            "/chat",
            json={"conversation_id": "test"},  # Missing message
        )
        assert response.status_code == 422

        response = client.post(
            "/summarize",
            json={},  # Missing conversation_id
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    @patch("src.api.app_langchain.rag_system")
    @patch("src.api.app_langchain.conversation_history")
    async def test_conversation_history_storage(
        self, mock_history, mock_rag_system, client
    ):
        """Test that conversation history is properly stored"""
        # Mock RAG system
        mock_rag_system.configure_mock(**{"is not None": True})
        mock_rag_system.process_query = AsyncMock(
            return_value=("Test response", ["Document Section 1"], 0.85)
        )

        # Mock conversation history
        mock_history.__contains__.return_value = False
        mock_history.__setitem__ = Mock()

        response = client.post(
            "/chat",
            json={"message": "Test question", "conversation_id": "test_conv_123"},
        )

        assert response.status_code == 200
        # Verify that conversation history was updated
        assert mock_history.__setitem__.called
