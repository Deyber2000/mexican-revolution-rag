#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures
"""

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def test_config():
    """Test configuration for the entire test session"""
    return {
        "openai_api_key": "test_key_12345",
        "model_name": "gpt-3.5-turbo",
        "embedding_model": "text-embedding-ada-002",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,
        "enable_performance_monitoring": False,
        "enable_advanced_calibration": False,
        "enable_statistical_optimization": False,
    }


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses"""
    with patch("openai.OpenAI") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        # Mock chat completion
        mock_instance.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))]
        )

        # Mock embeddings
        mock_instance.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])]
        )

        yield mock_instance


@pytest.fixture
def mock_conversation_history():
    """Mock conversation history for testing"""
    return {
        "test_conv_1": [
            {
                "role": "user",
                "content": "What was the Mexican Revolution?",
                "timestamp": "2025-08-24T10:00:00",
            },
            {
                "role": "assistant",
                "content": "The Mexican Revolution was a major armed struggle...",
                "sources": ["Document Section 1", "Document Section 2"],
                "confidence": 0.85,
                "timestamp": "2025-08-24T10:00:05",
            },
        ],
        "test_conv_2": [
            {
                "role": "user",
                "content": "Who was Emiliano Zapata?",
                "timestamp": "2025-08-24T10:05:00",
            },
            {
                "role": "assistant",
                "content": "Emiliano Zapata was a leading figure...",
                "sources": ["Document Section 3"],
                "confidence": 0.78,
                "timestamp": "2025-08-24T10:05:03",
            },
        ],
    }


@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing"""
    return {
        "message": "What was the Mexican Revolution?",
        "conversation_id": "test_conv_123",
    }


@pytest.fixture
def sample_chat_response():
    """Sample chat response for testing"""
    return {
        "response": "The Mexican Revolution was a major armed struggle...",
        "sources": ["Document Section 1", "Document Section 2"],
        "confidence": 0.85,
        "conversation_id": "test_conv_123",
        "timestamp": datetime.now(),
    }


@pytest.fixture
def sample_conversation_request():
    """Sample conversation request for testing"""
    return {"conversation_id": "test_conv_123"}


@pytest.fixture
def sample_summarize_response():
    """Sample summarize response for testing"""
    return {
        "summary": "This conversation discussed the Mexican Revolution...",
        "conversation_id": "test_conv_123",
    }


@pytest.fixture
def sample_classify_response():
    """Sample classify response for testing"""
    return {
        "category": "historical_figures",
        "confidence": 0.85,
        "conversation_id": "test_conv_123",
    }


@pytest.fixture
def mock_documents():
    """Mock documents for testing"""
    return [
        Mock(
            page_content="The Mexican Revolution was a major armed struggle that began in 1910...",
            metadata={"source": "Document Section 1", "page": 1},
        ),
        Mock(
            page_content="Emiliano Zapata was a leading figure in the Mexican Revolution...",
            metadata={"source": "Document Section 2", "page": 2},
        ),
        Mock(
            page_content="Porfirio Díaz was the dictator who ruled Mexico for over 30 years...",
            metadata={"source": "Document Section 3", "page": 3},
        ),
    ]


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for testing"""
    mock_system = Mock()

    # Mock async methods
    mock_system.process_query = Mock()
    mock_system.initialize = Mock()
    mock_system.load_documents = Mock()
    mock_system.setup_qa_chain = Mock()

    # Mock sync methods
    mock_system.calculate_confidence = Mock(return_value=0.85)
    mock_system.get_config_summary = Mock(
        return_value={
            "model_name": "gpt-3.5-turbo",
            "embedding_model": "text-embedding-ada-002",
            "chunk_size": 1000,
            "top_k": 5,
        }
    )
    mock_system.clear_caches = Mock()
    mock_system.get_cache_statistics = Mock(
        return_value={"concept_cache_size": 0, "similarity_cache_size": 0}
    )

    # Mock properties
    mock_system.confidence_calibration = {}

    return mock_system


@pytest.fixture
def mock_qa_chain():
    """Mock QA chain for testing"""
    mock_chain = Mock()
    mock_chain.ainvoke = Mock(
        return_value={
            "result": "Test response",
            "source_documents": [
                Mock(page_content="Source document 1"),
                Mock(page_content="Source document 2"),
            ],
        }
    )
    return mock_chain


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing"""
    mock_emb = Mock()
    mock_emb.embed_query = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    mock_emb.embed_documents = Mock(
        return_value=[[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]]
    )
    return mock_emb


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()
    mock_store.similarity_search = Mock(
        return_value=[
            Mock(page_content="Relevant document 1"),
            Mock(page_content="Relevant document 2"),
        ]
    )
    return mock_store


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing"""
    mock_ret = Mock()
    mock_ret.get_relevant_documents = Mock(
        return_value=[
            Mock(page_content="Retrieved document 1"),
            Mock(page_content="Retrieved document 2"),
        ]
    )
    return mock_ret


@pytest.fixture
def test_questions():
    """Test questions for RAG system testing"""
    return [
        "What was the Mexican Revolution?",
        "Who was Emiliano Zapata?",
        "When did the Mexican Revolution start?",
        "Who were the main leaders of the revolution?",
        "What role did Zapata play?",
        "How did the revolution end?",
    ]


@pytest.fixture
def test_categories():
    """Test categories for classification testing"""
    return [
        "historical_figures",
        "historical_events",
        "social_impact",
        "political_aspects",
        "general_information",
        "military_aspects",
    ]


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing"""
    env_vars = {
        "OPENAI_API_KEY": "test_key_12345",
        "ENABLE_PERFORMANCE_MONITORING": "false",
        "ENABLE_ADVANCED_CALIBRATION": "false",
        "ENABLE_STATISTICAL_OPTIMIZATION": "false",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Set test environment variables
    os.environ["TESTING"] = "true"

    yield

    # Cleanup after each test
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark all tests in test_rag_system.py as slow
        if "test_rag_system" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark all tests in test_api_endpoints.py as integration
        if "test_api_endpoints" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark all tests in test_models.py and test_utils.py as unit tests
        elif "test_models" in item.nodeid or "test_utils" in item.nodeid:
            item.add_marker(pytest.mark.unit)
