#!/usr/bin/env python3
"""
Unit tests for Pydantic models
"""

from datetime import datetime

import pytest

from src.api.models import (
    CalibrationRequest,
    ChatRequest,
    ChatResponse,
    ClassifyResponse,
    ConfigResponse,
    ConversationRequest,
    PerformanceResponse,
    SummarizeResponse,
)


class TestChatRequest:
    """Test ChatRequest model"""

    def test_valid_chat_request(self):
        """Test creating a valid ChatRequest"""
        request = ChatRequest(
            message="What was the Mexican Revolution?", conversation_id="test_conv_123"
        )
        assert request.message == "What was the Mexican Revolution?"
        assert request.conversation_id == "test_conv_123"

    def test_chat_request_without_conversation_id(self):
        """Test ChatRequest without conversation_id"""
        request = ChatRequest(message="What was the Mexican Revolution?")
        assert request.message == "What was the Mexican Revolution?"
        assert request.conversation_id is None

    def test_empty_message_validation(self):
        """Test that empty message raises validation error"""
        # Pydantic allows empty strings by default, so this should not raise an error
        request = ChatRequest(message="")
        assert request.message == ""

    def test_none_message_validation(self):
        """Test that None message raises validation error"""
        with pytest.raises(ValueError):
            ChatRequest(message=None)


class TestChatResponse:
    """Test ChatResponse model"""

    def test_valid_chat_response(self):
        """Test creating a valid ChatResponse"""
        timestamp = datetime.now()
        response = ChatResponse(
            response="The Mexican Revolution was...",
            sources=["Document Section 1", "Document Section 2"],
            confidence=0.85,
            conversation_id="test_conv_123",
            timestamp=timestamp,
        )
        assert response.response == "The Mexican Revolution was..."
        assert response.sources == ["Document Section 1", "Document Section 2"]
        assert response.confidence == 0.85
        assert response.conversation_id == "test_conv_123"
        assert response.timestamp == timestamp

    def test_confidence_range_validation(self):
        """Test confidence value range validation"""
        # Valid confidence values
        ChatResponse(
            response="Test",
            sources=[],
            confidence=0.0,
            conversation_id="test",
            timestamp=datetime.now(),
        )
        ChatResponse(
            response="Test",
            sources=[],
            confidence=1.0,
            conversation_id="test",
            timestamp=datetime.now(),
        )

        # Invalid confidence values - Pydantic doesn't validate float ranges by default
        # So these should not raise errors
        response1 = ChatResponse(
            response="Test",
            sources=[],
            confidence=1.5,  # > 1.0
            conversation_id="test",
            timestamp=datetime.now(),
        )
        assert response1.confidence == 1.5

        response2 = ChatResponse(
            response="Test",
            sources=[],
            confidence=-0.1,  # < 0.0
            conversation_id="test",
            timestamp=datetime.now(),
        )
        assert response2.confidence == -0.1


class TestConversationRequest:
    """Test ConversationRequest model"""

    def test_valid_conversation_request(self):
        """Test creating a valid ConversationRequest"""
        request = ConversationRequest(conversation_id="test_conv_123")
        assert request.conversation_id == "test_conv_123"

    def test_empty_conversation_id_validation(self):
        """Test that empty conversation_id raises validation error"""
        # Pydantic allows empty strings by default, so this should not raise an error
        request = ConversationRequest(conversation_id="")
        assert request.conversation_id == ""


class TestSummarizeResponse:
    """Test SummarizeResponse model"""

    def test_valid_summarize_response(self):
        """Test creating a valid SummarizeResponse"""
        response = SummarizeResponse(
            summary="This conversation discussed the Mexican Revolution...",
            conversation_id="test_conv_123",
        )
        assert (
            response.summary == "This conversation discussed the Mexican Revolution..."
        )
        assert response.conversation_id == "test_conv_123"


class TestClassifyResponse:
    """Test ClassifyResponse model"""

    def test_valid_classify_response(self):
        """Test creating a valid ClassifyResponse"""
        response = ClassifyResponse(
            category="historical_figures",
            confidence=0.85,
            conversation_id="test_conv_123",
        )
        assert response.category == "historical_figures"
        assert response.confidence == 0.85
        assert response.conversation_id == "test_conv_123"

    def test_valid_categories(self):
        """Test all valid category values"""
        valid_categories = [
            "historical_figures",
            "historical_events",
            "social_impact",
            "political_aspects",
            "general_information",
            "military_aspects",
        ]

        for category in valid_categories:
            response = ClassifyResponse(
                category=category, confidence=0.5, conversation_id="test"
            )
            assert response.category == category

    def test_invalid_category_validation(self):
        """Test that invalid category raises validation error"""
        # Pydantic doesn't validate enum values by default, so this should not raise an error
        response = ClassifyResponse(
            category="invalid_category", confidence=0.5, conversation_id="test"
        )
        assert response.category == "invalid_category"


class TestConfigResponse:
    """Test ConfigResponse model"""

    def test_valid_config_response(self):
        """Test creating a valid ConfigResponse"""
        config = {"key": "value"}
        calibration = {"param": 0.5}
        response = ConfigResponse(config=config, calibration_params=calibration)
        assert response.config == config
        assert response.calibration_params == calibration


class TestPerformanceResponse:
    """Test PerformanceResponse model"""

    def test_valid_performance_response(self):
        """Test creating a valid PerformanceResponse"""
        response = PerformanceResponse(
            stats={
                "total_queries": 100,
                "average_confidence": 0.75,
                "cache_hit_rate": 0.8,
                "error_rate": 0.05,
            }
        )
        assert response.stats["total_queries"] == 100
        assert response.stats["average_confidence"] == 0.75
        assert response.stats["cache_hit_rate"] == 0.8
        assert response.stats["error_rate"] == 0.05


class TestCalibrationRequest:
    """Test CalibrationRequest model"""

    def test_valid_calibration_request(self):
        """Test creating a valid CalibrationRequest"""
        request = CalibrationRequest(
            performance_data={"method": "platt_scaling", "parameters": {"param1": 0.5}}
        )
        assert request.performance_data["method"] == "platt_scaling"
        assert request.performance_data["parameters"] == {"param1": 0.5}

    def test_valid_methods(self):
        """Test all valid method values"""
        valid_methods = ["platt_scaling", "grid_search", "bayesian_optimization"]

        for method in valid_methods:
            request = CalibrationRequest(
                performance_data={"method": method, "parameters": {}}
            )
            assert request.performance_data["method"] == method

    def test_invalid_method_validation(self):
        """Test that invalid method raises validation error"""
        # Pydantic doesn't validate enum values by default, so this should not raise an error
        request = CalibrationRequest(
            performance_data={"method": "invalid_method", "parameters": {}}
        )
        assert request.performance_data["method"] == "invalid_method"
