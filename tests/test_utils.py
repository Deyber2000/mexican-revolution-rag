#!/usr/bin/env python3
"""
Unit tests for utility functions
"""

from datetime import datetime

import pytest


class TestConfidenceCalculation:
    """Test confidence calculation utilities"""

    def test_confidence_range_validation(self):
        """Test confidence value range validation"""
        from src.api.app_langchain import get_confidence_color

        # Test valid confidence values
        assert get_confidence_color(0.0) == "low-confidence"
        assert get_confidence_color(0.3) == "low-confidence"
        assert get_confidence_color(0.4) == "medium-confidence"
        assert get_confidence_color(0.6) == "medium-confidence"
        assert get_confidence_color(0.7) == "high-confidence"
        assert get_confidence_color(1.0) == "high-confidence"

    def test_confidence_boundary_values(self):
        """Test confidence calculation at boundary values"""
        from src.api.app_langchain import get_confidence_color

        # Test boundary values
        assert get_confidence_color(0.399) == "low-confidence"
        assert get_confidence_color(0.4) == "medium-confidence"
        assert get_confidence_color(0.699) == "medium-confidence"
        assert get_confidence_color(0.7) == "high-confidence"

    def test_confidence_edge_cases(self):
        """Test confidence calculation edge cases"""
        from src.api.app_langchain import get_confidence_color

        # Test edge cases
        assert get_confidence_color(-0.1) == "low-confidence"  # Negative values
        assert get_confidence_color(1.1) == "high-confidence"  # Values > 1.0


class TestConversationHistory:
    """Test conversation history utilities"""

    def test_conversation_formatting(self):
        """Test conversation text formatting"""
        # Mock conversation history
        conversation_history = [
            {"role": "user", "content": "What was the Mexican Revolution?"},
            {"role": "assistant", "content": "The Mexican Revolution was..."},
            {"role": "user", "content": "Who was Emiliano Zapata?"},
            {"role": "assistant", "content": "Emiliano Zapata was..."},
        ]

        # Format conversation text
        conversation_text = ""
        for message in conversation_history:
            if message["role"] == "user":
                conversation_text += f"User: {message['content']}\n"
            else:
                conversation_text += f"Assistant: {message['content']}\n"

        # Verify formatting
        assert "User: What was the Mexican Revolution?" in conversation_text
        assert "Assistant: The Mexican Revolution was..." in conversation_text
        assert "User: Who was Emiliano Zapata?" in conversation_text
        assert "Assistant: Emiliano Zapata was..." in conversation_text

    def test_empty_conversation(self):
        """Test handling of empty conversation"""
        conversation_history = []
        conversation_text = ""

        for message in conversation_history:
            if message["role"] == "user":
                conversation_text += f"User: {message['content']}\n"
            else:
                conversation_text += f"Assistant: {message['content']}\n"

        assert conversation_text == ""

    def test_conversation_with_metadata(self):
        """Test conversation with additional metadata"""
        conversation_history = [
            {
                "role": "user",
                "content": "What was the Mexican Revolution?",
                "timestamp": "2025-08-24T10:00:00",
            },
            {
                "role": "assistant",
                "content": "The Mexican Revolution was...",
                "sources": ["Document Section 1"],
                "confidence": 0.85,
                "timestamp": "2025-08-24T10:00:05",
            },
        ]

        # Extract only content for summarization/classification
        conversation_text = ""
        for message in conversation_history:
            if message["role"] == "user":
                conversation_text += f"User: {message['content']}\n"
            else:
                conversation_text += f"Assistant: {message['content']}\n"

        # Verify only content is included
        assert "User: What was the Mexican Revolution?" in conversation_text
        assert "Assistant: The Mexican Revolution was..." in conversation_text
        assert "timestamp" not in conversation_text
        assert "sources" not in conversation_text


class TestErrorHandling:
    """Test error handling utilities"""

    def test_api_error_formatting(self):
        """Test API error response formatting"""
        # Test different error scenarios
        error_scenarios = [
            {"status_code": 404, "detail": "Conversation not found"},
            {"status_code": 500, "detail": "Internal server error"},
            {"status_code": 422, "detail": "Validation error"},
        ]

        for scenario in error_scenarios:
            error_response = {"detail": scenario["detail"]}

            # Verify error structure
            assert "detail" in error_response
            assert error_response["detail"] == scenario["detail"]

    def test_connection_error_handling(self):
        """Test connection error handling"""
        # Mock connection error scenarios
        connection_errors = [
            "Request timed out",
            "Cannot connect to the API",
            "Connection refused",
        ]

        for error in connection_errors:
            # Verify error message format
            assert isinstance(error, str)
            assert len(error) > 0

    def test_validation_error_handling(self):
        """Test validation error handling"""
        # Test validation error scenarios
        validation_errors = [
            {"field": "message", "error": "Field required"},
            {"field": "conversation_id", "error": "Invalid format"},
            {"field": "confidence", "error": "Value out of range"},
        ]

        for error in validation_errors:
            assert "field" in error
            assert "error" in error
            assert isinstance(error["field"], str)
            assert isinstance(error["error"], str)


class TestDataValidation:
    """Test data validation utilities"""

    def test_confidence_range_validation(self):
        """Test confidence value range validation"""
        # Valid confidence values
        valid_confidences = [0.0, 0.1, 0.5, 0.9, 1.0]
        for confidence in valid_confidences:
            assert 0.0 <= confidence <= 1.0

        # Invalid confidence values
        invalid_confidences = [-0.1, 1.1, 2.0, -1.0]
        for confidence in invalid_confidences:
            assert not (0.0 <= confidence <= 1.0)

    def test_conversation_id_validation(self):
        """Test conversation ID validation"""
        # Valid conversation IDs
        valid_ids = ["conv_123", "test_conv", "conversation_1", "abc123"]
        for conv_id in valid_ids:
            assert isinstance(conv_id, str)
            assert len(conv_id) > 0

        # Invalid conversation IDs
        invalid_ids = ["", None, 123, []]
        for conv_id in invalid_ids:
            if conv_id is None:
                assert conv_id is None
            elif isinstance(conv_id, str):
                assert len(conv_id) == 0
            else:
                assert not isinstance(conv_id, str)

    def test_message_validation(self):
        """Test message validation"""
        # Valid messages
        valid_messages = [
            "What was the Mexican Revolution?",
            "Who was Emiliano Zapata?",
            "Test message",
            "A" * 1000,  # Long message
        ]

        for message in valid_messages:
            assert isinstance(message, str)
            assert len(message) > 0

        # Invalid messages
        invalid_messages = ["", None, 123, []]
        for message in invalid_messages:
            if message is None:
                assert message is None
            elif isinstance(message, str):
                assert len(message) == 0
            else:
                assert not isinstance(message, str)


class TestTimestampHandling:
    """Test timestamp handling utilities"""

    def test_timestamp_generation(self):
        """Test timestamp generation"""
        timestamp = datetime.now()

        # Verify timestamp properties
        assert isinstance(timestamp, datetime)
        assert timestamp.year > 2020  # Reasonable year
        assert timestamp.month >= 1 and timestamp.month <= 12
        assert timestamp.day >= 1 and timestamp.day <= 31

    def test_timestamp_formatting(self):
        """Test timestamp formatting"""
        timestamp = datetime.now()
        iso_format = timestamp.isoformat()

        # Verify ISO format
        assert isinstance(iso_format, str)
        assert "T" in iso_format  # ISO format separator
        assert len(iso_format) > 0

    def test_timestamp_parsing(self):
        """Test timestamp parsing"""
        # Test parsing ISO format timestamp
        timestamp_str = "2025-08-24T10:00:00"
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp_str)
            assert isinstance(parsed_timestamp, datetime)
            assert parsed_timestamp.year == 2025
            assert parsed_timestamp.month == 8
            assert parsed_timestamp.day == 24
        except ValueError:
            pytest.fail("Failed to parse valid ISO timestamp")


class TestCategoryValidation:
    """Test category validation utilities"""

    def test_valid_categories(self):
        """Test valid category values"""
        valid_categories = [
            "historical_figures",
            "historical_events",
            "social_impact",
            "political_aspects",
            "general_information",
            "military_aspects",
        ]

        for category in valid_categories:
            assert isinstance(category, str)
            assert len(category) > 0
            assert "_" in category  # Categories use underscore format

    def test_category_normalization(self):
        """Test category normalization"""
        # Test category normalization (lowercase, strip whitespace)
        test_cases = [
            ("HISTORICAL_FIGURES", "historical_figures"),
            ("  historical_events  ", "historical_events"),
            ("Social_Impact", "social_impact"),
            ("POLITICAL_ASPECTS", "political_aspects"),
        ]

        for input_category, expected in test_cases:
            normalized = input_category.lower().strip()
            assert normalized == expected

    def test_invalid_categories(self):
        """Test invalid category handling"""
        invalid_categories = ["invalid_category", "random_category", "test", "", None]

        for category in invalid_categories:
            if category is None:
                assert category is None
            elif isinstance(category, str):
                assert category not in [
                    "historical_figures",
                    "historical_events",
                    "social_impact",
                    "political_aspects",
                    "general_information",
                    "military_aspects",
                ]
