# Test Suite Documentation

This directory contains comprehensive unit tests for the Mexican Revolution RAG Conversational Agent.

## 📁 Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and shared fixtures
├── test_models.py           # Tests for Pydantic models
├── test_rag_system.py       # Tests for RAG system core functionality
├── test_api_endpoints.py    # Tests for FastAPI endpoints
├── test_utils.py            # Tests for utility functions
└── README.md                # This file
```

## 🧪 Test Categories

### 1. **Unit Tests** (`test_models.py`, `test_utils.py`)
- **Purpose**: Test individual components in isolation
- **Scope**: Pydantic models, utility functions, data validation
- **Speed**: Fast execution
- **Dependencies**: Minimal external dependencies

### 2. **Integration Tests** (`test_api_endpoints.py`)
- **Purpose**: Test API endpoints and their interactions
- **Scope**: FastAPI endpoints, request/response handling
- **Speed**: Medium execution
- **Dependencies**: FastAPI TestClient, mocked RAG system

### 3. **System Tests** (`test_rag_system.py`)
- **Purpose**: Test the complete RAG system functionality
- **Scope**: LangChain components, document processing, confidence calculation
- **Speed**: Slow execution (marked with `@pytest.mark.slow`)
- **Dependencies**: Mocked external APIs and services

## 🚀 Running Tests

### Using Make Commands
```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run tests with coverage
make test-coverage

# Run fast tests (excluding slow tests)
make test-fast
```

### Using Pytest Directly
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run tests with specific marker
pytest tests/ -m unit -v
pytest tests/ -m integration -v
pytest tests/ -m "not slow" -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

### Using Make Commands (Recommended)
```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-fast
make test-coverage
```

## 📊 Test Coverage

The test suite aims to provide comprehensive coverage of:

- **Pydantic Models**: 100% validation testing
- **API Endpoints**: All endpoints with success/error scenarios
- **RAG System**: Core functionality with mocked dependencies
- **Utility Functions**: Data validation, error handling, formatting
- **Error Handling**: Edge cases and error scenarios

## 🔧 Test Configuration

### Pytest Configuration (`conftest.py`)
- **Fixtures**: Shared test data and mock objects
- **Markers**: Test categorization (unit, integration, slow)
- **Path Setup**: Automatic src/ directory import
- **Environment**: Test environment variables

### Test Markers
- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: API integration tests
- `@pytest.mark.slow`: Tests that take longer to execute

## 🎯 Test Scenarios

### Model Validation Tests
- ✅ Valid data creation
- ✅ Invalid data rejection
- ✅ Edge case handling
- ✅ Type validation
- ✅ Range validation

### API Endpoint Tests
- ✅ Successful requests
- ✅ Error handling (404, 500, 422)
- ✅ Invalid JSON handling
- ✅ Missing required fields
- ✅ Conversation history management

### RAG System Tests
- ✅ Component initialization
- ✅ Document loading
- ✅ Query processing
- ✅ Confidence calculation
- ✅ Error handling
- ✅ Cache management

### Utility Function Tests
- ✅ Data validation
- ✅ Error formatting
- ✅ Timestamp handling
- ✅ Category validation
- ✅ Conversation formatting

## 🛠️ Mocking Strategy

### External Dependencies
- **OpenAI API**: Mocked responses for consistent testing
- **LangChain Components**: Mocked for isolated testing
- **File System**: Mocked document loading
- **Environment Variables**: Controlled test environment

### Test Data
- **Conversation History**: Predefined test conversations
- **Documents**: Mock document objects with metadata
- **API Responses**: Consistent mock responses
- **Error Scenarios**: Various error conditions

## 📈 Coverage Reports

After running tests with coverage, you can view detailed reports:

```bash
# Generate HTML coverage report
make test-coverage

# View coverage report
open htmlcov/index.html
```

## 🔍 Debugging Tests

### Verbose Output
```bash
pytest tests/ -v -s
```

### Specific Test Debugging
```bash
# Run single test
pytest tests/test_models.py::TestChatRequest::test_valid_chat_request -v -s

# Run tests matching pattern
pytest tests/ -k "test_valid" -v
```

### Test Isolation
```bash
# Run tests in isolation
pytest tests/ --tb=short --maxfail=1
```

## 🚨 Common Issues

### Import Errors
- Ensure you're running from the project root
- Check that `src/` is in the Python path
- Verify virtual environment is activated

### Mock Issues
- Check that mocks are properly configured
- Ensure async mocks are used for async functions
- Verify mock return values match expected types

### Environment Issues
- Set `TESTING=true` environment variable
- Ensure test environment variables are set
- Check that no real API calls are being made

## 📝 Adding New Tests

### Test File Structure
```python
#!/usr/bin/env python3
"""
Unit tests for [component name]
"""

import pytest
from unittest.mock import Mock, patch

class TestComponentName:
    """Test [Component Name] class"""
    
    def test_specific_functionality(self):
        """Test specific functionality"""
        # Arrange
        # Act
        # Assert
```

### Test Naming Conventions
- Test classes: `Test[ComponentName]`
- Test methods: `test_[description]`
- Use descriptive names that explain the test scenario

### Test Organization
- Group related tests in the same class
- Use fixtures for shared setup
- Keep tests independent and isolated
- Use appropriate markers for categorization

## 🎉 Success Criteria

A successful test run should show:
- ✅ All tests passing
- ✅ Good test coverage (>80%)
- ✅ No real API calls made
- ✅ Fast execution for unit tests
- ✅ Comprehensive error scenario coverage

