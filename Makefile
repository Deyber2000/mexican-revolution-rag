
.PHONY: help install test run docker-build docker-run clean

# Default target
help:
	@echo "Mexican Revolution RAG Conversational Agent (LangChain + RAGAS)"
	@echo ""
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  ragas-list-metrics       - List available RAGAS metrics"
	@echo "  ragas-evaluate           - Run RAGAS evaluation (all metrics by default)"
	@echo "  ragas-evaluate-custom    - Run RAGAS evaluation with custom metrics (e.g., make ragas-evaluate-custom METRICS=\"Faithfulness,ResponseRelevancy\")"
	@echo "  ragas-evaluate-docker    - Run RAGAS evaluation in Docker container"
	@echo "  run          - Run the application locally (LangChain)"
	@echo "  streamlit    - Run the Streamlit interface"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  docker-all   - Run combined FastAPI + Streamlit in Docker"
	@echo "  clean        - Clean up generated files"
	@echo "  setup        - Complete setup (install + test)"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	pytest tests/ -m unit -v --tb=short

# Run integration tests only
test-integration:
	@echo "Running integration tests..."
	pytest tests/ -m integration -v --tb=short

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run tests excluding slow tests
test-fast:
	@echo "Running fast tests..."
	pytest tests/ -m "not slow" -v --tb=short

# RAGAS Evaluation Commands
ragas-list-metrics:
	@echo "Listing available RAGAS metrics..."
	python -m src.evaluation.ragas_evaluate --list-metrics
	@echo "✅ Metrics listed"

ragas-evaluate:
	@echo "Running RAGAS evaluation (all metrics by default)..."
	python -m src.evaluation.ragas_evaluate
	@echo "✅ RAGAS evaluation completed"

# Usage: make ragas-evaluate-custom METRICS="Faithfulness,ResponseRelevancy"
ragas-evaluate-custom:
	@echo "Running RAGAS evaluation with custom metrics: $(METRICS)"
	python -m src.evaluation.ragas_evaluate --metrics $(shell echo $(METRICS) | tr ',' ' ')
	@echo "✅ RAGAS custom evaluation completed"

ragas-evaluate-docker:
	@echo "Running RAGAS evaluation in Docker container..."
	docker exec mexican-revolution-rag-agent-1 python -m src.evaluation.ragas_evaluate
	@echo "✅ Docker RAGAS evaluation completed"
	@echo "📋 Copying RAGAS evaluation report from container..."
	@docker exec mexican-revolution-rag-agent-1 sh -c "ls -t ragas_evaluation_report_*.json | head -1" > /tmp/latest_ragas_report.txt 2>/dev/null || echo "No RAGAS report found"
	@if [ -s /tmp/latest_ragas_report.txt ]; then \
		RAGAS_REPORT_FILE=$$(cat /tmp/latest_ragas_report.txt); \
		docker cp mexican-revolution-rag-agent-1:/app/$$RAGAS_REPORT_FILE .; \
		echo "✅ RAGAS evaluation report copied: $$RAGAS_REPORT_FILE"; \
	else \
		echo "⚠️  No RAGAS evaluation report found in container"; \
	fi
	@rm -f /tmp/latest_ragas_report.txt

# Run the application (LangChain)
run:
	@echo "Starting LangChain RAG Conversational Agent..."
	uvicorn src.api.app_langchain:app --host 0.0.0.0 --port 8000 --reload

# Run Streamlit interface
streamlit:
	@echo "Starting Streamlit interface..."
	streamlit run src/api/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t mexican-revolution-rag -f src/infrastructure/Dockerfile .
	@echo "✅ Docker image built"

# Run with Docker Compose
docker-run:
	@echo "Starting services with Docker Compose..."
	docker-compose -f docker-compose.yml up --build

# Run combined FastAPI + Streamlit in Docker
docker-all:
	@echo "Starting combined FastAPI + Streamlit in Docker..."
	docker-compose -f docker-compose.yml up --build rag-agent
	@echo "✅ Both services running in Docker"
	@echo "🌐 FastAPI: http://localhost:8000"
	@echo "📊 Streamlit: http://localhost:8501"

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	@echo "✅ Cleanup completed"

# Complete setup
setup: install test
	@echo "✅ Setup completed successfully"

# Development mode
dev:
	@echo "Starting development mode..."
	python run.py &
	sleep 5
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Stop development mode
dev-stop:
	@echo "Stopping development mode..."
	pkill -f "python run.py"
	pkill -f "uvicorn app:app"

# Check system health
health:
	@echo "Checking system health..."
	curl -f http://localhost:8000/health || echo "❌ System not responding"

# Show logs
logs:
	@echo "Showing application logs..."
	docker-compose logs -f rag-agent

# Reset system
reset: clean
	@echo "Resetting system..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ System reset completed" 