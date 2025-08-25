"""
FastAPI application for the Mexican Revolution RAG Conversational Agent
Using LangChain-based RAG system
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
from src.core.langchain_rag_system import LangChainRAGSystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mexican Revolution RAG Conversational Agent",
    description="A RAG system for answering questions about the Mexican Revolution using LangChain",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[LangChainRAGSystem] = None

# In-memory conversation storage (in production, use a database)
conversation_history: Dict[str, List[Dict]] = {}
MAX_CONVERSATION_LENGTH = 20  # Maximum messages per conversation
CONVERSATION_EXPIRY_HOURS = 24  # Hours before conversation expires


def cleanup_expired_conversations():
    """Remove conversations older than the expiry time"""
    current_time = datetime.now()
    expired_conversations = []

    for conv_id, history in conversation_history.items():
        if history:
            # Check if the last message is older than expiry time
            last_message_time = datetime.fromisoformat(history[-1]["timestamp"])
            if current_time - last_message_time > timedelta(
                hours=CONVERSATION_EXPIRY_HOURS
            ):
                expired_conversations.append(conv_id)

    for conv_id in expired_conversations:
        del conversation_history[conv_id]
        logger.info(f"Cleaned up expired conversation: {conv_id}")


def limit_conversation_length(conversation_id: str):
    """Limit conversation to maximum length, keeping most recent messages"""
    if conversation_id in conversation_history:
        history = conversation_history[conversation_id]
        if len(history) > MAX_CONVERSATION_LENGTH:
            # Keep the most recent messages
            conversation_history[conversation_id] = history[-MAX_CONVERSATION_LENGTH:]
            logger.info(
                f"Limited conversation {conversation_id} to {MAX_CONVERSATION_LENGTH} messages"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    global rag_system

    try:
        logger.info("Initializing LangChain RAG system...")

        # Create configuration
        config = LangChainRAGSystem.create_config()

        # Override features based on environment variables
        config["enable_performance_monitoring"] = ENABLE_PERFORMANCE_MONITORING
        config["enable_advanced_calibration"] = ENABLE_ADVANCED_CALIBRATION
        config["enable_statistical_optimization"] = ENABLE_STATISTICAL_OPTIMIZATION

        # Initialize RAG system
        rag_system = LangChainRAGSystem(config)
        await rag_system.initialize()

        logger.info("LangChain RAG system initialized successfully")
        logger.info(f"Active configuration: {rag_system.get_config_summary()}")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    finally:
        # Cleanup if needed
        logger.info("Shutting down RAG system...")


# Update app with lifespan
app.router.lifespan_context = lifespan


def get_confidence_color(confidence: float) -> str:
    """Get CSS class for confidence level"""
    if confidence < 0.4:
        return "low-confidence"
    elif confidence < 0.7:
        return "medium-confidence"
    else:
        return "high-confidence"


# Environment variables
ENABLE_PERFORMANCE_MONITORING = (
    os.getenv("ENABLE_PERFORMANCE_MONITORING", "false").lower() == "true"
)
ENABLE_ADVANCED_CALIBRATION = (
    os.getenv("ENABLE_ADVANCED_CALIBRATION", "false").lower() == "true"
)
ENABLE_STATISTICAL_OPTIMIZATION = (
    os.getenv("ENABLE_STATISTICAL_OPTIMIZATION", "false").lower() == "true"
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mexican Revolution RAG Conversational Agent",
        "version": "2.0.0",
        "framework": "LangChain",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_system_initialized": rag_system is not None,
        "active_conversations": len(conversation_history),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Clean up expired conversations
        cleanup_expired_conversations()

        # Generate conversation ID if not provided
        conversation_id = (
            request.conversation_id or f"conv_{datetime.now().timestamp()}"
        )

        # Get conversation history for context
        current_history = conversation_history.get(conversation_id, [])

        # Process the query with conversation context (stateless approach)
        response, sources, confidence = await rag_system.process_query(
            request.message, current_history
        )

        # Store conversation history for reference
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = []

        conversation_history[conversation_id].append(
            {
                "role": "user",
                "content": request.message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        conversation_history[conversation_id].append(
            {
                "role": "assistant",
                "content": response,
                "sources": sources,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Limit conversation length
        limit_conversation_length(conversation_id)

        return ChatResponse(
            response=response,
            sources=sources,
            confidence=confidence,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_conversation(request: ConversationRequest):
    """Summarize a conversation using the RAG system"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Get conversation history
        if request.conversation_id not in conversation_history:
            raise HTTPException(status_code=404, detail="Conversation not found")

        history = conversation_history[request.conversation_id]

        if not history:
            raise HTTPException(
                status_code=400, detail="No conversation history to summarize"
            )

        # Create a summary prompt
        conversation_text = ""
        for message in history:
            if message["role"] == "user":
                conversation_text += f"User: {message['content']}\n"
            else:
                conversation_text += f"Assistant: {message['content']}\n"

        summary_prompt = f"""Please provide a concise summary of the following conversation about the Mexican Revolution:

{conversation_text}

Summary:"""

        # Use the RAG system to generate summary
        summary, _, _ = await rag_system.process_query(summary_prompt)

        return SummarizeResponse(
            summary=summary, conversation_id=request.conversation_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error summarizing conversation: {str(e)}"
        )


@app.post("/classify", response_model=ClassifyResponse)
async def classify_conversation(request: ConversationRequest):
    """Classify a conversation using the RAG system"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Get conversation history
        if request.conversation_id not in conversation_history:
            raise HTTPException(status_code=404, detail="Conversation not found")

        history = conversation_history[request.conversation_id]

        if not history:
            raise HTTPException(
                status_code=400, detail="No conversation history to classify"
            )

        # Create a classification prompt
        conversation_text = ""
        for message in history:
            if message["role"] == "user":
                conversation_text += f"User: {message['content']}\n"
            else:
                conversation_text += f"Assistant: {message['content']}\n"

        classification_prompt = f"""Please classify the following conversation about the Mexican Revolution into one of these categories:

Categories:
- historical_figures: Questions about specific people (Zapata, Villa, Díaz, etc.)
- historical_events: Questions about specific events, battles, or dates
- social_impact: Questions about social changes, reforms, or consequences
- political_aspects: Questions about political changes, constitutions, or governance
- general_information: General questions about the revolution
- military_aspects: Questions about battles, strategies, or military leaders

Conversation:
{conversation_text}

Please respond with only the category name:"""

        # Use the RAG system to classify
        classification, _, confidence = await rag_system.process_query(
            classification_prompt
        )

        # Clean up the classification response
        category = classification.strip().lower()
        if category not in [
            "historical_figures",
            "historical_events",
            "social_impact",
            "political_aspects",
            "general_information",
            "military_aspects",
        ]:
            category = "general_information"  # Default fallback

        return ClassifyResponse(
            category=category,
            confidence=confidence,
            conversation_id=request.conversation_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying conversation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error classifying conversation: {str(e)}"
        )


@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history for a specific conversation ID"""
    if conversation_id not in conversation_history:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "history": conversation_history[conversation_id],
        "message_count": len(conversation_history[conversation_id]),
    }


@app.get("/conversations")
async def list_conversations():
    """List all active conversations"""
    cleanup_expired_conversations()
    return {
        "conversations": [
            {
                "conversation_id": conv_id,
                "message_count": len(history),
                "last_activity": history[-1]["timestamp"] if history else None,
            }
            for conv_id, history in conversation_history.items()
        ],
        "total_conversations": len(conversation_history),
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation"""
    if conversation_id in conversation_history:
        del conversation_history[conversation_id]
        return {"message": f"Conversation {conversation_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


@app.delete("/conversations")
async def clear_all_conversations():
    """Clear all conversations"""
    conversation_history.clear()
    return {"message": "All conversations cleared"}


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current system configuration"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        config_summary = rag_system.get_config_summary()
        calibration_params = rag_system.confidence_calibration

        return ConfigResponse(
            config=config_summary, calibration_params=calibration_params
        )

    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting configuration: {str(e)}"
        )


@app.get("/performance", response_model=PerformanceResponse)
async def get_performance_stats():
    """Get performance statistics"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        if rag_system.performance_stats:
            stats = rag_system.performance_stats
        else:
            stats = {"note": "Performance monitoring disabled"}

        return PerformanceResponse(stats=stats)

    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting performance stats: {str(e)}"
        )


@app.post("/calibrate")
async def calibrate_system(request: CalibrationRequest):
    """Calibrate the system based on performance data"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # This would trigger calibration if enabled
        if rag_system.config["enable_advanced_calibration"]:
            rag_system.calibrate_confidence_parameters(request.performance_data)
            return {"message": "Calibration completed successfully"}
        else:
            return {"message": "Advanced calibration is disabled"}

    except Exception as e:
        logger.error(f"Error during calibration: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error during calibration: {str(e)}"
        )


@app.post("/clear-caches")
async def clear_caches():
    """Clear system caches"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        rag_system.clear_caches()
        return {"message": "Caches cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing caches: {str(e)}")


@app.get("/docs")
async def get_documentation():
    """Get API documentation"""
    return {
        "endpoints": {
            "GET /": "Root endpoint with system information",
            "GET /health": "Health check endpoint",
            "POST /chat": "Process a chat message",
            "POST /summarize": "Summarize a conversation",
            "POST /classify": "Classify a conversation",
            "GET /config": "Get system configuration",
            "GET /performance": "Get performance statistics",
            "POST /calibrate": "Calibrate the system",
            "POST /clear-caches": "Clear system caches",
        },
        "framework": "LangChain",
        "version": "2.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
