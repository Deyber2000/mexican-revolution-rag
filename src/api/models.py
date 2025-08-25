"""
Pydantic models for the RAG conversational agent API
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""

    message: str = Field(..., description="User's message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")


class ConversationRequest(BaseModel):
    """Request model for conversation-based endpoints (summarize, classify)"""

    conversation_id: str = Field(..., description="Conversation ID")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""

    response: str = Field(..., description="Assistant's response")
    conversation_id: str = Field(..., description="Conversation ID")
    confidence: float = Field(..., description="Confidence score (0-1)")
    sources: List[str] = Field(
        default_factory=list, description="Source documents used"
    )
    timestamp: datetime = Field(..., description="Response timestamp")


class ConversationSummary(BaseModel):
    """Model for conversation summary"""

    conversation_id: str = Field(..., description="Conversation ID")
    summary: str = Field(..., description="Generated summary")
    timestamp: datetime = Field(..., description="Summary timestamp")


class ConversationClassification(BaseModel):
    """Model for conversation classification"""

    conversation_id: str = Field(..., description="Conversation ID")
    classification: dict = Field(..., description="Classification results")
    timestamp: datetime = Field(..., description="Classification timestamp")


class EscalationRequest(BaseModel):
    """Request model for escalation"""

    conversation_id: str = Field(..., description="Conversation ID to escalate")
    reason: Optional[str] = Field(None, description="Reason for escalation")


class EscalationResponse(BaseModel):
    """Response model for escalation"""

    conversation_id: str = Field(..., description="Conversation ID")
    escalated: bool = Field(..., description="Whether escalation was successful")
    reason: str = Field(..., description="Reason for escalation")
    timestamp: datetime = Field(..., description="Escalation timestamp")


class DocumentChunk(BaseModel):
    """Model for document chunks"""

    content: str = Field(..., description="Chunk content")
    page: int = Field(..., description="Page number")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class SearchResult(BaseModel):
    """Model for search results"""

    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., description="Similarity score")
    source: str = Field(..., description="Source document")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ConversationHistory(BaseModel):
    """Model for conversation history"""

    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[dict] = Field(default_factory=list, description="Message history")
    created_at: datetime = Field(..., description="Conversation creation time")
    updated_at: datetime = Field(..., description="Last update time")


class SummarizeResponse(BaseModel):
    """Response model for summarize endpoint"""

    summary: str = Field(..., description="Generated summary")
    conversation_id: str = Field(..., description="Conversation ID")


class ClassifyResponse(BaseModel):
    """Response model for classify endpoint"""

    category: str = Field(..., description="Classification category")
    confidence: float = Field(..., description="Classification confidence")
    conversation_id: str = Field(..., description="Conversation ID")


class ConfigResponse(BaseModel):
    """Response model for config endpoint"""

    config: dict = Field(..., description="System configuration")
    calibration_params: dict = Field(..., description="Calibration parameters")


class PerformanceResponse(BaseModel):
    """Response model for performance endpoint"""

    stats: dict = Field(..., description="Performance statistics")


class CalibrationRequest(BaseModel):
    """Request model for calibration endpoint"""

    performance_data: dict = Field(..., description="Performance data for calibration")
