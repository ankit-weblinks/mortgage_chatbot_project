from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime
from db.models import ChatMessageRole # Import the enum

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    
    model_config = ConfigDict(from_attributes=True)

# --- Schemas for conversation list (from previous step) ---

class ConversationInfo(BaseModel):
    """Minimal info for listing conversations."""
    id: str
    summary: Optional[str] = None
    # First user message in the conversation (if any) to show context in lists
    firstUserMessage: Optional[str] = None
    createdAt: datetime
    
    model_config = ConfigDict(from_attributes=True)

class MessageDetail(BaseModel):
    """Detailed message for viewing a conversation."""
    id: str
    role: ChatMessageRole
    content: str
    createdAt: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ConversationDetail(BaseModel):
    """Full conversation details with all messages."""
    id: str
    summary: Optional[str] = None
    createdAt: datetime
    messages: List[MessageDetail]
    
    model_config = ConfigDict(from_attributes=True)

# --- NEW SCHEMAS ADDED FOR STREAMING ---

class StreamResponseInfo(BaseModel):
    """Payload for the initial 'info' message in a stream."""
    type: str = "info"
    conversation_id: str

class StreamResponseChunk(BaseModel):
    """Payload for a 'chunk' message in a stream."""
    type: str = "chunk"
    content: str