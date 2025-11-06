# api/routers/chat.py
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse

from db.session import get_db_session
from core.schemas import ChatRequest, ChatResponse
from core.services import process_chat_message
from db import crud
from core.schemas import (
    ChatRequest, 
    ConversationInfo, ConversationDetail,
    StreamResponseInfo, StreamResponseChunk
)
from typing import List, AsyncGenerator


router = APIRouter()

@router.post("/chat")
async def chat_with_agent(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Handles a chat request from the user and returns the AI response.
    """
    try:
        response = await process_chat_message(request, db)
        return response
    except Exception as e:
        print(f"[ERROR] Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Endpoints for conversation history (no changes needed) ---

@router.get("/conversations", response_model=List[ConversationInfo])
async def list_conversations(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get a list of all conversations, most recent first.
    """
    conversations = await crud.get_all_conversations(db)
    return conversations

@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_details(
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get the full details and messages for a single conversation.
    """
    conversation = await crud.get_conversation_by_id(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = await crud.get_messages_for_conversation(db, conversation_id)
    
    return ConversationDetail(
        id=conversation.id,
        summary=conversation.summary,
        createdAt=conversation.createdAt,
        messages=messages
    )