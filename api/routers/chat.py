# api/routers/chat.py
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse

from db.session import get_db_session
from core.schemas import ChatRequest
from core.services import stream_chat_message
from db import crud
from core.schemas import (
    ChatRequest, 
    ConversationInfo, ConversationDetail
)
from typing import List


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
        async def response_generator():
            async for chunk in stream_chat_message(request, db, background_tasks):
                yield f"{chunk}\n"

        return StreamingResponse(
            response_generator(),
            media_type="application/x-json-stream"
        )
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

    # For each conversation include the first user message if present
    infos = []
    for conv in conversations:
        first_msg = await crud.get_first_user_message_for_conversation(db, conv.id)
        infos.append(
            ConversationInfo(
                id=conv.id,
                summary=conv.summary,
                createdAt=conv.createdAt,
                firstUserMessage=first_msg,
            )
        )

    return infos


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Delete a conversation and all its messages. Returns 204 on success or 404 if not found.
    """
    deleted = await crud.delete_conversation_by_id(db, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=204)

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