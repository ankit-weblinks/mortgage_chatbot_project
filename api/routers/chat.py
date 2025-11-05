from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import get_db_session
from core.schemas import (
    ChatRequest, 
    ConversationInfo, ConversationDetail,
    StreamResponseInfo, StreamResponseChunk  # Import streaming schemas
)
from core.services import generate_and_update_summary # Correct import
from core.agent import chain, llm
from db import crud
from db.models import ChatMessageRole
from typing import List, AsyncGenerator
import json

router = APIRouter()

@router.post("/chat") # Replaced the old /chat endpoint
async def handle_chat_message_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Main chat endpoint (streaming).
    Handles a chat message, saves it, and streams the AI response.
    Saves the full AI response and updates summary after the stream.
    """
    
    # 1. Get or create the conversation
    conversation = await crud.get_or_create_conversation(db, request.conversation_id)
    conversation_id = conversation.id
    
    # 2. Save the user's message
    await crud.add_message_to_conversation(
        db, conversation_id, ChatMessageRole.USER, request.message
    )
    
    # 3. Load history and summary
    history_messages = await crud.get_chat_history_messages(db, conversation_id)
    conversation_summary = conversation.summary or "No summary yet."
    
    # 4. Define the async generator for the streaming response
    async def response_generator() -> AsyncGenerator[str, None]:
        # First, send the conversation ID
        info_payload = StreamResponseInfo(conversation_id=conversation_id).model_dump_json()
        yield f"{info_payload}\n" # Send as a JSON string with newline delimiter
        
        full_ai_content = ""
        try:
            # Use .astream() to get chunks from the agent chain
            async for chunk in chain.astream({
                "conversation_summary": conversation_summary,
                "history": history_messages,
                "input": request.message
            }):
                # chunk is an AIMessageChunk object
                content = chunk.content
                if content:
                    full_ai_content += content
                    # Send the text chunk
                    chunk_payload = StreamResponseChunk(content=content).model_dump_json()
                    yield f"{chunk_payload}\n" # Send as a JSON string
        
        except Exception as e:
            print(f"Error during LLM stream: {e}")
            # Optionally yield an error chunk to the client
            error_payload = {"type": "error", "content": "Error processing stream."}
            yield f"{json.dumps(error_payload)}\n"
        
        finally:
            # 5. Save the full AI response (after streaming is complete)
            if full_ai_content:
                await crud.add_message_to_conversation(
                    db, conversation_id, ChatMessageRole.AI, full_ai_content
                )
            
            # 6. Schedule summary update (after streaming is complete)
            background_tasks.add_task(generate_and_update_summary, db, conversation_id, llm)

    # 7. Return the streaming response
    # The media type "application/x-json-stream" is commonly used for streaming JSON objects
    return StreamingResponse(response_generator(), media_type="application/x-json-stream")


# --- Endpoints for conversation history (from previous step) ---

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