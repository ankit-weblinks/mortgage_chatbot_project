from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import get_db_session
from core.schemas import ChatRequest, ChatResponse
from core.services import generate_and_update_summary
from core.agent import chain, llm
from db.crud import (
    get_or_create_conversation,
    add_message_to_conversation,
    get_chat_history_messages,
)
from db.models import ChatMessageRole

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def handle_chat_message(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Main chat endpoint.
    """
    # 1. Get or create the conversation
    conversation = await get_or_create_conversation(db, request.conversation_id)
    
    # 2. Save the user's message
    await add_message_to_conversation(
        db, conversation.id, ChatMessageRole.USER, request.message
    )
    
    # 3. Load history and summary
    history_messages = await get_chat_history_messages(db, conversation.id)
    conversation_summary = conversation.summary or "No summary yet."
    
    # 4. Invoke the agent chain
    ai_response = await chain.ainvoke({
        "conversation_summary": conversation_summary,
        "history": history_messages,
        "input": request.message
    })
    ai_content = ai_response.content
    
    # 5. Save the AI's response
    await add_message_to_conversation(
        db, conversation.id, ChatMessageRole.AI, ai_content
    )
    
    # 6. Schedule the summary to be updated in the background
    background_tasks.add_task(generate_and_update_summary, db, conversation.id, llm)
    
    # 7. Return the response
    return ChatResponse(response=ai_content, conversation_id=conversation.id)
