from sqlalchemy.ext.asyncio import AsyncSession
from core.schemas import ChatRequest, ChatResponse
from db.crud import (
    get_or_create_conversation, 
    add_message_to_conversation,
    get_chat_history_messages,
    get_recent_messages,
    update_conversation_summary
)
from db.models import ChatMessageRole
from core.tools import llm
from core.agent import create_agent_executor

async def process_chat_message(
    request: ChatRequest, db: AsyncSession
) -> ChatResponse:
    
    # 1. Get or create the conversation
    conversation = await get_or_create_conversation(db, request.conversation_id)
    
    # 2. Save the user's message
    await add_message_to_conversation(
        db, conversation.id, ChatMessageRole.USER, request.message
    )
    
    # 3. Load history and summary for the agent
    history_messages = await get_chat_history_messages(db, conversation.id)
    conversation_summary = conversation.summary or "No summary yet."
    
    # 4. Create an agent bound to this DB session (formatting the system prompt with the conversation summary)
    agent_executor = create_agent_executor(db, conversation_summary)
    ai_response = await agent_executor.ainvoke({
        "conversation_summary": conversation_summary,
        "history": history_messages,
        "input": request.message
    })

    # Extract content from response (handle different shapes)
    if isinstance(ai_response, dict):
        ai_content = ai_response.get("output") or ai_response.get("text") or ai_response.get("output_text") or str(ai_response)
    else:
        ai_content = getattr(ai_response, "content", str(ai_response))
    
    # 5. Save the AI's response
    await add_message_to_conversation(
        db, conversation.id, ChatMessageRole.AI, ai_content
    )
    
    # 6. Update the summary (can be run in background)
    await generate_and_update_summary(db, conversation.id, llm)
    
    # 7. Return the response
    return ChatResponse(response=ai_content, conversation_id=conversation.id)


async def generate_and_update_summary(db: AsyncSession, conversation_id: str, llm):
    """Generates a new summary for the conversation."""
    recent_messages = await get_recent_messages(db, conversation_id, limit=6)
    
    if not recent_messages:
        return

    history_text = "\n".join(
        f"{msg.role.name}: {msg.content}" for msg in recent_messages
    )
    
    summary_prompt = (
        f"Concisely summarize the following conversation, focusing on key loan "
        f"parameters, user preferences, and products discussed.\n\n"
        f"Conversation:\n{history_text}\n\nSummary:"
    )
    
    try:
        summary_response = await llm.ainvoke(summary_prompt)
        new_summary = summary_response.content
        await update_conversation_summary(db, conversation_id, new_summary)
    except Exception as e:
        print(f"Error updating summary: {e}")
