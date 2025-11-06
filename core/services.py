# core/services.py
import json
from sqlalchemy.ext.asyncio import AsyncSession
from core.schemas import ChatRequest, StreamResponseInfo, StreamResponseChunk
from db.crud import (
    get_or_create_conversation, 
    add_message_to_conversation,
    get_chat_history_messages,
    get_recent_messages,
    update_conversation_summary
)
from db.models import ChatMessageRole

# --- *** START OF REQUIRED CHANGES *** ---
# Fix imports: Import the 'chain', 'llm', and 'system_prompt' from our new agent
from core.agent import chain, llm, system_prompt
# Add message types for constructing the graph input
from langchain_core.messages import SystemMessage, HumanMessage
# --- *** END OF REQUIRED CHANGES *** ---


async def stream_chat_message(
    request: ChatRequest, db: AsyncSession, background_tasks
):
    """
    Streams AI responses chunk by chunk from the agent (chain.astream).
    """
    # 1. Get or create conversation
    conversation = await get_or_create_conversation(db, request.conversation_id)
    conversation_id = conversation.id

    # 2. Save user's message
    await add_message_to_conversation(
        db, conversation_id, ChatMessageRole.USER, request.message
    )

    # 3. Load history and summary
    history_messages = await get_chat_history_messages(db, conversation_id)
    conversation_summary = conversation.summary or "No summary yet."

    # 4. Yield initial info message
    info_payload = StreamResponseInfo(conversation_id=conversation_id).model_dump_json()
    yield info_payload

    # 5. Start streaming model output
    full_ai_content = ""

    try:
        formatted_system_prompt = system_prompt.format(
            conversation_summary=conversation_summary
        )
        messages_input = [
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=request.message)
        ]

        final_content = None

        async for chunk in chain.astream({"messages": messages_input}):
            if "agent" in chunk:
                agent_output = chunk["agent"]
                if "messages" in agent_output:
                    last_message = agent_output["messages"][-1]
                    
                    if last_message.content and not last_message.tool_calls:
                        final_content = last_message.content

        if final_content:
            full_ai_content = final_content # Save for DB
            chunk_payload = StreamResponseChunk(content=full_ai_content).model_dump_json()
            yield chunk_payload
        else:
            print("[STREAM WARNING] Graph finished without a final AI message.")

    except Exception as e:
        print(f"[STREAM ERROR] {e}")
        error_payload = json.dumps({
            "type": "error",
            "content": "Error during streaming."
        })
        yield error_payload

    finally:
        # 6. Save final AI message
        if full_ai_content:
            await add_message_to_conversation(
                db, conversation_id, ChatMessageRole.AI, full_ai_content
            )

        # 7. Update summary asynchronously
        background_tasks.add_task(generate_and_update_summary, db, conversation_id, llm)


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