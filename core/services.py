# core/services.py
import json
from sqlalchemy.ext.asyncio import AsyncSession
from core.schemas import ChatRequest, StreamResponseInfo, StreamResponseChunk
from db.crud import (
    get_or_create_conversation, 
    add_message_to_conversation,
    get_chat_history_messages,
    get_recent_messages,
    get_conversation_by_id,
    update_conversation_summary
)
from db.models import ChatMessageRole

from core.agent import chain, llm, system_prompt
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import StdOutCallbackHandler


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

    handler = StdOutCallbackHandler()

    # Define the config for the stream, now including the callback
    stream_config = {
        "configurable": {"conversation_id": conversation_id},
        "callbacks": [handler],  # <-- This enables verbose logging
        "recursion_limit": 100
    }

    # 5. Start streaming model output
    full_ai_content = ""

    try:
        # Provide the system prompt both the conversation summary and the conversation id
        formatted_system_prompt = system_prompt.format(
            conversation_summary=conversation_summary,
            conversation_id=conversation_id
        )
        messages_input = [
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=request.message)
        ]

        final_content = None

        async for chunk in chain.astream({"messages": messages_input}, stream_config):
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
    # Fetch current conversation to include the current summary
    conversation = await get_conversation_by_id(db, conversation_id)
    conversation_summary = conversation.summary if conversation and conversation.summary else "No summary yet."

    # Fetch a recent window of messages and pick the last 2 user + last 2 AI messages
    recent_messages = await get_recent_messages(db, conversation_id, limit=20)

    if not recent_messages:
        return

    # Separate by role
    user_msgs = [m for m in recent_messages if m.role == ChatMessageRole.USER]
    ai_msgs = [m for m in recent_messages if m.role == ChatMessageRole.AI]

    # Take the most recent two of each (or fewer if not available)
    selected_user = user_msgs[-2:]
    selected_ai = ai_msgs[-2:]

    # Combine and sort chronologically so the agent sees the real turn order
    combined = sorted(selected_user + selected_ai, key=lambda m: m.createdAt)

    history_text = "\n".join(
        f"{msg.role.name}: {msg.content}" for msg in combined
    )

    summary_prompt = f"""
    You are a summarization assistant for an expert mortgage agent. Your purpose is to
    create a concise "briefing" for the agent based on the recent conversation. This
    summary MUST be optimized for the agent's "Triage" workflow.

    **Triage the conversation and structure your summary accordingly:**

    **1. If the user has "Scenario Intent" (providing borrower qualifications):**
    Your summary MUST state this intent and clearly list:
    - **Collected Parameters:** (e.g., FICO: 720, LTV: 75%, Loan Amount: 500k)
    - **Missing Parameters:** (e.g., Occupancy, Loan Purpose)
    - **Status:** (e.g., "Agent just asked for LTV.", "User just provided FICO.")

    **2. If the user has "Program-Specific Intent" (asking about a named program):**
    Your summary MUST state this intent and:
    - **Program Name:** (e.g., "DSCR Plus")
    - **User's Question:** (e.g., "Wants to know max LTV for 740 FICO.")

    **3. If the user has "General Question Intent" (asking an open-ended question):**
    Your summary MUST state this intent and:
    - **Topic:** (e.g., "User is asking for the general policy on gift funds.")

    ---
    **CURRENT SUMMARY:**
    {conversation_summary}

    ---
    **CONVERSATION HISTORY (most recent 2 user turns and 2 AI turns):**
    {history_text}

    ---
    **GENERATED SUMMARY (for the agent's next turn):**
    """
    # --- END NEW SUMMARY PROMPT ---

    try:
        summary_response = await llm.ainvoke(summary_prompt)
        new_summary = summary_response.content
        await update_conversation_summary(db, conversation_id, new_summary)
    except Exception as e:
        print(f"Error updating summary: {e}")