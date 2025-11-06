# core/services.py
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

# --- *** START OF REQUIRED CHANGES *** ---
# Fix imports: Import the 'chain', 'llm', and 'system_prompt' from our new agent
from core.agent import chain, llm, system_prompt
# Add message types for constructing the graph input
from langchain_core.messages import SystemMessage, HumanMessage
# --- *** END OF REQUIRED CHANGES *** ---


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
    # history_messages = await get_chat_history_messages(db, conversation.id)
    conversation_summary = conversation.summary or "No summary yet."
    
    # --- *** START OF REQUIRED CHANGES *** ---

    # 4. Format the system prompt
    final_system_prompt = system_prompt.format(
        conversation_summary=conversation_summary
    )
    
    # 5. Construct the input message list
    input_messages = [SystemMessage(content=final_system_prompt)]
    # input_messages.extend(history_messages)
    input_messages.append(HumanMessage(content=request.message))
    
    # 6. Create the graph input
    graph_input = {"messages": input_messages}
    
    # 7. Invoke the LangGraph agent
    # The output is the final state of the graph
    ai_response_state = await chain.ainvoke(graph_input)

    # 8. Extract the final message from the state
    ai_content = ""
    if "messages" in ai_response_state and ai_response_state["messages"]:
        # The last message in the state is the AI's final response
        ai_content = ai_response_state["messages"][-1].content
    else:
        ai_content = "I'm sorry, I encountered an error."
    
    # --- *** END OF REQUIRED CHANGES *** ---

    # 9. Save the AI's response
    await add_message_to_conversation(
        db, conversation.id, ChatMessageRole.AI, ai_content
    )
    
    # 10. Update the summary (can be run in background)
    # This uses the 'llm' object we exported from core/agent.py
    await generate_and_update_summary(db, conversation.id, llm)
    
    # 11. Return the response
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