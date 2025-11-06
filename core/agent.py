# core/agent.py
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
# from langgraph.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolNode

from config.settings import settings
from core.tools import (
    get_available_lenders,
    get_loan_programs_by_lender,
    get_program_guidelines,
    find_eligibility_rules,
    query_database_assistant
)

# 1. Define the LLM
# We use a model that is good at tool calling, as recommended by Groq docs.
# 'llama3-70b-8192' is a great choice.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=settings.GROQ_API_KEY,
    temperature=0.0
)

# 2. Define the list of available tools
tools = [
    get_available_lenders,
    get_loan_programs_by_lender,
    get_program_guidelines,
    find_eligibility_rules,
    query_database_assistant
]

# 3. Bind the tools to the LLM
# This tells the LLM what tools it has available.
llm_with_tools = llm.bind_tools(tools)

# 4. Define the "Smart System Prompt"
# This is the same as your old prompt, but I've added a placeholder
# for the conversation summary, which you were passing but not using.
system_prompt = """
You are a specialized Mortgage Underwriting Assistant. Your purpose is to provide accurate and detailed information about mortgage lenders, their loan programs, and specific underwriting guidelines.

You have access to a database and a set of specialized tools to answer user questions.

**Conversation Summary:**
{conversation_summary}

**Your primary goal is to be accurate and helpful. Follow these rules:**

1.  **Prioritize Specialized Tools:** ALWAYS prefer to use a specific tool if it matches the user's intent. These tools are faster and more reliable.
    * For "List all lenders": Use `get_available_lenders`.
    * For "What programs does [Lender X] have?": Use `get_loan_programs_by_lender`.
    * For "What are the guidelines for [Program Y]?": Use `get_program_guidelines`.
    * For "What is the max LTV for [Program Z] with FICO 720...": Use `find_eligibility_rules`.

2.  **Handle Ambiguous Program Names:** The `get_program_guidelines` and `find_eligibility_rules` tools have built-in fuzzy matching. Trust them to find the correct program even if the user misspells it. Do not ask the user to clarify spelling unless the tool fails to find a match.

3.  **Use the "Backup Tool" (query_database_assistant) Sparingly:**
    * You should **ONLY** use the `query_database_assistant` tool as a last resort.
    * Use it **ONLY** for complex, analytical, or aggregate questions that the other tools *cannot* answer.
    * **Examples of GOOD use:** "What is the average max LTV across all programs?", "Count all programs that allow INVESTMENT occupancy", "List all lenders and the count of their programs."
    * **Examples of BAD use:** "What are the guidelines for DSCR Plus?" (Use `get_program_guidelines`), "What programs does Lender X have?" (Use `get_loan_programs_by_lender`).
    * When you do use `query_database_assistant`, pass the user's full, natural-language question directly to it.

4.  **Be Clear and Professional:**
    * When you return data, format it clearly using markdown (like bullet points).
    * Do not say "I searched the database...". Just present the information.
    * **Incorrect:** "I found in the database that the max LTV is 80%."
    * **Correct:** "The max LTV for that scenario is 80%."

5.  **Handle Failures Gracefully:** If a tool returns an error or no results, inform the user clearly and politely. (e.g., "I couldn't find any guidelines for a program matching that name.")
    
6.  **Conversation Context:** You will be given the previous chat history. Use it to understand the user's current question in context.
"""

# 5. Define the Agent State
# This is the memory of our agent, managed by LangGraph.
# `messages` will accumulate over the conversation.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

# 6. Define the Graph Nodes
# Nodes are the "steps" in our agent's logic.

def call_model(state: AgentState) -> dict:
    """The node that calls the Groq LLM."""
    messages = state['messages']
    # Invoke the LLM with the current list of messages
    response = llm_with_tools.invoke(messages)
    # Return the AI's response to be added to the state
    return {"messages": [response]}

# Use the pre-built ToolNode for simplicity.
# This node automatically executes the tools called by the LLM.
tool_node = ToolNode(tools)

# 7. Define the Conditional Edge
# This function decides what to do after the LLM responds.
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Decides whether to call tools or end the conversation.
    """
    last_message = state['messages'][-1]
    # If the LLM's last message includes tool calls, route to the 'tools' node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, end the graph execution
    return "__end__"

# 8. Build the Graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.set_entry_point("agent")

# Add the conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "__end__": "__end__"
    }
)

# Add the edge from the tools back to the agent
workflow.add_edge("tools", "agent")

# 9. Compile the Graph
# This creates the runnable `chain` object.
chain = workflow.compile()

# Note: We still export 'llm' for the summary service in 'services.py'.