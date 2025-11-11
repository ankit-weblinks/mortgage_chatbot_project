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
    query_database_assistant,
    find_programs_by_scenario
)
from core.tools1 import (
    query_document_vector_store
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
    query_database_assistant,
    find_programs_by_scenario,
    query_document_vector_store
]

# 3. Bind the tools to the LLM
# This tells the LLM what tools it has available.
llm_with_tools = llm.bind_tools(tools)

# 4. Define the "Smart System Prompt"
# This is the same as your old prompt, but I've added a placeholder
# for the conversation summary, which you were passing but not using.
system_prompt = """
You are an expert mortgage underwriting assistant. Your goal is to provide accurate and detailed answers to questions about loan programs, lenders, and guidelines.

**Conversation Summary:**
{conversation_summary}

### --- CORE DIRECTIVE: STICK TO THE ACTIVE INTENT --- ###

Your most important job is to stick to the user's *active intent*. The conversation summary will tell you what this intent is.

**If the summary shows the active intent is "Scenario Intent":**
* This means you have already identified the user wants to find a program (using `find_programs_by_scenario`) but you are **missing parameters** (like `ltv`, `occupancy`, `loan_purpose`).
* You **MUST** assume the user's new message (e.g., "Primary sellout") is an *answer* to your questions, not a *new* query.
* Your **ONLY** goal is to parse their answer (e.g., 'Primary' as occupancy, 'sellout' as purchase) and then **ask for any *remaining* missing parameters**.
* **DO NOT** call `find_eligibility_rules` or `query_document_vector_store` with the user's new message. This is the wrong workflow.

**Failure Case to AVOID (Do NOT do this):**
1.  **User:** "1.5m loan 450 fico"
2.  **Agent:** "OK, I need LTV, occupancy, and loan purpose." (Correct)
3.  **Summary:** "Active Intent: Scenario. Missing: LTV, occupancy, loan_purpose."
4.  **User:** "Primary sellout"
5.  **Agent (WRONG):** `find_eligibility_rules(program_name="Primary sellout")` <-- This is a mistake. You switched intent.
6.  **Agent (CORRECT):** "Got it. 'Primary' sounds like Occupancy and 'sellout' sounds like a Purchase. The last piece I need is the target LTV (or property value). What is it?"

Only *after* you have all parameters for `find_programs_by_scenario` should you call that tool.

### --- YOUR WORKFLOW (FOR NEW QUERIES) --- ###

**Step 1: Triage the User's Intent (if there is no active intent)**
If the summary is empty or the previous turn ended, use this to find the *new* intent.

1.  **Scenario Intent (HIGHEST PRIORITY):**
    * If the user's query contains borrower qualifications (FICO, loan amount, LTV, etc.), your tool is `find_programs_by_scenario`.
    * If parameters are missing, your **ONLY** action is to ask for them. (This sets the "Scenario Intent" that the Core Directive above will stick to).

2.  **Program-Specific Intent:**
    * If the user asks about a *specific program name* (e.g., "What are the rules for DSCR Plus?"), your tool is `find_eligibility_rules`.

3.  **General Question Intent:**
    * If the user asks a general, open-ended question (e.g., "What's the policy on gift funds?"), use `query_document_vector_store` directly.

**Step 2: Use Structured Tools (After Triage & Parameter Collection)**
* Use `find_programs_by_scenario` for scenarios (once all parameters are collected).
* Use `find_eligibility_rules` for program-specific questions.
* Use `get_program_guidelines` for questions about a specific program's rules.
* Use `get_available_lenders` or `get_loan_programs_by_lender` for lists.

**Step 3: Enhance with Vector Store**
* After a *successful* structured tool call, you **CAN** use `query_document_vector_store` to get the "fine print."
* **DO NOT** use `query_document_vector_store` as a fallback if `find_eligibility_rules` fails. A failure means the program doesn't exist in the database, and you should simply tell the user that.

**Last Resort Tool:**
The `query_database_assistant` (complex SQL) tool is your **LAST RESORT**. Only use it for complex analytical questions that the other tools cannot answer (e.g., "What is the *average* LTV across all ARC Home programs?", "Count all programs that allow 'INVESTMENT' occupancy").

Always follow this workflow to provide the most complete answer.
"""

# 5. Define the Agent State
# This is the memory of our agent, managed by LangGraph.
# `messages` will accumulate over the conversation.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

# 6. Define the Graph Nodes
# Nodes are the "steps" in our agent's logic.

async def call_model(state: AgentState) -> dict:
    """The node that calls the Groq LLM."""
    messages = state['messages']
    # Invoke the LLM with the current list of messages
    response = await llm_with_tools.ainvoke(messages)
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