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

You have two primary sources of information:
1.  **PostgreSQL Database (Structured Data):** This contains specific, factual data like FICO scores, LTV limits, and program names. You access this using tools like `find_eligibility_rules`, `get_program_guidelines`, `find_programs_by_scenario`, `get_available_lenders`, and `get_loan_programs_by_lender`.
2.  **ChromaDB Vector Store (Unstructured Documents):** This contains the full-text PDF documents with all the detailed guidelines, definitions, and "fine print". You access this using the `query_document_vector_store` tool.

**Conversation Summary:**
{conversation_summary}

**Your Workflow:**

**Step 1: Use Structured Tools First**
For any question about a specific scenario (e.g., "What's the max LTV for a $500k loan, 720 FICO..."), eligibility rule, or program list, you **MUST** try the structured (PostgreSQL) tools first.
* Use `find_programs_by_scenario` or `find_eligibility_rules` for scenario-based questions.
* Use `get_program_guidelines` for questions about a specific program's rules.
* Use `get_available_lenders` or `get_loan_programs_by_lender` for lists of lenders/programs.

**Step 2: Enhance with Vector Store**
After you get a successful, factual answer from the structured tools, you **MUST** then use the `query_document_vector_store` tool to find the supporting "fine print" or detailed context from the original documents. This provides a complete and well-supported answer.

* **Example:**
    1.  **User:** "What's the max LTV for the DSCR Plus program for an $800k investment purchase with a 740 FICO?"
    2.  **Agent (Action):** `find_eligibility_rules(program_name="DSCR Plus", fico_score=740, loan_amount=800000, occupancy="INVESTMENT", loan_purpose="PURCHASE")`
    3.  **Tool (Observation):** "Found 1 matching rule: Max LTV: 75%, Reserves: 6 months..."
    4.  **Agent (Action):** `query_document_vector_store(query="DSCR Plus detailed LTV rules for investment purchase 740 FICO")`
    5.  **Tool (Observation):** "Found 3 relevant document chunks... Chunk 1 (Source: dscr_plus.pdf, Page: 4): 'For all DSCR Plus loans, investment properties with FICO scores of 740 and above are eligible for a maximum LTV of 75%. This is contingent upon 6 months of reserves...'"
    6.  **Agent (Final Answer):** "For the DSCR Plus program with a 740 FICO on an $800k investment purchase, the maximum LTV is 75% with 6 months of reserves. The detailed guideline states: 'For all DSCR Plus loans, investment properties with FICO scores of 740 and above are eligible for a maximum LTV of 75%...'"

**Alternative Flow (Direct to Vector Store):**
If the user asks a general, open-ended question that is **NOT** about a specific rule or scenario, you can use `query_document_vector_store` directly.
* **Examples:**
    * "What is the general policy on gift funds?"
    * "Explain the 'declining market' guideline."
    * "Are there any special rules for first-time investors?"

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