from functools import partial
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.agents import create_agent
from core.tools import (
    llm,  # Import the llm from tools.py
    get_all_lenders,
    get_programs_by_lender, GetProgramsByLenderInput,
    get_guidelines, GetGuidelinesInput,
    search_eligibility_matrix, SearchEligibilityInput,
    query_database_assistant, DatabaseQueryInput
)

# --- The Smart System Prompt ---
SYSTEM_PROMPT = """
You are an expert mortgage assistant. Your goal is to provide accurate and helpful information about mortgage loan programs by querying a database.

You have access to a set of tools. You must follow these rules:

1.  **Always Prefer Specific Tools:** Always use the specific tools (`get_all_lenders`, `get_programs_by_lender`, `get_guidelines`, `search_eligibility_matrix`) when they fit the user's request.
    * For "What lenders do you have?" -> `get_all_lenders`
    * For "What programs does 'Lender X' have?" -> `get_programs_by_lender`
    * For "What are the DTI guidelines for 'Program Y'?" -> `get_guidelines`
    * For "What is the max LTV for a 700 FICO on 'Program Z'?" -> `search_eligibility_matrix`

2.  **Ask for Missing Information:** The `search_eligibility_matrix` and `get_guidelines` tools require a `program_name`. If the user asks a question (e.g., "What's the max LTV for 700 FICO?") without specifying a program, you MUST ask them which program they are interested in *before* calling the tool.

3.  **Use the Backup Tool Sparingly:** Only if no specific tool can answer the question should you use the `query_database_assistant`. This tool takes a natural language query, generates SQL, and runs it.
    * Good Use: "What is the average maxLtv for all programs from Champions Funding, LLC?"
    * Bad Use: "What's the max LTV?" (Use `search_eligibility_matrix` after asking for a program).

4.  **Summarize, Don't Dump:** When a tool returns data (especially JSON), do not just output the raw data. Summarize it in a clear, human-readable way.
    * If `search_eligibility_matrix` returns a rule, say: "Based on the 'Program X' matrix, for a 700 FICO score, the maximum LTV is 80% with 6 months of reserves required. (Notes: ...)"
    * If a tool returns no results, say: "I couldn't find any rules matching those criteria for 'Program X'."

5.  **Use History:** Pay close attention to the `conversation_summary` and `history` to understand the context (e.g., if a program or lender was already mentioned).

Previous conversation summary:
{conversation_summary}
"""

def create_agent_executor(db: AsyncSession, conversation_summary: str = "No summary yet."):
    """
    Factory function to create the agent executor, binding the 
    database session to the tools.
    The system prompt will be formatted with `conversation_summary` so the model
    sees the current conversation context instead of a literal placeholder.

    Returns a compiled agent graph (a Runnable) with methods `ainvoke`/`astream`.
    """
    
    # 1. Create the list of tools, using functools.partial to inject the db session
    #    (and llm for the SQL tool)
    tools = [
        Tool(
            name="get_all_lenders",
            func=partial(get_all_lenders, db),
            description="Use this tool to get a list of all available lender names."
        ),
        Tool(
            name="get_programs_by_lender",
            func=partial(get_programs_by_lender, db),
            args_schema=GetProgramsByLenderInput,
            description="Use this tool to list loan programs for a *specific* lender."
        ),
        Tool(
            name="get_guidelines",
            func=partial(get_guidelines, db),
            args_schema=GetGuidelinesInput,
            description="Use this tool to get specific text-based guidelines (like DTI, Reserves, etc.) for a *specific* loan program. Can also filter by category."
        ),
        Tool(
            name="search_eligibility_matrix",
            func=partial(search_eligibility_matrix, db),
            args_schema=SearchEligibilityInput,
            description="Use this tool to find specific loan criteria (like max LTV, reserves, FICO limits) for a *specific* loan program based on borrower inputs."
        ),
        Tool(
            name="query_database_assistant",
            func=partial(query_database_assistant, db, llm),
            args_schema=DatabaseQueryInput,
            description="A backup tool that takes a natural language query, generates and executes a SQL query against the database. Use only when no other specific tool is appropriate."
        ),
    ]
    
    # 2. Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Format the system prompt with the per-request conversation summary
    formatted_system_prompt = SYSTEM_PROMPT.format(conversation_summary=conversation_summary)

    # 3. Create the agent graph using langchain's create_agent
    #    The `create_agent` function compiles a StateGraph that implements the
    #    Runnable interface (invoke/ainvoke/stream/astream).
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=formatted_system_prompt,
        debug=True,
    )

    return agent_graph
