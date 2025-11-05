# core/agent.py
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from config.settings import settings

# Import the new tools you created
# (Make sure your tools.py is in the correct path, e.g., 'core/tools.py' or just 'tools.py')
# Assuming tools.py is in the same 'core' directory:
from core.tools import (
    get_available_lenders,
    get_loan_programs_by_lender,
    get_program_guidelines,
    find_eligibility_rules,
    query_database_assistant
)

# 1. Initialize the Groq Chat LLM
# We use a more powerful model for better agentic behavior and tool-calling
llm = ChatGroq(
    model="openai/gpt-20b",  # Switched to a more capable model
    groq_api_key=settings.GROQ_API_KEY,
    temperature=0.0  # Set to 0.0 for more deterministic and accurate tool use
)

# 2. Define the list of available tools
tools = [
    get_available_lenders,
    get_loan_programs_by_lender,
    get_program_guidelines,
    find_eligibility_rules,
    query_database_assistant  # The "backup" tool
]

# 3. Define the "Smart System Prompt"
# This prompt guides the agent on how and when to use the tools.
system_prompt = """
You are a specialized Mortgage Underwriting Assistant. Your purpose is to provide accurate and detailed information about mortgage lenders, their loan programs, and specific underwriting guidelines.

You have access to a database and a set of specialized tools to answer user questions.

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

# 4. Define the Chat Prompt Template
# This prompt structure is required for the agent to work.
# 'agent_scratchpad' is a special key for the agent's internal thoughts.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 5. Create the Agent
# This binds the LLM, tools, and prompt together.
agent = create_openai_tools_agent(llm, tools, prompt)

# 6. Create the Agent Executor
# This is the runnable object that handles the agent's logic,
# tool execution, and response generation.
# We name it 'chain' for compatibility with your existing 'services.py'.
chain = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,  # Set to True for debugging, False for production
    handle_parsing_errors=True # Gracefully handle any agent errors
)

# Note: The 'llm' object is also exported, as your 'services.py'
# uses it for generating summaries.