from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.settings import settings

# 1. Initialize the Groq Chat LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=settings.GROQ_API_KEY,
    temperature=0.7
)

# 2. Define the Chat Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are an expert mortgage assistant. "
            "Previous conversation summary: {conversation_summary}"
        )),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# 3. Create the main chain (stateless)
chain = prompt | llm
