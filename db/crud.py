import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from db.models import Conversation, ChatMessage, ChatMessageRole
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List

async def get_or_create_conversation(db: AsyncSession, conversation_id: str | None) -> Conversation:
    """Gets a conversation by ID or creates a new one."""
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalars().first()
        if conversation:
            return conversation

    # Create new conversation
    new_conversation = Conversation(id=str(uuid.uuid4()), summary="")
    db.add(new_conversation)
    await db.commit()
    await db.refresh(new_conversation)
    return new_conversation

async def add_message_to_conversation(
    db: AsyncSession, 
    conversation_id: str, 
    role: ChatMessageRole, 
    content: str
):
    """Adds a new chat message to the database."""
    message = ChatMessage(
        conversationId=conversation_id,
        role=role,
        content=content
    )
    db.add(message)
    await db.commit()

async def get_chat_history_messages(db: AsyncSession, conversation_id: str) -> list[BaseMessage]:
    """Fetches chat history from DB and converts to LangChain message objects."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversationId == conversation_id)
        .order_by(ChatMessage.createdAt.asc())
    )
    messages = result.scalars().all()
    
    langchain_messages = []
    for msg in messages:
        if msg.role == ChatMessageRole.USER:
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == ChatMessageRole.AI:
            langchain_messages.append(AIMessage(content=msg.content))
    return langchain_messages

async def update_conversation_summary(db: AsyncSession, conversation_id: str, new_summary: str):
    """Updates the summary for a conversation."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalars().first()
    if conversation:
        conversation.summary = new_summary
        await db.commit()

async def get_recent_messages(db: AsyncSession, conversation_id: str, limit: int = 5) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversationId == conversation_id)
        .order_by(ChatMessage.createdAt.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))

# --- NEW CRUD FUNCTIONS ADDED BELOW ---

async def get_all_conversations(db: AsyncSession) -> List[Conversation]:
    """Fetches all conversations, most recent first."""
    result = await db.execute(
        select(Conversation).order_by(Conversation.createdAt.desc())
    )
    return result.scalars().all()

async def get_conversation_by_id(db: AsyncSession, conversation_id: str) -> Conversation | None:
    """Fetches a single conversation by its ID."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    return result.scalars().first()

async def get_messages_for_conversation(db: AsyncSession, conversation_id: str) -> List[ChatMessage]:
    """Fetches all messages for a specific conversation, oldest first."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversationId == conversation_id)
        .order_by(ChatMessage.createdAt.asc())
    )
    return result.scalars().all()