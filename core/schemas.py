from pydantic import BaseModel, ConfigDict
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    
    model_config = ConfigDict(from_attributes=True)
