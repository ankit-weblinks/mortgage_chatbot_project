from fastapi import FastAPI
from api.routers import chat as chat_router

app = FastAPI(
    title="Mortgage AI Chatbot",
    description="A chatbot for querying mortgage guidelines.",
    version="1.0.0"
)

# Include your chat router
app.include_router(chat_router.router, prefix="/api/v1", tags=["Chat"])

@app.get("/", tags=["Health"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Mortgage Chatbot API!"}
