from fastapi import FastAPI
from api.routers import chat as chat_router
from fastapi.middleware.cors import CORSMiddleware  # 1. Import the middleware

app = FastAPI(
    title="Mortgage AI Chatbot",
    description="A chatbot for querying mortgage guidelines.",
    version="1.0.0"
)

# 2. Define your allowed origins
# The origin from your error log is "http://192.168.29.11:8080"
# It's good practice to also include localhost ports for development.
origins = ["*"]

# 3. Add the middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of origins allowed
    allow_credentials=True,    # Allow cookies
    allow_methods=["*"],       # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],       # Allow all headers
)

# Include your chat router
app.include_router(chat_router.router, prefix="/api/v1", tags=["Chat"])

@app.get("/", tags=["Health"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Mortgage Chatbot API!"}