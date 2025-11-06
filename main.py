from fastapi import FastAPI, Request, HTTPException
from api.routers import chat as chat_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(
    title="Mortgage AI Chatbot",
    description="A chatbot for querying mortgage guidelines.",
    version="1.0.0"
)

# Define your allowed origins
origins = ["*"]

# Add the middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of origins allowed
    allow_credentials=True,    # Allow cookies
    allow_methods=["*"],       # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],       # Allow all headers
)

# Include your chat router
# API routes are registered first
app.include_router(chat_router.router, prefix="/api/v1", tags=["Chat"])

# --- Static File Serving Logic ---

react_build_path = Path("dist")

if react_build_path.exists():
    print("✅ React build folder found. Serving static files.")
    
    # Serve other assets
    app.mount("/assets", StaticFiles(directory="dist/assets", check_dir=False), name="assets")
    
    @app.get("/manifest.json")
    async def get_manifest():
        return FileResponse("dist/manifest.json")
    
    @app.get("/favicon.ico")
    async def get_favicon():
        return FileResponse("dist/favicon.ico")
    
    # Catch-all route for React Router (must be last)
    @app.get("/{full_path:path}", tags=["Client App"])
    async def serve_react_app(request: Request, full_path: str):
        """
        Serves the React application.
        All routes not matched above (e.g., /api/v1/*, /assets/*, /manifest.json)
        will be handled by this, serving the index.html.
        """
        # The API router is already registered, so /api/v1 calls will be handled before this.
        # We just need to serve the main app.
        return FileResponse("dist/index.html")

else:
    print("⚠️  React build folder 'dist' not found. Please run 'npm run build' first.")
    
    @app.get("/", tags=["Health"])
    async def no_react_build():
        return {"status": "warning", "message": "React app not built. Run 'npm run build' and restart the server to serve the UI."}