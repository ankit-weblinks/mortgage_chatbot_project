# ... other imports
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from config.settings import settings # Assuming your config has the VSTORE_DIR, etc.

# --- Vector Store Tool ---

# Helper function to initialize the persistent vector store
# We cache this so we don't re-load the model and DB connection on every call
_vector_store = None

def _get_vector_store():
    """
    Initializes and returns a connection to the *persistent* Chroma vector store
    based on the configuration in settings.py.
    Caches the connection globally to avoid re-loading on every call.
    """
    global _vector_store
    
    # If already initialized, return the cached connection
    if _vector_store is not None:
        return _vector_store

    try:
        # --- CRITICAL FIX ---
        # We must ensure these settings are in your `config/settings.py` file.
        # Based on your snippet, these are the correct values.
        
        # 1. Check if settings are present
        if not all([settings.VSTORE_DIR, settings.COLLECTION_NAME, settings.EMBEDDING_MODEL]):
            raise ValueError("Vector store environment variables (VSTORE_DIR, COLLECTION_NAME, EMBEDDING_MODEL) are not set in settings.")

        print(f"Initializing vector store from: {settings.VSTORE_DIR}")
        
        # 2. Initialize the embedding model
        # This model MUST match the one used to *create* the database
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        
        # 3. Connect to the existing persistent database
        # This does NOT build a new DB. It loads the existing one.
        _vector_store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.VSTORE_DIR
        )
        
        print(f"Successfully connected to persistent vector store. Collection: '{settings.COLLECTION_NAME}'")
        return _vector_store
    
    except Exception as e:
        # This will show up in your server logs if connection fails
        print(f"[Vector Store ERROR] Could not initialize vector store: {e}")
        # Return None so the tool can gracefully tell the user it failed
        return None

@tool
async def query_document_vector_store(query: str, k: int = 5) -> str:
    """
    Searches the full-text document vector store (ChromaDB) for detailed context,
    definitions, and specific guidelines. Use this to find the "fine print"
    or to get more detail on a topic.

    **WHEN TO USE:**
    1.  **After** using other tools (like `find_eligibility_rules` or `get_program_guidelines`),
        use their output to create a query for this tool to get the original,
        detailed text.
    2.  For general, open-ended questions about policies, definitions, or topics
        that don't fit a structured query (e.g., "What is the policy on gift funds?",
        "Explain 'declining market' policies").

    Args:
        query (str): The specific question or search term to find in the documents.
                     (e.g., "DSCR Plus detailed LTV guidelines", "ARC Home policy on first-time investors").
        k (int): The number of document chunks to return. Defaults to 5.
    """
    try:
        # 1. Get the persistent vector store connection
        vstore = _get_vector_store()
        
        # 2. Handle connection failure
        if vstore is None:
            return "Error: The document vector store is not available or failed to initialize. Please check server logs."

        # 3. Get the retriever and search for documents
        retriever = vstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        if not docs:
            return f"No detailed documents found matching the query: '{query}'"

        # 4. Format the output
        result_str = f"Found {len(docs)} relevant document chunks for '{query}':\n"
        for i, doc in enumerate(docs, 1):
            # Re-creating the metadata logic from your build_pdf_retriever
            source = doc.metadata.get("sourcePath", doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            
            result_str += f"\n**--- Chunk {i} (Source: {source}, Page: {page}) ---**\n"
            result_str += doc.page_content + "\n"

        return result_str

    except Exception as e:
        print(f"[query_document_vector_store ERROR] {e}")
        return f"Error querying vector store: {e}"