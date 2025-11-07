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
    Initializes and returns a connection to the persistent Chroma vector store.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    try:
        # --- IMPORTANT ---
        # You need to add VSTORE_DIR, COLLECTION_NAME, and EMBEDDING_MODEL
        # to your `config/settings.py` file, based on the snippet you provided.
        #
        # Example for settings.py:
        # VSTORE_DIR = "./data/vectorstores/chroma_dbs" 
        # COLLECTION_NAME = "loan_guidelines"
        # EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        # ---------------------

        if not all([settings.VSTORE_DIR, settings.COLLECTION_NAME, settings.EMBEDDING_MODEL]):
            raise ValueError("Vector store environment variables (VSTORE_DIR, COLLECTION_NAME, EMBEDDING_MODEL) are not set in settings.")

        print(f"Initializing vector store from: {settings.VSTORE_DIR}")
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        
        _vector_store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.VSTORE_DIR
        )
        print("Vector store initialized successfully.")
        return _vector_store
    
    except Exception as e:
        print(f"[Vector Store ERROR] Could not initialize vector store: {e}")
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
        vstore = _get_vector_store()
        if vstore is None:
            return "Error: The document vector store is not available or failed to initialize."

        retriever = vstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        if not docs:
            return f"No detailed documents found matching the query: '{query}'"

        result_str = f"Found {len(docs)} relevant document chunks for '{query}':\n"
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("sourcePath", "Unknown")
            page = doc.metadata.get("page", "N/A")
            
            result_str += f"\n**--- Chunk {i} (Source: {source}, Page: {page}) ---**\n"
            result_str += doc.page_content + "\n"

        return result_str

    except Exception as e:
        print(f"[query_document_vector_store ERROR] {e}")
        return f"Error querying vector store: {e}"