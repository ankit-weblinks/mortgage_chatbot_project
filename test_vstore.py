import sys
from pathlib import Path
from rich import print  # For pretty printing
from rich.panel import Panel

# --- Setup Project Root ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ---------------------------

try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from config.settings import settings
except ImportError as e:
    print(f"[bold red]Error:[/bold red] Failed to import necessary modules.")
    print("Please make sure you have all requirements installed: [code]pip install -r requirements.txt[/code]")
    print(f"Details: {e}")
    sys.exit(1)

def run_query(vstore: Chroma, query: str, k: int = 3):
    """Helper function to run and print a query."""
    print(f"\n[bold]Running query:[/bold] [yellow]'{query}'[/yellow] (k={k})")
    
    try:
        docs = vstore.similarity_search(query, k=k)

        if not docs:
            print("[bold yellow]Query ran successfully, but no matching documents were found.[/bold yellow]")
            return

        print(f"[bold green]Found {len(docs)} relevant document chunks:[/bold green]")
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("sourcePath", doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            
            content_panel = Panel(
                doc.page_content,
                title=f"[bold yellow]Result {i}[/bold yellow]",
                subtitle=f"[cyan]Source:[/cyan] {source} | [cyan]Page:[/cyan] {page}",
                border_style="blue"
            )
            print(content_panel)
            
    except Exception as e:
        print(f"[bold red]An error occurred during this query:[/bold red] {e}")


def test_vector_store():
    """
    Connects to the persistent ChromaDB, checks the document count,
    and runs test queries.
    """
    print(f"Attempting to connect to vector store at: [cyan]{settings.VSTORE_DIR}[/cyan]")
    print(f"Using collection: [cyan]{settings.COLLECTION_NAME}[/cyan]")
    print(f"Loading embedding model: [cyan]{settings.EMBEDDING_MODEL}[/cyan]\n")

    try:
        # 1. Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        
        # 2. Connect to the existing persistent database
        vstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.VSTORE_DIR
        )
        
        print("[bold green]Successfully connected to vector store.[/bold green]")

        # --- NEW DIAGNOSTIC STEP ---
        # 3. Get the total count of documents
        collection_count = vstore._collection.count()
        
        print(Panel(
            f"[bold]Total document chunks in collection:[/bold] [bright_magenta]{collection_count}[/bright_magenta]",
            title="Database Diagnostics",
            border_style="green"
        ))
        
        if collection_count == 0:
            print("[bold red]Error: The vector store is empty.[/bold red]")
            print("This is the problem. The database is connected, but no documents have been loaded.")
            print("You need to run your data ingestion script (like the one based on `build_pdf_retriever`) to add documents.")
            return
        # --- END DIAGNOSTIC STEP ---

        # 4. Run Queries
        
        # Test 1: Broad query
        BROAD_TEST_QUERY = "loan to value ratio"
        run_query(vstore, BROAD_TEST_QUERY, k=3)
        
        print("\n" + "-"*80 + "\n")
        
        # Test 2: Your specific query
        SPECIFIC_TEST_QUERY = "Flex Select program guidelines NQM FUNDING"
        run_query(vstore, SPECIFIC_TEST_QUERY, k=3)

    except Exception as e:
        print(f"[bold red]An error occurred during the test:[/bold red]")
        print(e)
        print("\n[bold yellow]Debug Tips:[/bold yellow]")
        print(f"1. Make sure the path '[code]{settings.VSTORE_DIR}[/code]' is correct.")
        print(f"2. Ensure the collection '[code]{settings.COLLECTION_NAME}[/code]' exists in that database.")

if __name__ == "__main__":
    test_vector_store()