import sys
import shutil
from pathlib import Path
from rich import print  # For pretty printing
from rich.progress import track  # For a nice progress bar

# --- Setup Project Root ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

PDF_DIR = PROJECT_ROOT / "pdf"  # As you specified, 'pdf' in root
# ---------------------------

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from config.settings import settings
except ImportError as e:
    print(f"[bold red]Error:[/bold red] Failed to import necessary modules.")
    print("Please make sure you have all requirements installed: [code]pip install -r requirements.txt[/code]")
    print(f"Details: {e}")
    sys.exit(1)

def build_vector_store():
    """
    Builds a new persistent Chroma vector store from all PDFs in the PDF_DIR.
    """
    # --- 1. Get Settings ---
    VSTORE_DIR = Path(settings.VSTORE_DIR)
    COLLECTION_NAME = settings.COLLECTION_NAME
    EMBEDDING_MODEL = settings.EMBEDDING_MODEL
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 150

    print(f"[bold]Starting Vector Store Build[/bold]")
    print(f"  Target Directory: [cyan]{VSTORE_DIR}[/cyan]")
    print(f"  Collection Name:  [cyan]{COLLECTION_NAME}[/cyan]")
    print(f"  PDF Source:       [cyan]{PDF_DIR}[/cyan]\n")

    # --- 2. Clean Existing Vector Store ---
    if VSTORE_DIR.exists():
        print(f"[yellow]Warning:[/yellow] Existing vector store found. Deleting '[cyan]{VSTORE_DIR}[/cyan]' for a fresh build.")
        try:
            shutil.rmtree(VSTORE_DIR)
        except OSError as e:
            print(f"[bold red]Error:[/bold red] Could not delete directory '{VSTORE_DIR}'. Is it in use?")
            print(f"Details: {e}")
            sys.exit(1)
            
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    print("Old directory cleared. Starting file processing...")

    # --- 3. Find and Process PDFs ---
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[bold red]Error:[/bold red] No PDF files found in '{PDF_DIR}'. Aborting.")
        sys.exit(1)

    print(f"Found [magenta]{len(pdf_files)}[/magenta] PDF files to process.")

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Use `track` for a progress bar
    for pdf_path in track(pdf_files, description="Processing PDFs..."):
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # Split the document
            chunks = text_splitter.split_documents(pages)
            
            # Add the source filename to metadata for each chunk
            for chunk in chunks:
                chunk.metadata["source"] = pdf_path.name
            
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"\n[bold yellow]Warning:[/bold yellow] Failed to process '{pdf_path.name}'. Skipping file.")
            print(f"  Error: {e}")

    if not all_chunks:
        print("[bold red]Error:[/bold red] No documents were successfully processed. Vector store will not be built.")
        sys.exit(1)

    print(f"\nSuccessfully created [magenta]{len(all_chunks)}[/magenta] total text chunks.")

    # --- 4. Load Embedding Model ---
    print("\nLoading embedding model ([cyan]{EMBEDDING_MODEL}[/cyan])... (This may take a moment)")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to load embedding model.")
        print("Check your internet connection and the model name in settings.py.")
        print(f"Details: {e}")
        sys.exit(1)
    print("Embedding model loaded.")

    # --- 5. Create and Persist Vector Store ---
    print("Initializing Chroma vector store and adding documents... (This is the final step and may take time)")
    try:
        db = Chroma.from_documents(
            all_chunks,
            embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(VSTORE_DIR)
        )
        
        # Note: `from_documents` with `persist_directory` handles saving.
        # No explicit `db.persist()` is needed.
        
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to create Chroma database.")
        print(f"Details: {e}")
        sys.exit(1)

    print("\n[bold green]SUCCESS![/bold green]")
    print(f"Vector store has been created and persisted at '[cyan]{VSTORE_DIR}[/cyan]'.")
    print(f"Total chunks added: [magenta]{len(all_chunks)}[/magenta]")
    print("\nYou can now run [code]python test_vstore.py[/code] to verify the new database.")

if __name__ == "__main__":
    build_vector_store()