from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

CROMA_DB_DIR = Path("./cromaDb")
class Settings(BaseSettings):
    DATABASE_URL: str
    GROQ_API_KEY: str

    VSTORE_DIR: str = str(CROMA_DB_DIR)
    COLLECTION_NAME: str = "loan_guidelines"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Allow extra keys in the .env (like PROJECT_NAME, PORT) so Alembic
    # and other tools can load the file without raising validation errors.
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


# Create a single instance to be imported by other modules
settings = Settings()
