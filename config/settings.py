from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    GROQ_API_KEY: str

    # Allow extra keys in the .env (like PROJECT_NAME, PORT) so Alembic
    # and other tools can load the file without raising validation errors.
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


# Create a single instance to be imported by other modules
settings = Settings()
