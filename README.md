# Mortgage Chatbot Project

This repository is a starter scaffold for a modular FastAPI application that uses:
- FastAPI
- SQLAlchemy (async)
- PostgreSQL (asyncpg)
- Alembic for migrations
- LangChain + Groq LLM integration


### Clone the Project:

```
# Replace with the actual repository URL
git clone <repository_url>
cd mortgage_chatbot_project
```

### Create and Activate a Python Virtual Environment:
```
python3 -m venv venv
source venv/bin/activate
```
(Note: You must run source venv/bin/activate every time you open a new terminal session to work on this project.)

### Install Python Requirements:

```
pip install -r requirements.txt
```

## Configure the Database and Environment
### Create the PostgreSQL Database:
```
createdb mortgage_db
```
### Create Your .env File: 
Copy the example file to create your local configuration.

```
cp .env.example .env
```

### Edit Your .env File: 
Open the new .env file with a text editor.

- DATABASE_URL: Update this to match your local PostgreSQL setup. (Replace your_mac_username with your actual Mac username).

```
# Standard Homebrew setup with no password
DATABASE_URL="postgresql+asyncpg://your_mac_username@localhost:5432/mortgage_db"
```

- GROQ_API_KEY: Get an API key from Groq and paste it here. The app will not work without it.

```
GROQ_API_KEY="paste_your_groq_api_key_here"
PORT: You can leave this as 8085.
```

## Set Up and Populate the Databases
This project uses two databases: a PostgreSQL database for structured data and a Chroma vector database for PDF guideline text.

#### 1. Run Alembic Migrations (SQL DB): This command creates all the necessary tables in your PostgreSQL database.

```
alembic upgrade head
```
#### 2. Import SQL Data (PostgreSQL): This script loads the contents of db/data.json into the tables you just created.

```
python db.import_data.py
```
You should see a "✅ Data successfully imported..." message.

#### 3. Ingest Vector Data (Chroma DB):

- Add PDF Files: Create a new folder named pdf in the project's root directory. Place all your mortgage guideline PDF documents into this folder.

- Run the Ingestion Script: This script processes the PDFs in the pdf folder and builds the local vector store.

```
python ingest_data.py
```
This may take a few minutes. You should see a "✅ Vector store has been created..." message when it's done.

#### 4. Run the Server
You are now ready to run the application.

```
uvicorn main:app --reload --host 0.0.0.0 --port 8100
```
Your server is now running. You can access:

- The Application: http://localhost:8085

- API Docs (Swagger UI): http://localhost:8085/docs
