# Intelligent Legal RAG System

An AI-powered legal retrieval and guidance system for Indian law, built using retrieval-augmented generation (RAG). Currently covers the Indian Penal Code, 1860 and the Code of Criminal Procedure, 1973.

## Setup

1. From the project root, create and activate a virtual environment:

python -m venv .venv

Activate it (Windows Git Bash):

source .venv/Scripts/activate

Activate it (Windows Command Prompt / PowerShell):

.venv\Scripts\activate

You should see (.venv) appear at the start of your terminal prompt once it's active. Run all the following commands with this environment activated.

2. Install dependencies:

pip install fastapi uvicorn pydantic pydantic-settings python-dotenv sentence-transformers faiss-cpu rank-bm25 google-generativeai streamlit requests beautifulsoup4 pytest

3. Create your .env file at the project root:

cp .env.example .env

4. Add your Gemini API key to .env:

GEMINI_API_KEY=your_actual_key_here

5. Confirm .env is listed in .gitignore.

## Build the data index

python backend/scripts/build_combined_index.py

## Run the app

Make sure your .venv is activated in both terminals below.

Check nothing is already running on port 8000:

netstat -ano | findstr :8000

If something shows up, kill it:

taskkill //PID <the_number_shown> //F

Terminal 1 - backend:

python -m uvicorn backend.app.main:app --reload

Terminal 2 - frontend:

python -m streamlit run frontend/streamlit_app.py

Opens at http://localhost:8501

## Quick test questions

- What is the punishment for murder? -> should cite IPC Section 302
- What is the procedure for arrest without a warrant? -> should cite CrPC sections
- What is the GST rate on restaurant food? -> should refuse, not answer

## Notes

- Gemini free tier is capped at 20 requests/day. On quota errors, the app falls back to listing retrieved sources instead of a generated answer.
- If the backend shows old/stale data after a change, check for a leftover process on port 8000 and kill it, then restart uvicorn.
- If commands fail with "No module named X," check that your .venv is actually activated - you should see (.venv) in your terminal prompt. If not, activate it again using the command in step 1.
- .venv/ and .pytest_cache/ should both be listed in .gitignore - never commit these.