# Intelligent Legal RAG System

An AI-powered legal retrieval and guidance system for Indian law, built using retrieval-augmented generation (RAG). Currently covers the Indian Penal Code, 1860 and the Code of Criminal Procedure, 1973.

## Setup

1. From the project root, install dependencies:

pip install fastapi uvicorn pydantic pydantic-settings python-dotenv sentence-transformers faiss-cpu rank-bm25 google-generativeai streamlit requests beautifulsoup4 pytest

2. Create your .env file at the project root:

cp .env.example .env

3. Add your Gemini API key to .env:

GEMINI_API_KEY=your_actual_key_here

4. Confirm .env is listed in .gitignore.

## Build the data index

python backend/scripts/build_combined_index.py

## Run the app

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
- If you see "No module named uvicorn," you're likely inside an unrelated virtual environment — this project runs on plain system Python, no venv needed.
