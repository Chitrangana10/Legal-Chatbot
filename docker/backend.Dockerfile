FROM python:3.11-slim

WORKDIR /app

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Pre-download the embedding model at build time so the container doesn't
# hit the network on every cold start. Without this, RAGEngine's startup
# (inside FastAPI's lifespan, which blocks all requests including /health
# until it finishes) has to download the model from the internet before it
# can serve anything - this is what was causing the CI smoke test to time
# out and fail.
ARG EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL_NAME}')"

COPY backend/ ./backend/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
