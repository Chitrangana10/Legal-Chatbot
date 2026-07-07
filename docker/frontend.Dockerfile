FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY frontend ./frontend

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

