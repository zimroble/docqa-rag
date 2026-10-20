FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml ./
COPY docqa ./docqa
RUN pip install --no-cache-dir . \
    && python -c "from docqa.embeddings import get_embedder; get_embedder()"  # fail the build if the model can't load

RUN useradd --create-home app
USER app

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn docqa.main:app --host 0.0.0.0 --port ${PORT}"]
