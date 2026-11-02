FROM python:3.13-slim

WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# O modelo de embeddings entra na imagem: o contêiner sobe sem depender de download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY rag rag
COPY documentos documentos
COPY supabase supabase

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
