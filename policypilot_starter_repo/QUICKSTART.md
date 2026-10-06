# Quickstart notes

## Suggested free local stack
- Run Qdrant locally with Docker.
- Run Ollama locally for both chat and embeddings.
- Use FastAPI for the backend and Streamlit for the UI.
- Use SQLite for metadata and chat/session history.

## Suggested first implementation order
1. Read markdown and PDF files
2. Create chunks by heading/section
3. Embed with `nomic-embed-text`
4. Store in Qdrant
5. Retrieve + answer with citations
6. Add compare/extract/checklist tools
