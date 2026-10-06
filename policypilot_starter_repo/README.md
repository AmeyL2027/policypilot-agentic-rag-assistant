# PolicyPilot

PolicyPilot is a local RAG assistant for querying policy documents with source-grounded answers.

## Stack
- **Embedding model:** `nomic-embed-text` via Ollama
- **Chat model:** `qwen2.5:1.5b` via Ollama
- **Vector database:** local Qdrant (`data/qdrant_local/`)
- **UI:** Streamlit

## Clean project layout
```text
app/
  ask_rag.py
  chunk_docs.py
  index_qdrant.py
  ingest_docs.py
  search_qdrant.py
  paths.py
ui/
  streamlit_app.py
data/
  public_docs/              # place downloaded public PDFs here
  public_doc_sources/       # links to recommended public PDF sources
  synthetic_docs/           # synthetic markdown policy docs
  processed/                # generated CSV artifacts (documents/chunks)
  qdrant_local/             # local vector index data
```

## Setup
1. Create and activate your venv.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Pull Ollama models:
   - `ollama pull nomic-embed-text`
   - `ollama pull qwen2.5:1.5b`

## Data prep
1. Keep synthetic docs in `data/synthetic_docs/`.
2. Download 6-8 public PDFs using:
   - `data/public_doc_sources/PUBLIC_DOCUMENT_SOURCES.md`
3. Put those PDFs in:
   - `data/public_docs/`

## Run pipeline
1. `python app/ingest_docs.py`
2. `python app/chunk_docs.py`
3. `python app/index_qdrant.py`
4. `python app/search_qdrant.py` (optional CLI check)
5. `python app/ask_rag.py` (optional CLI answer check)
6. `streamlit run ui/streamlit_app.py`
