import requests
from qdrant_client import QdrantClient
from app.paths import QDRANT_LOCAL_DIR

QDRANT_PATH = str(QDRANT_LOCAL_DIR)
COLLECTION_NAME = "policypilot_chunks"


def get_embedding(text: str):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def search(query: str, limit: int = 5):
    client = QdrantClient(path=QDRANT_PATH)
    query_vector = get_embedding(query)

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    print(f"\nQuery: {query}\n")
    for i, hit in enumerate(result.points, start=1):
        payload = hit.payload or {}
        print("=" * 80)
        print(f"Rank: {i}")
        print(f"Score: {hit.score:.4f}")
        print(f"File: {payload.get('file_name')}")
        print(f"Source Type: {payload.get('source_type')}")
        print(f"Chunk Index: {payload.get('chunk_index')}")
        print("-" * 80)
        print((payload.get("text") or "")[:700])
        print()


def main():
    query = "What is the travel reimbursement policy?"
    search(query=query, limit=5)


if __name__ == "__main__":
    main()