import csv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
from app.paths import PROCESSED_DIR, QDRANT_LOCAL_DIR

CHUNKS_CSV = PROCESSED_DIR / "chunks.csv"

COLLECTION_NAME = "policypilot_chunks"
QDRANT_PATH = str(QDRANT_LOCAL_DIR)


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
    data = response.json()
    return data["embedding"]


def load_chunks():
    rows = []
    with CHUNKS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    rows = load_chunks()
    print(f"Loaded {len(rows)} chunks from {CHUNKS_CSV}")

    sample_embedding = get_embedding(rows[0]["text"])
    vector_size = len(sample_embedding)
    print(f"Embedding size: {vector_size}")

    client = QdrantClient(path=QDRANT_PATH)

    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = []
    for i, row in enumerate(rows, start=1):
        embedding = get_embedding(row["text"])
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "chunk_id": row["chunk_id"],
                    "source_type": row["source_type"],
                    "file_name": row["file_name"],
                    "doc_type": row["doc_type"],
                    "page_count": row["page_count"],
                    "char_count": row["char_count"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                },
            )
        )

        if i % 20 == 0:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"Indexed {i} chunks...")
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Finished indexing {len(rows)} chunks into Qdrant.")


if __name__ == "__main__":
    main()