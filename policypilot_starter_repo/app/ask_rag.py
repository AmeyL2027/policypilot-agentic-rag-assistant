import requests
from qdrant_client import QdrantClient
from app.paths import QDRANT_LOCAL_DIR

QDRANT_PATH = str(QDRANT_LOCAL_DIR)
COLLECTION_NAME = "policypilot_chunks"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:1.5b"


def get_embedding(text: str):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def retrieve_chunks(query: str, limit: int = 3):
    client = QdrantClient(path=QDRANT_PATH)
    query_vector = get_embedding(query)

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return result.points


def build_context(points):
    context_parts = []

    for i, point in enumerate(points, start=1):
        payload = point.payload or {}
        file_name = payload.get("file_name", "unknown")
        chunk_index = payload.get("chunk_index", "unknown")
        text = payload.get("text", "")

        context_parts.append(
            f"[Source {i}] File: {file_name} | Chunk: {chunk_index}\n{text}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query: str, context: str):
    prompt = f"""
You are a policy assistant answering questions using only the provided context.

Rules:
- Answer only from the context below.
- If the answer is not clearly supported, say: "I could not find enough evidence in the provided documents."
- Be concise but clear.
- At the end, include a short "Sources" section listing the source files you used.

Question:
{query}

Context:
{context}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    query = "What is the travel reimbursement policy?"

    points = retrieve_chunks(query=query, limit=3)
    context = build_context(points)
    answer = generate_answer(query=query, context=context)

    print("\n" + "=" * 100)
    print("QUESTION:")
    print(query)
    print("=" * 100)
    print("ANSWER:")
    print(answer)
    print("=" * 100)
    print("RETRIEVED SOURCES:")
    for i, point in enumerate(points, start=1):
        payload = point.payload or {}
        print(
            f"{i}. {payload.get('file_name')} | chunk {payload.get('chunk_index')} | score={point.score:.4f}"
        )


if __name__ == "__main__":
    main()