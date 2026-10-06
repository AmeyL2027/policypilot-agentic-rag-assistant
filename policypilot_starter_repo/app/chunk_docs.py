import csv
import re
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.paths import PROCESSED_DIR

INPUT_CSV = PROCESSED_DIR / "documents.csv"
OUTPUT_CSV = PROCESSED_DIR / "chunks.csv"


# Increase CSV field size limit safely
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_documents():
    rows = []
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["text"] = clean_text(row["text"])
            rows.append(row)
    return rows


def chunk_documents(rows):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunk_rows = []
    chunk_id = 1

    for row in rows:
        chunks = splitter.split_text(row["text"])
        for idx, chunk in enumerate(chunks, start=1):
            chunk_rows.append(
                {
                    "chunk_id": f"chunk_{chunk_id}",
                    "source_type": row["source_type"],
                    "file_name": row["file_name"],
                    "doc_type": row["doc_type"],
                    "page_count": row["page_count"],
                    "char_count": len(chunk),
                    "chunk_index": idx,
                    "text": chunk,
                }
            )
            chunk_id += 1

    return chunk_rows


def save_chunks(rows):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_id",
                "source_type",
                "file_name",
                "doc_type",
                "page_count",
                "char_count",
                "chunk_index",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    docs = load_documents()
    chunks = chunk_documents(docs)
    save_chunks(chunks)
    print(f"Saved {len(chunks)} chunks to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()