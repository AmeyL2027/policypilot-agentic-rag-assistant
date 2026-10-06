import csv
import fitz  # PyMuPDF
from pathlib import Path

from app.paths import PROCESSED_DIR, SYNTHETIC_DOCS_DIR, resolve_public_docs_dir

PUBLIC_DIR = resolve_public_docs_dir()
SYNTHETIC_DIR = SYNTHETIC_DOCS_DIR
OUTPUT_CSV = PROCESSED_DIR / "documents.csv"


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    text = "\n".join(pages).strip()
    return text, len(doc)


def extract_markdown_text(md_path: Path) -> tuple[str, int]:
    text = md_path.read_text(encoding="utf-8").strip()
    return text, 1


def collect_documents():
    rows = []

    for pdf_file in PUBLIC_DIR.glob("*.pdf"):
        text, page_count = extract_pdf_text(pdf_file)
        rows.append(
            {
                "source_type": "public",
                "file_name": pdf_file.name,
                "doc_type": "pdf",
                "page_count": page_count,
                "char_count": len(text),
                "text": text,
            }
        )

    for md_file in SYNTHETIC_DIR.glob("*.md"):
        text, page_count = extract_markdown_text(md_file)
        rows.append(
            {
                "source_type": "synthetic",
                "file_name": md_file.name,
                "doc_type": "markdown",
                "page_count": page_count,
                "char_count": len(text),
                "text": text,
            }
        )

    return rows


def save_to_csv(rows):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_type",
                "file_name",
                "doc_type",
                "page_count",
                "char_count",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = collect_documents()
    save_to_csv(rows)
    print(f"Saved {len(rows)} documents to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()