import csv
import re
from collections import defaultdict

import requests
import streamlit as st
from qdrant_client import QdrantClient
from app.paths import PROCESSED_DIR, QDRANT_LOCAL_DIR

# -----------------------------------------------------------------------------
# App / model configuration
# -----------------------------------------------------------------------------

QDRANT_PATH = str(QDRANT_LOCAL_DIR)
CHUNKS_CSV = PROCESSED_DIR / "chunks.csv"
COLLECTION_NAME = "policypilot_chunks"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:1.5b"

# -----------------------------------------------------------------------------
# Ollama + Qdrant helpers
# -----------------------------------------------------------------------------


def get_embedding(text: str):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def retrieve_chunks(query: str, limit: int = 8):
    client = QdrantClient(path=QDRANT_PATH)
    query_vector = get_embedding(query)

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    return result.points


def generate_generic_answer(query: str, context: str):
    prompt = f"""
You are a policy assistant answering questions using only the provided context.

Rules:
- Answer only from the context below.
- Do not invent facts.
- Quote exact values when possible.
- If the answer is not clearly supported, say: "I could not find enough evidence in the provided documents."
- Keep the answer concise and clear.
- End with a short "Sources" section listing the source files used.

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


# -----------------------------------------------------------------------------
# Query classification / routing
# -----------------------------------------------------------------------------


def classify_query(query: str) -> str:
    q = query.lower()

    if any(word in q for word in ["compare", "difference", "different", "changed", "change"]):
        return "compare"

    if any(word in q for word in ["latest", "current", "newest"]):
        return "latest"

    return "general"


def is_travel_policy_query(query: str) -> bool:
    q = query.lower()
    return "travel" in q and "policy" in q


def extract_version_from_filename(file_name: str) -> int:
    match = re.search(r"_V(\d+)", file_name.upper())
    if match:
        return int(match.group(1))
    return 0


# -----------------------------------------------------------------------------
# CSV loading for deterministic travel-policy handling
# -----------------------------------------------------------------------------


def load_chunk_rows():
    rows = []
    with CHUNKS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_travel_policy_rows():
    rows = load_chunk_rows()
    travel_rows = [
        row for row in rows
        if row["file_name"].upper().startswith("TRAVEL_EXPENSE_POLICY_V")
    ]
    return sorted(
        travel_rows,
        key=lambda r: (
            extract_version_from_filename(r["file_name"]),
            int(r.get("chunk_index", 0)),
        ),
    )


def group_rows_by_file(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["file_name"]].append(row)

    for file_name in grouped:
        grouped[file_name] = sorted(
            grouped[file_name],
            key=lambda r: int(r.get("chunk_index", 0)),
        )

    return grouped


def combine_rows_text(rows):
    return "\n\n".join(row.get("text", "") for row in rows)


# -----------------------------------------------------------------------------
# Deterministic parsing for Travel Expense Policy
# -----------------------------------------------------------------------------


def get_section(text: str, section_title: str) -> str:
    pattern = rf"##\s*\d+\.\s*{re.escape(section_title)}\s*(.*?)(?=##\s*\d+\.|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return ""


def get_sentence_containing(text: str, phrase: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    for sentence in sentences:
        if phrase.lower() in sentence.lower():
            return sentence.strip()
    return ""


def parse_currency_value(text: str, label: str):
    patterns = [
        rf"{label}\s*[:\-]?\s*(\d+)\s*CAD",
        rf"(\d+)\s*CAD\s*for\s*{label}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def parse_travel_policy(text: str, file_name: str):
    cleaned = " ".join(text.split())

    metadata_version = None
    version_match = re.search(r"version:\s*([0-9.]+)", cleaned, flags=re.IGNORECASE)
    if version_match:
        metadata_version = version_match.group(1)

    effective_date = None
    date_match = re.search(r"effective_date:\s*([0-9\-]+)", cleaned, flags=re.IGNORECASE)
    if date_match:
        effective_date = date_match.group(1)

    air_travel_section = get_section(cleaned, "Air Travel")
    hotel_section = get_section(cleaned, "Hotels")
    meals_section = get_section(cleaned, "Meals")
    ground_transport_section = get_section(cleaned, "Ground Transport")
    receipt_rules_section = get_section(cleaned, "Receipt Rules")
    submission_timeline_section = get_section(cleaned, "Submission Timeline")

    premium_hours = None
    premium_match = re.search(
        r"premium economy.*?longer than\s+(\d+)\s+hours",
        air_travel_section,
        flags=re.IGNORECASE,
    )
    if premium_match:
        premium_hours = int(premium_match.group(1))

    business_class_rule = get_sentence_containing(air_travel_section, "Business class")
    if not business_class_rule:
        business_class_rule = get_sentence_containing(cleaned, "Business class")

    hotel_cap = None
    hotel_match = re.search(
        r"hotel costs up to\s+(\d+)\s*CAD",
        hotel_section,
        flags=re.IGNORECASE,
    )
    if hotel_match:
        hotel_cap = int(hotel_match.group(1))

    breakfast_cap = parse_currency_value(meals_section, "breakfast")
    lunch_cap = parse_currency_value(meals_section, "lunch")
    dinner_cap = parse_currency_value(meals_section, "dinner")

    receipt_threshold = None
    receipt_match = re.search(
        r"above\s+(\d+)\s*CAD",
        receipt_rules_section,
        flags=re.IGNORECASE,
    )
    if receipt_match:
        receipt_threshold = int(receipt_match.group(1))

    submission_days = None
    submission_match = re.search(
        r"within\s+(\d+)\s+calendar days",
        submission_timeline_section,
        flags=re.IGNORECASE,
    )
    if submission_match:
        submission_days = int(submission_match.group(1))

    return {
        "file_name": file_name,
        "version_number": extract_version_from_filename(file_name),
        "version_label": metadata_version or f"{extract_version_from_filename(file_name)}.0",
        "effective_date": effective_date,
        "premium_hours": premium_hours,
        "business_class_rule": business_class_rule,
        "hotel_cap": hotel_cap,
        "breakfast_cap": breakfast_cap,
        "lunch_cap": lunch_cap,
        "dinner_cap": dinner_cap,
        "ground_transport": ground_transport_section,
        "receipt_threshold": receipt_threshold,
        "submission_days": submission_days,
    }


def get_travel_policy_docs():
    rows = get_travel_policy_rows()
    grouped = group_rows_by_file(rows)

    docs = {}
    for file_name, file_rows in grouped.items():
        combined_text = combine_rows_text(file_rows)
        docs[file_name] = {
            "rows": file_rows,
            "text": combined_text,
            "parsed": parse_travel_policy(combined_text, file_name),
        }

    return docs


# -----------------------------------------------------------------------------
# Deterministic answer formatting for travel policy queries
# -----------------------------------------------------------------------------


def format_business_class(rule: str) -> str:
    return rule if rule else "Business class rule is not clearly stated."


def format_ground_transport(section: str) -> str:
    return section if section else "Ground transport rules are not clearly stated."


def format_meal_caps(parsed: dict) -> str:
    if all(
        value is not None
        for value in [parsed["breakfast_cap"], parsed["lunch_cap"], parsed["dinner_cap"]]
    ):
        return (
            f"breakfast {parsed['breakfast_cap']} CAD, "
            f"lunch {parsed['lunch_cap']} CAD, "
            f"dinner {parsed['dinner_cap']} CAD"
        )
    return "Meal caps are not clearly stated."


def format_general_travel_answer(docs):
    ordered_files = sorted(
        docs.keys(),
        key=lambda f: docs[f]["parsed"]["version_number"]
    )

    lines = []
    lines.append("Multiple versions of the travel reimbursement policy were found.")
    lines.append("")

    for file_name in ordered_files:
        parsed = docs[file_name]["parsed"]
        lines.append(f"### Policy v{parsed['version_number']}")
        if parsed["effective_date"]:
            lines.append(f"- Effective date: {parsed['effective_date']}")
        lines.append("- Air Travel:")
        lines.append("  - Economy airfare is reimbursable.")
        if parsed["premium_hours"] is not None:
            lines.append(
                f"  - Premium economy requires director approval for flights longer than {parsed['premium_hours']} hours."
            )
        lines.append(f"  - {format_business_class(parsed['business_class_rule'])}")
        if parsed["hotel_cap"] is not None:
            lines.append(f"- Hotel cap: {parsed['hotel_cap']} CAD per night before taxes in major Canadian cities.")
        lines.append(f"- Meal caps: {format_meal_caps(parsed)}.")
        lines.append(f"- Ground transport: {format_ground_transport(parsed['ground_transport'])}")
        if parsed["receipt_threshold"] is not None:
            lines.append(f"- Receipt rule: receipts required for expenses above {parsed['receipt_threshold']} CAD.")
        if parsed["submission_days"] is not None:
            lines.append(f"- Submission timeline: claims must be submitted within {parsed['submission_days']} calendar days.")
        lines.append("")

    lines.append("### Sources")
    for file_name in ordered_files:
        lines.append(f"- {file_name}")

    return "\n".join(lines)


def format_latest_travel_answer(docs):
    latest_file = max(
        docs.keys(),
        key=lambda f: docs[f]["parsed"]["version_number"]
    )
    parsed = docs[latest_file]["parsed"]

    lines = []
    lines.append(f"The latest travel reimbursement policy is **Policy v{parsed['version_number']}**.")
    if parsed["effective_date"]:
        lines.append(f"- Effective date: {parsed['effective_date']}")
    lines.append("- Air Travel:")
    lines.append("  - Economy airfare is reimbursable.")
    if parsed["premium_hours"] is not None:
        lines.append(
            f"  - Premium economy requires director approval for flights longer than {parsed['premium_hours']} hours."
        )
    lines.append(f"  - {format_business_class(parsed['business_class_rule'])}")
    if parsed["hotel_cap"] is not None:
        lines.append(f"- Hotel cap: {parsed['hotel_cap']} CAD per night before taxes in major Canadian cities.")
    lines.append(f"- Meal caps: {format_meal_caps(parsed)}.")
    lines.append(f"- Ground transport: {format_ground_transport(parsed['ground_transport'])}")
    if parsed["receipt_threshold"] is not None:
        lines.append(f"- Receipt rule: receipts required for expenses above {parsed['receipt_threshold']} CAD.")
    if parsed["submission_days"] is not None:
        lines.append(f"- Submission timeline: claims must be submitted within {parsed['submission_days']} calendar days.")
    lines.append("")
    lines.append("### Sources")
    lines.append(f"- {latest_file}")

    return "\n".join(lines)


def format_compare_travel_answer(docs):
    ordered_files = sorted(
        docs.keys(),
        key=lambda f: docs[f]["parsed"]["version_number"]
    )

    if len(ordered_files) < 2:
        return "I could not find enough relevant policy versions to compare."

    first = docs[ordered_files[0]]["parsed"]
    second = docs[ordered_files[1]]["parsed"]

    lines = []
    lines.append(f"### Changes between Policy v{first['version_number']} and Policy v{second['version_number']}")
    lines.append("")

    lines.append("- **Air Travel**")
    if first["premium_hours"] != second["premium_hours"]:
        lines.append(
            f"  - Premium economy approval threshold changed from flights longer than **{first['premium_hours']} hours** to flights longer than **{second['premium_hours']} hours**."
        )
    else:
        lines.append("  - Premium economy approval threshold did not change.")
    if first["business_class_rule"] != second["business_class_rule"]:
        lines.append("  - Business class rule changed.")
        lines.append(f"  - v{first['version_number']}: {format_business_class(first['business_class_rule'])}")
        lines.append(f"  - v{second['version_number']}: {format_business_class(second['business_class_rule'])}")
    else:
        lines.append("  - Business class rule did not change.")

    lines.append("- **Hotel Cap**")
    if first["hotel_cap"] != second["hotel_cap"]:
        lines.append(
            f"  - Hotel reimbursement cap changed from **{first['hotel_cap']} CAD** to **{second['hotel_cap']} CAD** per night before taxes."
        )
    else:
        lines.append(f"  - Hotel reimbursement cap stayed the same at **{first['hotel_cap']} CAD**.")

    lines.append("- **Meal Caps**")
    if first["breakfast_cap"] != second["breakfast_cap"]:
        lines.append(f"  - Breakfast changed from **{first['breakfast_cap']} CAD** to **{second['breakfast_cap']} CAD**.")
    else:
        lines.append(f"  - Breakfast stayed the same at **{first['breakfast_cap']} CAD**.")

    if first["lunch_cap"] != second["lunch_cap"]:
        lines.append(f"  - Lunch changed from **{first['lunch_cap']} CAD** to **{second['lunch_cap']} CAD**.")
    else:
        lines.append(f"  - Lunch stayed the same at **{first['lunch_cap']} CAD**.")

    if first["dinner_cap"] != second["dinner_cap"]:
        lines.append(f"  - Dinner changed from **{first['dinner_cap']} CAD** to **{second['dinner_cap']} CAD**.")
    else:
        lines.append(f"  - Dinner stayed the same at **{first['dinner_cap']} CAD**.")

    lines.append("- **Ground Transport**")
    if first["ground_transport"] != second["ground_transport"]:
        lines.append("  - Ground transport coverage changed.")
        lines.append(f"  - v{first['version_number']}: {format_ground_transport(first['ground_transport'])}")
        lines.append(f"  - v{second['version_number']}: {format_ground_transport(second['ground_transport'])}")
    else:
        lines.append("  - Ground transport coverage did not materially change.")

    lines.append("- **Receipt Rules**")
    if first["receipt_threshold"] != second["receipt_threshold"]:
        lines.append(
            f"  - Receipt threshold changed from **{first['receipt_threshold']} CAD** to **{second['receipt_threshold']} CAD**."
        )
    else:
        lines.append(f"  - Receipt threshold stayed the same at **{first['receipt_threshold']} CAD**.")

    lines.append("- **Submission Timeline**")
    if first["submission_days"] != second["submission_days"]:
        lines.append(
            f"  - Submission deadline changed from **{first['submission_days']} days** to **{second['submission_days']} days** after travel ends."
        )
    else:
        lines.append(f"  - Submission timeline stayed the same at **{first['submission_days']} days**.")

    lines.append("")
    lines.append("### Sources")
    lines.append(f"- {ordered_files[0]}")
    lines.append(f"- {ordered_files[1]}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Generic semantic-retrieval rendering for non-travel queries
# -----------------------------------------------------------------------------


def has_relevant_results(points, min_score: float = 0.62) -> bool:
    if not points:
        return False
    return points[0].score >= min_score


def build_context_from_points(points):
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


def render_point_sources(points):
    st.subheader("Retrieved Sources")

    for i, point in enumerate(points, start=1):
        payload = point.payload or {}
        with st.expander(
            f"{i}. {payload.get('file_name')} | chunk {payload.get('chunk_index')} | score={point.score:.4f}"
        ):
            st.write(payload.get("text", ""))


def render_row_sources(rows):
    st.subheader("Retrieved Sources")

    for i, row in enumerate(rows, start=1):
        with st.expander(
            f"{i}. {row.get('file_name')} | chunk {row.get('chunk_index')}"
        ):
            st.write(row.get("text", ""))


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="PolicyPilot", page_icon="📄", layout="wide")

st.title("PolicyPilot")
st.write("A local RAG assistant for querying policy documents with source-grounded answers.")

query = st.text_input(
    "Enter your question:",
    "What is the travel reimbursement policy?"
)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            mode = classify_query(query)

            if is_travel_policy_query(query):
                docs = get_travel_policy_docs()

                if not docs:
                    st.warning("I could not find relevant travel policy documents.")
                else:
                    if mode == "compare":
                        answer = format_compare_travel_answer(docs)
                    elif mode == "latest":
                        answer = format_latest_travel_answer(docs)
                    else:
                        answer = format_general_travel_answer(docs)

                    st.subheader("Answer")
                    st.markdown(answer)

                    if mode == "latest":
                        latest_file = max(
                            docs.keys(),
                            key=lambda f: docs[f]["parsed"]["version_number"]
                        )
                        render_row_sources(docs[latest_file]["rows"])
                    else:
                        all_rows = []
                        for file_name in sorted(
                            docs.keys(),
                            key=lambda f: docs[f]["parsed"]["version_number"]
                        ):
                            all_rows.extend(docs[file_name]["rows"])
                        render_row_sources(all_rows)

            else:
                points = retrieve_chunks(query, limit=8)
                points = points[:4]

                if not has_relevant_results(points):
                    st.warning("I could not find strong enough evidence in the indexed documents.")
                else:
                    context = build_context_from_points(points)
                    answer = generate_generic_answer(query, context)

                    st.subheader("Answer")
                    st.write(answer)
                    render_point_sources(points)