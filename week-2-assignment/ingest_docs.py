"""Ingest the Northwind sample documents through POST /ingest.

The API must already be running:

    uvicorn main:app --reload
    python ingest_docs.py                 # default directory, see DOCS_DIR
    python ingest_docs.py <directory>     # somewhere else
    python ingest_docs.py <directory> --fresh   # empty the index first

Each file's document_id comes from its own "Document ID:" header rather than
its filename, so re-running this overwrites the same chunks instead of writing
a second copy under a different name.
"""

import re
import sys
import time
from pathlib import Path

import httpx

import rag

API_BASE = "http://127.0.0.1:8000"
DOCS_DIR = Path.home() / "ai-eng-bootcamp-resources" / "RAG" / "northwind-sample-docs"

# The header line every Northwind document carries, e.g. "Document ID: POL-101".
DOCUMENT_ID = re.compile(r"^Document ID:\s*(\S+)\s*$", re.MULTILINE)
# Far enough in to cover the header block, short enough not to match body text.
HEADER_CHARS = 1000


def document_id_for(path: Path, text: str) -> str:
    """
    The id a document keeps across re-runs. Taken from the document itself, so
    renaming the file does not silently create a second copy in the index.

    Raises rather than falling back to the filename: a fallback here writes
    chunks under an id nobody expects and nothing complains until a citation
    reads wrong.
    """
    match = DOCUMENT_ID.search(text[:HEADER_CHARS])
    if match is None:
        raise SystemExit(f"{path.name}: no 'Document ID:' header in the first {HEADER_CHARS} characters")
    return match.group(1)


def stored_vector_count(expected: int, attempts: int = 10) -> int:
    """
    Read the vector count back from the store.

    Polled, because Pinecone's stats are eventually consistent -- reading
    immediately after a write usually reports the count from before it.
    """
    for attempt in range(attempts):
        count = rag._index_handle().describe_index_stats().total_vector_count
        if count == expected:
            return count
        time.sleep(2)
    return count


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--fresh"]
    directory = Path(args[0]) if args else DOCS_DIR
    paths = sorted(directory.glob("doc*.txt"))
    if not paths:
        raise SystemExit(f"no doc*.txt files in {directory}")

    if "--fresh" in sys.argv[1:]:
        # Everything, not just this run's ids: a document whose id changed since
        # last time would otherwise linger under its old name.
        rag._index_handle().delete(delete_all=True)
        print(f"emptied {rag.INDEX_NAME}\n")

    print(f"{'document_id':12} {'file':30} {'chunks':>6}")
    print("-" * 50)
    total = 0
    with httpx.Client(base_url=API_BASE, timeout=120.0) as client:
        for path in paths:
            text = path.read_text(encoding="utf-8")
            document_id = document_id_for(path, text)
            response = client.post(
                "/ingest",
                json={
                    "document_id": document_id,
                    "text": text,
                    "metadata": {"source": path.name},
                },
            )
            if response.status_code != 200:
                print(f"{document_id:12} {path.name:30} {'FAILED':>6}  {response.status_code} {response.text[:100]}")
                continue
            chunks = response.json()["chunks_indexed"]
            total += chunks
            print(f"{document_id:12} {path.name:30} {chunks:>6}")

    print("-" * 50)
    print(f"{'':12} {'indexed this run':30} {total:>6}")
    print(f"{'':12} {'in the vector store':30} {stored_vector_count(total):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
