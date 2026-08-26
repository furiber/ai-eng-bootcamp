"""Pinecone vector store for the Week 2 assignment.

Owns embeddings and the vector store so main.py stays the HTTP layer. Every
setting comes from the environment (see .env.example); nothing here holds a
secret, and no client is constructed at import time, so importing this module
can never stop the app booting or /health answering.

Confirm Pinecone is reachable without spending anything at OpenAI:

    python rag.py
"""

import asyncio
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec

from env_setup import load_env

load_env()

# One model for both paths. Ingest and query MUST embed identically -- vectors
# from different models are not comparable, and the resulting search results
# are silently wrong rather than obviously broken.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Sent on every embedding call and checked against the live index by check().
# text-embedding-3-small is natively 1536; `dimensions` can truncate below that
# but never above it.
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# The SDK would read PINECONE_API_KEY by itself, but it is read explicitly here
# so a missing key is a clear message instead of an SDK-internal error.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ask-api-docs")
# Used only when creating the index. Querying an existing one ignores both.
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Chunking. Both are counted in CHARACTERS, not tokens -- the splitter's default
# length function is len(). Overlap carries the tail of one chunk into the next so
# a sentence split across a boundary is still retrievable from one of them.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Per-request caps from the embeddings API. Hitting these returns a 400 from
# OpenAI with no hint which input was at fault, so they are checked up front.
MAX_INPUTS_PER_REQUEST = 2048
# Ids accepted by one delete call.
MAX_IDS_PER_DELETE = 1000

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Same placeholder trick as main.py: openai>=3 raises at construction with no
# key, and this module must stay importable without one.
_openai = AsyncOpenAI(api_key=OPENAI_API_KEY or "missing-key", timeout=60.0, max_retries=2)

# Built on first use rather than at import. pc.index() resolves the index host
# over the network, so both are cached -- one describe call per process.
_pinecone: Pinecone | None = None
_index = None
_splitter_cache: RecursiveCharacterTextSplitter | None = None


def _splitter() -> RecursiveCharacterTextSplitter:
    """
    The chunker, built once per process.

    Built lazily on purpose. The constructor raises ValueError when the overlap
    is not smaller than the chunk size, and both come from the environment --
    at import time that would stop the app booting and take /health down with
    it. Deferred, a bad setting is a clear error on /ingest instead.
    """
    global _splitter_cache
    if _splitter_cache is None:
        # chunk_size and chunk_overlap are not declared parameters; they reach
        # the base TextSplitter through **kwargs. Its own defaults are 4000/200,
        # so leaving them out silently gives chunks five times the intended size.
        _splitter_cache = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    return _splitter_cache


def chunk_text(text: str) -> list[str]:
    """
    Split text on the largest natural boundary that fits: paragraphs, then lines,
    then words, then raw characters. Returns [] for empty or whitespace-only text.
    """
    if not text.strip():
        return []
    return _splitter().split_text(text)


def _client() -> Pinecone:
    """The Pinecone client, constructed once per process."""
    global _pinecone
    if _pinecone is None:
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not configured")
        _pinecone = Pinecone(api_key=PINECONE_API_KEY)
    return _pinecone


def _index_handle():
    """Data-plane handle for the configured index, resolved once per process."""
    global _index
    if _index is None:
        _index = _client().index(name=INDEX_NAME)
    return _index


def ensure_index() -> None:
    """
    Create the index if it is absent. Safe to call repeatedly -- exists() first,
    because create() on an existing name raises ConflictError.
    """
    pc = _client()
    if pc.indexes.exists(INDEX_NAME):
        return
    pc.indexes.create(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )


async def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed one batch of texts. The single entry point for both ingest and query,
    so the two can never drift onto different models or dimensions.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    if not texts:
        return []
    if len(texts) > MAX_INPUTS_PER_REQUEST:
        raise ValueError(
            f"{len(texts)} inputs exceeds the {MAX_INPUTS_PER_REQUEST} per-request limit"
        )

    response = await _openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    # Sorted by the index the API echoes back, so an embedding is never paired
    # with the wrong id further down -- that failure is silent, not loud.
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def _delete_stale_chunks(document_id: str, keep: int) -> None:
    """
    Delete chunks left over from a longer previous version of a document.

    Upserting only overwrites the ids being written, so a document that shrinks
    from ten chunks to five leaves five orphans behind that keep matching
    searches. This removes every chunk of this document numbered `keep` or above.

    Listing is eventually consistent, but only ids at or past the new chunk count
    are deleted and none of those were just written, so a stale listing can at
    worst miss an orphan until the next ingest -- it can never delete a chunk
    that belongs.
    """
    index = _index_handle()
    prefix = f"{document_id}#"
    stale = []
    for page in index.list(prefix=prefix):
        for item in page.vectors:
            suffix = item.id[len(prefix) :]
            # Only "<document_id>#<int>" is ours. A different document whose id
            # happens to start with this one plus "#" shares the prefix, and its
            # chunk ids leave a non-numeric remainder, so they are skipped.
            if suffix.isdigit() and int(suffix) >= keep:
                stale.append(item.id)

    # embed() caps a document at MAX_INPUTS_PER_REQUEST chunks, which is above
    # the per-call delete cap, so the batching is not decorative.
    for start in range(0, len(stale), MAX_IDS_PER_DELETE):
        index.delete(ids=stale[start : start + MAX_IDS_PER_DELETE])


async def ingest_document(
    document_id: str,
    text: str,
    metadata: dict[str, str] | None = None,
) -> int:
    """
    Chunk, embed and store one document. Returns the number of chunks written.

    Each chunk is stored under a deterministic id, "<document_id>#<chunk_index>",
    so re-ingesting the same document overwrites its chunks rather than
    duplicating them. A document that has since got shorter also has its surplus
    tail chunks removed, so what is stored always matches what was last sent.

    Pinecone metadata values must be strings, numbers, booleans or lists of
    strings, which is why the caller's metadata is typed dict[str, str].
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    extra = dict(metadata or {})
    # Pulled out so it is always present, even when the caller omitted it.
    source = extra.pop("source", "")

    vectors = await embed(chunks)
    records = [
        (
            f"{document_id}#{index}",
            vector,
            {
                **extra,
                "document_id": document_id,
                "chunk_index": index,
                "source": source,
                # Stored so a search can return the text without a second lookup.
                "text": chunk,
            },
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    # The Pinecone client is synchronous; off-thread so it cannot block the
    # event loop. show_progress defaults to True and would draw a tqdm bar into
    # the server log.
    await asyncio.to_thread(_index_handle().upsert, vectors=records, show_progress=False)
    # After the write, never before: a failed upsert must not take the previous
    # version's chunks with it.
    await asyncio.to_thread(_delete_stale_chunks, document_id, len(records))
    return len(records)


async def search(question: str, top_k: int = 5) -> list[dict]:
    """
    Embed a question and return the closest stored chunks, best match first.

    No LLM is involved -- this is the retrieval half on its own, which is what
    makes it usable to check what a prompt would have been built from.
    """
    vectors = await embed([question])
    response = await asyncio.to_thread(
        _index_handle().query,
        vector=vectors[0],
        top_k=top_k,
        include_metadata=True,
    )
    matches = []
    for match in response.matches:
        metadata = match.metadata or {}
        matches.append(
            {
                "id": match.id,
                "score": match.score,
                "document_id": metadata.get("document_id", ""),
                # Pinecone stores every metadata number as a double, so this
                # comes back as 3.0 rather than 3.
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "source": metadata.get("source", ""),
                "text": metadata.get("text", ""),
            }
        )
    return matches


def check() -> bool:
    """
    Confirm Pinecone is configured, reachable, and agrees with our embedding
    settings. Costs nothing -- it makes no OpenAI call -- and never prints a key.

    A dimension mismatch is reported as a failure: it is accepted at config time
    and only surfaces later as every upsert being rejected.
    """
    print(f"index:      {INDEX_NAME}")
    print(f"embedding:  {EMBEDDING_MODEL} @ {EMBEDDING_DIMENSIONS} dims")
    print(f"PINECONE_API_KEY found: {bool(PINECONE_API_KEY)} (length {len(PINECONE_API_KEY or '')})")
    if not PINECONE_API_KEY:
        return False

    try:
        pc = _client()
        if not pc.indexes.exists(INDEX_NAME):
            print(f"index '{INDEX_NAME}' does not exist -- call ensure_index() to create it")
            return False
        described = pc.indexes.describe(INDEX_NAME)
        stats = _index_handle().describe_index_stats()
    except Exception as exc:  # ponytail: one message for the caller; the trace is in the log
        print(f"unreachable: {type(exc).__name__}: {exc}")
        return False

    print(f"reachable:  yes (state {described.status.state}, ready {described.status.ready})")
    print(f"dimension:  {stats.dimension}, vectors: {stats.total_vector_count}")
    if stats.dimension != EMBEDDING_DIMENSIONS:
        print(f"MISMATCH: index is {stats.dimension}, EMBEDDING_DIMENSIONS is {EMBEDDING_DIMENSIONS}")
        print("Every upsert will be rejected until these agree.")
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
