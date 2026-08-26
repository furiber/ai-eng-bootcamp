"""Offline checks for rag.py — OpenAI and Pinecone are both stubbed.

Nothing is billed and no index is touched. Run: python test_rag.py
"""

import asyncio
from types import SimpleNamespace

import rag


def _stub_embeddings(vectors, *, shuffle=False):
    """
    Replace the embeddings call with a coroutine returning canned vectors.

    shuffle=True returns the data out of order while keeping each item's own
    `index`, which is what the API's ordering contract actually guarantees.
    """
    data = [SimpleNamespace(index=i, embedding=v) for i, v in enumerate(vectors)]
    if shuffle:
        data.reverse()

    async def fake_create(**_kwargs):
        return SimpleNamespace(data=data, model=rag.EMBEDDING_MODEL)

    rag._openai.embeddings.create = fake_create


class _FakeIndex:
    """Captures what would have been sent to Pinecone."""

    def __init__(self, stored_ids=()):
        self.upserted = None
        self.queried = None
        self.deleted = []
        # Ids already in the index, as a previous ingest would have left them.
        self.stored_ids = list(stored_ids)

    def list(self, **kwargs):
        """One page of ListItem-shaped objects, matching the real response."""
        prefix = kwargs.get("prefix") or ""
        matching = [SimpleNamespace(id=i) for i in self.stored_ids if i.startswith(prefix)]
        return iter([SimpleNamespace(vectors=matching)])

    def delete(self, **kwargs):
        self.deleted.extend(kwargs["ids"])

    def upsert(self, **kwargs):
        self.upserted = kwargs
        return SimpleNamespace(upserted_count=len(kwargs["vectors"]))

    def query(self, **kwargs):
        self.queried = kwargs
        match = SimpleNamespace(
            id="handbook#3",
            score=0.91,
            metadata={
                "text": "stored text",
                "document_id": "handbook",
                # Pinecone returns metadata numbers as doubles, not ints.
                "chunk_index": 3.0,
                "source": "handbook.md",
            },
        )
        return SimpleNamespace(matches=[match])


def _stub_index():
    fake = _FakeIndex()
    rag._index = fake
    return fake


PARAGRAPH_BREAK = chr(10) * 2

_CONFIGURED_CHUNKING = (rag.CHUNK_SIZE, rag.CHUNK_OVERLAP)


def _use_chunk_settings(size=None, overlap=None):
    """
    Set the chunk size and overlap, resetting the cached splitter so the change
    takes effect. Called with no arguments it restores the configured values, so
    tests stay independent of the order they run in.
    """
    if size is None:
        size, overlap = _CONFIGURED_CHUNKING
    rag.CHUNK_SIZE, rag.CHUNK_OVERLAP = size, overlap
    rag._splitter_cache = None


def test_embed_of_nothing_makes_no_call():
    """An empty batch must not become a billed request for zero inputs."""
    rag._openai.embeddings.create = None  # any call would raise TypeError
    assert asyncio.run(rag.embed([])) == []
    print("PASS  embedding an empty list returns [] without calling OpenAI")


def test_oversized_batch_is_rejected_locally():
    _stub_embeddings([[0.1]])
    too_many = ["x"] * (rag.MAX_INPUTS_PER_REQUEST + 1)
    try:
        asyncio.run(rag.embed(too_many))
    except ValueError as exc:
        assert str(rag.MAX_INPUTS_PER_REQUEST) in str(exc), exc
        print("PASS  a batch over the per-request limit is rejected before the network")
        return
    raise AssertionError("oversized batch was not rejected")


def test_embeddings_are_reordered_by_index():
    """Out-of-order response data must come back in input order, not wire order."""
    _stub_embeddings([[1.0], [2.0], [3.0]], shuffle=True)
    assert asyncio.run(rag.embed(["a", "b", "c"])) == [[1.0], [2.0], [3.0]]
    print("PASS  embeddings are returned in input order regardless of response order")


def test_ingest_pairs_each_chunk_with_its_own_vector():
    """The silent-corruption case: a chunk stored against the wrong vector."""
    # Two short paragraphs only split when the chunk size is smaller than they are.
    _use_chunk_settings(20, 0)
    _stub_embeddings([[1.0], [2.0]], shuffle=True)
    fake = _stub_index()
    written = asyncio.run(
        rag.ingest_document("doc-a", "first para\n\nsecond para", {"source": "notes.md"})
    )

    assert written == 2, written
    ids = [vector[0] for vector in fake.upserted["vectors"]]
    values = [vector[1] for vector in fake.upserted["vectors"]]
    assert ids == ["doc-a#0", "doc-a#1"], ids
    assert values == [[1.0], [2.0]], values
    # Default is True, which would draw a progress bar into the server log.
    assert fake.upserted["show_progress"] is False, fake.upserted
    print("PASS  ingest pairs each chunk with its own vector, in order")


def test_ingest_writes_the_required_metadata():
    # Two short paragraphs only split when the chunk size is smaller than they are.
    _use_chunk_settings(20, 0)
    _stub_embeddings([[1.0], [2.0]])
    fake = _stub_index()
    asyncio.run(rag.ingest_document("doc-a", "first para\n\nsecond para", {"source": "notes.md"}))

    first, second = (vector[2] for vector in fake.upserted["vectors"])
    assert first["document_id"] == "doc-a" == second["document_id"], (first, second)
    assert (first["chunk_index"], second["chunk_index"]) == (0, 1), (first, second)
    assert first["source"] == "notes.md" == second["source"], (first, second)
    assert first["text"] == "first para", first
    print("PASS  each chunk carries document_id, chunk_index, source and text")


def test_ingest_keeps_extra_metadata_and_defaults_source():
    _stub_embeddings([[1.0]])
    fake = _stub_index()
    asyncio.run(rag.ingest_document("doc-a", "only para", {"author": "furiber"}))

    metadata = fake.upserted["vectors"][0][2]
    assert metadata["author"] == "furiber", metadata
    # Always present, so a search never has to guess whether the key exists.
    assert metadata["source"] == "", metadata
    print("PASS  extra metadata is kept and an omitted source defaults to empty")


def test_reingesting_a_document_reuses_its_ids():
    """Deterministic ids are what make a retry overwrite instead of duplicate."""
    fake = _stub_index()
    _stub_embeddings([[1.0]])
    asyncio.run(rag.ingest_document("doc-a", "first version"))
    first_ids = [vector[0] for vector in fake.upserted["vectors"]]
    _stub_embeddings([[9.0]])
    asyncio.run(rag.ingest_document("doc-a", "second version"))

    assert first_ids == [vector[0] for vector in fake.upserted["vectors"]] == ["doc-a#0"]
    print("PASS  re-ingesting a document writes the same ids")


def test_ingest_of_empty_text_touches_neither_service():
    rag._openai.embeddings.create = None
    rag._index = None  # any index call would fail resolving a host
    assert asyncio.run(rag.ingest_document("doc-a", "   \n  ")) == 0
    print("PASS  ingesting whitespace-only text is a no-op")


def test_chunking_respects_size_and_overlap():
    _use_chunk_settings()
    text = "word " * 600  # ~3000 chars, well past one chunk
    chunks = rag.chunk_text(text)

    assert len(chunks) > 1, len(chunks)
    assert all(len(chunk) <= rag.CHUNK_SIZE for chunk in chunks), [len(c) for c in chunks]
    # Overlap is what keeps a sentence split across a boundary retrievable.
    assert chunks[0][-rag.CHUNK_OVERLAP:] in chunks[1], "second chunk does not overlap the first"
    print(f"PASS  chunking honours size {rag.CHUNK_SIZE} and overlap {rag.CHUNK_OVERLAP}")


def test_chunking_empty_text_gives_no_chunks():
    _use_chunk_settings()
    assert rag.chunk_text("") == []
    assert rag.chunk_text("   \n\t ") == []
    print("PASS  empty or whitespace-only text produces no chunks")


def test_shrinking_a_document_deletes_its_surplus_chunks():
    """The orphan case: five chunks left behind when a ten-chunk document shrinks."""
    _use_chunk_settings(20, 0)
    _stub_embeddings([[1.0], [2.0]])
    fake = _stub_index()
    fake.stored_ids = [f"doc-a#{i}" for i in range(5)]

    asyncio.run(rag.ingest_document("doc-a", "first para" + PARAGRAPH_BREAK + "second para"))

    assert fake.deleted == ["doc-a#2", "doc-a#3", "doc-a#4"], fake.deleted
    print("PASS  a document that shrinks has its surplus chunks deleted")


def test_growing_a_document_deletes_nothing():
    _use_chunk_settings(20, 0)
    _stub_embeddings([[1.0], [2.0]])
    fake = _stub_index()
    fake.stored_ids = ["doc-a#0"]

    asyncio.run(rag.ingest_document("doc-a", "first para" + PARAGRAPH_BREAK + "second para"))

    assert fake.deleted == [], fake.deleted
    print("PASS  a document that grows deletes nothing")


def test_cleanup_leaves_other_documents_alone():
    """"doc-a#extra" shares the "doc-a#" prefix but is a different document."""
    _use_chunk_settings()
    _stub_embeddings([[1.0]])
    fake = _stub_index()
    fake.stored_ids = ["doc-a#0", "doc-a#1", "doc-a#extra#0", "doc-a#extra#1"]

    asyncio.run(rag.ingest_document("doc-a", "one short document"))

    # Only doc-a's own numbered chunks past the new count.
    assert fake.deleted == ["doc-a#1"], fake.deleted
    print("PASS  cleanup skips ids that are not this document's numbered chunks")


def test_cleanup_runs_after_the_upsert_not_before():
    """A failed write must not take the previous version's chunks with it."""
    _use_chunk_settings()
    _stub_embeddings([[1.0]])
    fake = _stub_index()
    fake.stored_ids = ["doc-a#0", "doc-a#1"]

    def exploding_upsert(**_kwargs):
        raise RuntimeError("upsert failed")

    fake.upsert = exploding_upsert
    try:
        asyncio.run(rag.ingest_document("doc-a", "one short document"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("the failing upsert did not propagate")

    assert fake.deleted == [], fake.deleted
    print("PASS  nothing is deleted when the upsert fails")


def test_search_asks_for_metadata_and_returns_text():
    _stub_embeddings([[0.5]])
    fake = _stub_index()
    results = asyncio.run(rag.search("a question", top_k=3))

    assert fake.queried["top_k"] == 3, fake.queried
    assert fake.queried["vector"] == [0.5], fake.queried
    # Without this the text is never returned and every hit reads as empty.
    assert fake.queried["include_metadata"] is True, fake.queried
    assert results == [
        {
            "id": "handbook#3",
            "score": 0.91,
            "document_id": "handbook",
            "chunk_index": 3,
            "source": "handbook.md",
            "text": "stored text",
        }
    ], results
    print("PASS  search requests metadata and returns it with the stored text")


def test_search_coerces_the_double_pinecone_returns_for_chunk_index():
    """Pinecone stores metadata numbers as doubles; 3.0 in a JSON body reads wrong."""
    _stub_embeddings([[0.5]])
    _stub_index()
    assert isinstance(asyncio.run(rag.search("a question"))[0]["chunk_index"], int)
    print("PASS  chunk_index comes back as an int, not a float")


def test_search_defaults_to_five_matches():
    _stub_embeddings([[0.5]])
    fake = _stub_index()
    asyncio.run(rag.search("a question"))
    assert fake.queried["top_k"] == 5, fake.queried
    print("PASS  search asks for five matches by default")


def test_ingest_and_query_use_the_same_model():
    """Vectors from different models are not comparable — one constant, both paths."""
    seen = []

    async def fake_create(**kwargs):
        seen.append((kwargs["model"], kwargs["dimensions"]))
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.5])])

    rag._openai.embeddings.create = fake_create
    _stub_index()
    asyncio.run(rag.ingest_document("doc-a", "first"))
    asyncio.run(rag.search("a question"))

    assert len(seen) == 2 and seen[0] == seen[1], seen
    assert seen[0] == (rag.EMBEDDING_MODEL, rag.EMBEDDING_DIMENSIONS), seen
    print("PASS  ingest and query embed with the same model and dimensions")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall checks passed")
