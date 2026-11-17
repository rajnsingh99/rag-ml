"""Embeddings (SentenceTransformers) + vector storage and similarity search (ChromaDB)."""

import re
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from .chunker import Chunk

# bge models were trained with an instruction prefix on the *query* side only.
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

_models: dict[str, SentenceTransformer] = {}


def get_embedder(name: str) -> SentenceTransformer:
    if name not in _models:
        _models[name] = SentenceTransformer(name)
    return _models[name]


def collection_name(model: str, chunk_size: int, overlap: int) -> str:
    short = re.sub(r"[^a-zA-Z0-9]+", "-", model.split("/")[-1]).strip("-").lower()
    return f"{short}-cs{chunk_size}-ov{overlap}"


@dataclass
class Hit:
    rank: int
    score: float  # cosine similarity, higher = more similar
    chunk_id: str
    chapter: str
    page_label: str
    text: str


class VectorStore:
    def __init__(self, db_dir, model: str, chunk_size: int, overlap: int):
        self.model_name = model
        self.embedder = get_embedder(model)
        self.client = chromadb.PersistentClient(path=str(db_dir))
        self.name = collection_name(model, chunk_size, overlap)
        self.collection = self.client.get_or_create_collection(
            self.name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def build(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """(Re)build the collection from scratch so re-running never leaves stale chunks."""
        self.client.delete_collection(self.name)
        self.collection = self.client.create_collection(self.name, metadata={"hnsw:space": "cosine"})
        texts = [c.text for c in chunks]
        vectors = self.embedder.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=texts,
            embeddings=vectors.tolist(),
            metadatas=[{
                "chapter": c.chapter,
                "page_label": c.page_label,
                "pdf_pages": ",".join(map(str, c.pdf_pages)),
                "n_tokens": c.n_tokens,
            } for c in chunks],
        )

    def embed_query(self, query: str):
        prefix = QUERY_PREFIX.get(self.model_name, "")
        return self.embedder.encode(prefix + query, normalize_embeddings=True)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        res = self.collection.query(query_embeddings=[self.embed_query(query).tolist()], n_results=k)
        hits = []
        for i, (cid, doc, meta, dist) in enumerate(zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )):
            hits.append(Hit(
                rank=i + 1,
                score=round(1 - dist, 4),
                chunk_id=cid,
                chapter=meta["chapter"],
                page_label=meta["page_label"],
                text=doc,
            ))
        return hits
