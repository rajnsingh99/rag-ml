"""Ties the pieces together: document processing -> retrieval -> generation."""

import re
import time
from dataclasses import dataclass, field

from .chunker import Chunk, chunk_pages
from .config import Settings
from .generator import NOT_FOUND, OllamaGenerator
from .loader import load_pages
from .store import Hit, VectorStore


@dataclass
class Answer:
    question: str
    answer: str
    hits: list[Hit]
    cited: list[int] = field(default_factory=list)  # which S# the answer cited
    retrieval_secs: float = 0.0
    generation_secs: float = 0.0

    @property
    def found(self) -> bool:
        return NOT_FOUND.lower().rstrip(".") not in self.answer.lower()

    @property
    def sources(self) -> list[Hit]:
        return [self.hits[i - 1] for i in self.cited if 0 < i <= len(self.hits)]


def cited_ids(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[S(\d+)\]", text)})


def normalise_refusal(text: str) -> str:
    """Small models sometimes refuse and then keep talking. If the answer starts
    with the refusal sentence and cites nothing, keep just the refusal."""
    if text.lower().startswith(NOT_FOUND.lower().rstrip(".")) and not cited_ids(text):
        return NOT_FOUND
    return text


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        s = self.settings
        self.store = VectorStore(s.db_dir, s.embed_model, s.chunk_size, s.chunk_overlap)
        self.generator = OllamaGenerator(s.llm_model, s.num_ctx, s.temperature)

    # --- document processing ---------------------------------------------
    def build_index(self) -> list[Chunk]:
        s = self.settings
        pages = load_pages(s.pdf_path)
        chunks = chunk_pages(pages, s.chunk_size, s.chunk_overlap)
        self.store.build(chunks)
        return chunks

    def ensure_index(self) -> None:
        if self.store.count() == 0:
            self.build_index()

    # --- retrieval + generation ------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.store.search(question, k or self.settings.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question, k)
        t1 = time.perf_counter()
        text = normalise_refusal(self.generator.generate(question, hits))
        t2 = time.perf_counter()
        return Answer(question, text, hits, cited_ids(text), round(t1 - t0, 2), round(t2 - t1, 2))
