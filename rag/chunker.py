"""Token-based sliding-window chunking that remembers which pages each chunk came from."""

from dataclasses import dataclass, field

import tiktoken

from .loader import Page

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    chunk_id: str
    chapter: str
    text: str
    n_tokens: int
    pdf_pages: list[int] = field(default_factory=list)
    book_pages: list[int] = field(default_factory=list)

    @property
    def page_label(self) -> str:
        # front-matter pages have roman-numeral page numbers we can't read, so fall back to PDF pages
        prefix, pages = ("", self.book_pages) if self.book_pages else ("PDF ", self.pdf_pages)
        if not pages:
            return "?"
        return f"{prefix}p. {pages[0]}" if pages[0] == pages[-1] else f"{prefix}pp. {pages[0]}-{pages[-1]}"


def chunk_pages(pages: list[Page], chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Chunk each chapter separately so a chunk never mixes two chapters.

    Every token is tagged with its page, so after slicing a window we know
    exactly which pages the chunk spans (used for citations).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    by_chapter: dict[str, list[Page]] = {}
    for p in pages:
        by_chapter.setdefault(p.chapter, []).append(p)

    chunks = []
    step = chunk_size - overlap
    for chapter, chapter_pages in by_chapter.items():
        tokens, token_pages = [], []
        for p in chapter_pages:
            ids = _ENC.encode(p.text + "\n")
            tokens.extend(ids)
            token_pages.extend([(p.pdf_page, p.book_page)] * len(ids))

        for start in range(0, len(tokens), step):
            window = tokens[start:start + chunk_size]
            # a tiny leftover at the end is already covered by the previous window's overlap
            if start > 0 and len(window) <= overlap:
                break
            spans = token_pages[start:start + chunk_size]
            pdf_pages = sorted({pdf for pdf, _ in spans})
            book_pages = sorted({bp for _, bp in spans if bp is not None})
            chunks.append(Chunk(
                chunk_id=f"c{len(chunks):04d}",
                chapter=chapter,
                text=_ENC.decode(window).strip(),
                n_tokens=len(window),
                pdf_pages=pdf_pages,
                book_pages=book_pages,
            ))
    return chunks
