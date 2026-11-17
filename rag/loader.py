"""Load the book PDF page by page and clean up the extraction problems we found during exploration."""

import re
import unicodedata
from dataclasses import dataclass

from pypdf import PdfReader

# Running footers look like "52 | Chapter 2: Supervised Learning" (even pages)
# or "Supervised Machine Learning Algorithms | 53" (odd pages).
FOOTER_LEFT = re.compile(r"^(\d{1,3}) \| .+$")
FOOTER_RIGHT = re.compile(r"^.+ \| (\d{1,3})$")

# The book wraps words with a unicode hyphen (U+2010), e.g. "pow‐\nerful".
HYPHEN_BREAK = re.compile(r"(\w)[‐-]\n(\w)")
CODE_IN = re.compile(r"^In\[\d+\]:\s*", re.M)
CODE_OUT = re.compile(r"^Out\[\d+\]:\s*", re.M)


@dataclass
class Page:
    pdf_page: int        # 1-based page number in the PDF file
    book_page: int | None  # page number printed in the book (from the footer)
    chapter: str
    text: str


def chapter_ranges(reader: PdfReader) -> list[tuple[int, str]]:
    """Return (start_pdf_index, title) for each top-level outline entry, sorted by page."""
    entries = []
    for item in reader.outline:
        if isinstance(item, list):
            continue
        entries.append((reader.get_destination_page_number(item), item.title))
    return sorted(entries)


def chapter_for(index: int, ranges: list[tuple[int, str]]) -> str:
    title = "Front matter"
    for start, name in ranges:
        if index >= start:
            title = name
    return title


def clean_page(raw: str) -> tuple[str, int | None]:
    """Strip the running footer (returning the printed page number) and fix extraction artefacts."""
    text = unicodedata.normalize("NFKC", raw)  # ligatures like "ﬁ" -> "fi"
    lines = text.splitlines()

    book_page = None
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        last = lines[-1].strip()
        m = FOOTER_LEFT.match(last) or FOOTER_RIGHT.match(last)
        if m:
            book_page = int(m.group(1))
            lines.pop()

    text = "\n".join(lines)
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = CODE_IN.sub("Code: ", text)
    text = CODE_OUT.sub("Output: ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), book_page


# Pages that hurt retrieval more than they help: copyright page, table of
# contents and the index (lists of terms + page numbers match almost any query).
SKIP_SECTIONS = {"Front matter", "Copyright", "Table of Contents", "Index", "About the Authors", "Colophon"}


def load_pages(pdf_path, skip_sections=SKIP_SECTIONS) -> list[Page]:
    reader = PdfReader(str(pdf_path))
    ranges = chapter_ranges(reader)
    pages = []
    for i, pdf_page in enumerate(reader.pages):
        chapter = chapter_for(i, ranges)
        if chapter in skip_sections:
            continue
        text, book_page = clean_page(pdf_page.extract_text() or "")
        if len(text) < 50:  # blank / image-only pages
            continue
        pages.append(Page(pdf_page=i + 1, book_page=book_page, chapter=chapter, text=text))
    _fill_missing_book_pages(pages)
    return pages


def _fill_missing_book_pages(pages: list[Page]) -> None:
    """Chapter-opening pages have no footer; infer their number from the nearest numbered page."""
    offsets = [p.book_page - p.pdf_page for p in pages if p.book_page is not None]
    if not offsets:
        return
    for idx, p in enumerate(pages):
        if p.book_page is not None:
            continue
        neighbours = [q for q in pages[idx + 1:idx + 4] + pages[max(0, idx - 3):idx] if q.book_page is not None]
        if neighbours:
            p.book_page = neighbours[0].book_page - neighbours[0].pdf_page + p.pdf_page
