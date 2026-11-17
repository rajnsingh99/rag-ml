from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    pdf_path: Path = PROJECT_DIR / "intro-to-ml-book.pdf"
    db_dir: Path = PROJECT_DIR / "vectordb"

    # chunking (tiktoken cl100k tokens) - picked from the chunk-size experiment in the notebook
    chunk_size: int = 300
    chunk_overlap: int = 60

    # embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # retrieval
    top_k: int = 5

    # generation (local Ollama)
    llm_model: str = "llama3.1:8b"
    num_ctx: int = 8192  # Ollama's default context is too small for 5-7 chunks
    temperature: float = 0.0
