"""Small RAG pipeline over an ML book PDF: load -> clean -> chunk -> embed -> retrieve -> generate."""

from .config import Settings
from .pipeline import RAGPipeline

__all__ = ["Settings", "RAGPipeline"]
