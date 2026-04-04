"""
Chunking Strategies Module

Implements three chunking approaches for benchmarking:
1. Fixed-size: Token-count based with overlap
2. Sentence-based: Uses NLTK sentence boundaries with grouping
3. Semantic: Embedding similarity-based splitting

Each strategy produces Document objects with metadata preserved from
the ingestion layer, enabling per-strategy retrieval evaluation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import nltk
import numpy as np
import tiktoken

logger = logging.getLogger(__name__)

# Ensure sentence tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


@dataclass
class Document:
    """A chunked document with content and metadata."""
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def token_count(self) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(self.content))

    @property
    def id(self) -> str:
        """Generate a deterministic ID from metadata."""
        parts = [
            self.metadata.get("ticker", ""),
            self.metadata.get("filing_date", ""),
            self.metadata.get("section", ""),
            self.metadata.get("chunk_strategy", ""),
            str(self.metadata.get("chunk_index", 0)),
        ]
        return "_".join(parts)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> list[Document]:
        """Split text into chunks with metadata."""
        pass

    def chunk_sections(self, sections: list[dict]) -> list[Document]:
        """Chunk a list of filing sections, preserving metadata."""
        all_docs = []
        for section in sections:
            base_metadata = {
                "company": section.get("company", ""),
                "ticker": section.get("ticker", ""),
                "filing_type": section.get("filing_type", ""),
                "filing_date": section.get("filing_date", ""),
                "section": section.get("section_name", ""),
                "chunk_strategy": self.strategy_name,
            }
            chunks = self.chunk(section["content"], base_metadata)
            all_docs.extend(chunks)

        logger.info(
            f"[{self.strategy_name}] Chunked {len(sections)} sections "
            f"into {len(all_docs)} documents"
        )
        return all_docs


class FixedChunker(BaseChunker):
    """
    Fixed-size chunking by token count with overlap.

    Simple baseline strategy. Splits text into chunks of exactly
    `chunk_size` tokens with `chunk_overlap` token overlap.
    """

    strategy_name = "fixed"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str, metadata: dict) -> list[Document]:
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            chunk_meta = {
                **metadata,
                "chunk_index": len(chunks),
                "token_count": len(chunk_tokens),
            }
            chunks.append(Document(content=chunk_text, metadata=chunk_meta))

            # Move forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap
            if end == len(tokens):
                break

        return chunks


class SentenceChunker(BaseChunker):
    """
    Sentence-based chunking using NLTK boundaries.

    Groups sentences together within min/max bounds, preserving
    semantic boundaries at sentence level.
    """

    strategy_name = "sentence"

    def __init__(self, min_sentences: int = 3, max_sentences: int = 8):
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str, metadata: dict) -> list[Document]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_group = []

        for sentence in sentences:
            current_group.append(sentence)

            if len(current_group) >= self.max_sentences:
                chunk_text = " ".join(current_group)
                chunk_meta = {
                    **metadata,
                    "chunk_index": len(chunks),
                    "sentence_count": len(current_group),
                    "token_count": len(self.encoder.encode(chunk_text)),
                }
                chunks.append(Document(content=chunk_text, metadata=chunk_meta))
                current_group = []

        # Handle remaining sentences
        if len(current_group) >= self.min_sentences:
            chunk_text = " ".join(current_group)
            chunk_meta = {
                **metadata,
                "chunk_index": len(chunks),
                "sentence_count": len(current_group),
                "token_count": len(self.encoder.encode(chunk_text)),
            }
            chunks.append(Document(content=chunk_text, metadata=chunk_meta))
        elif current_group and chunks:
            # Merge short remainder into the last chunk
            last = chunks[-1]
            merged = last.content + " " + " ".join(current_group)
            last.content = merged
            last.metadata["sentence_count"] += len(current_group)
            last.metadata["token_count"] = len(self.encoder.encode(merged))

        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunking based on embedding similarity.

    Splits text at points where consecutive sentence embeddings
    diverge beyond a threshold, creating chunks that are semantically
    coherent rather than arbitrarily cut.

    Uses sentence-transformers for embedding computation.
    """

    strategy_name = "semantic"

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def chunk(self, text: str, metadata: dict) -> list[Document]:
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return [Document(
                content=text,
                metadata={**metadata, "chunk_index": 0, "token_count": len(self.encoder.encode(text))},
            )]

        # Compute embeddings for all sentences
        embeddings = self.model.encode(sentences, show_progress_bar=False)

        # Find split points where consecutive similarity drops below threshold
        split_indices = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.similarity_threshold:
                split_indices.append(i + 1)

        # Build chunks from split points
        chunks = []
        prev = 0
        for split_idx in split_indices:
            chunk_sentences = sentences[prev:split_idx]
            chunk_text = " ".join(chunk_sentences)

            # Enforce min/max size constraints
            token_count = len(self.encoder.encode(chunk_text))
            if token_count < self.min_chunk_size and chunks:
                # Merge small chunk into previous
                last = chunks[-1]
                last.content += " " + chunk_text
                last.metadata["token_count"] = len(self.encoder.encode(last.content))
            elif token_count > self.max_chunk_size:
                # Split oversized chunk with fixed strategy as fallback
                fallback = FixedChunker(
                    chunk_size=self.max_chunk_size // 2,
                    chunk_overlap=32,
                )
                sub_chunks = fallback.chunk(chunk_text, {
                    **metadata,
                    "chunk_strategy": "semantic",
                })
                for j, sc in enumerate(sub_chunks):
                    sc.metadata["chunk_index"] = len(chunks) + j
                chunks.extend(sub_chunks)
            else:
                chunk_meta = {
                    **metadata,
                    "chunk_index": len(chunks),
                    "sentence_count": len(chunk_sentences),
                    "token_count": token_count,
                }
                chunks.append(Document(content=chunk_text, metadata=chunk_meta))

            prev = split_idx

        # Handle remaining sentences
        if prev < len(sentences):
            chunk_text = " ".join(sentences[prev:])
            token_count = len(self.encoder.encode(chunk_text))
            if token_count >= self.min_chunk_size or not chunks:
                chunks.append(Document(
                    content=chunk_text,
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                        "sentence_count": len(sentences) - prev,
                        "token_count": token_count,
                    },
                ))
            elif chunks:
                last = chunks[-1]
                last.content += " " + chunk_text
                last.metadata["token_count"] = len(self.encoder.encode(last.content))

        return chunks


def get_chunker(strategy: str, config: dict) -> BaseChunker:
    """Factory function to instantiate a chunker by strategy name."""
    chunkers = {
        "fixed": lambda: FixedChunker(
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 64),
        ),
        "sentence": lambda: SentenceChunker(
            min_sentences=config.get("min_sentences", 3),
            max_sentences=config.get("max_sentences", 8),
        ),
        "semantic": lambda: SemanticChunker(
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            similarity_threshold=config.get("similarity_threshold", 0.75),
            min_chunk_size=config.get("min_chunk_size", 100),
            max_chunk_size=config.get("max_chunk_size", 1000),
        ),
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(chunkers.keys())}")

    return chunkers[strategy]()
