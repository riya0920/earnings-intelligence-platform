"""
Retrieval Strategies Module

Implements four retrieval approaches for benchmarking:
1. Dense: Semantic search via sentence-transformers embeddings
2. Sparse: BM25 keyword matching
3. Hybrid: Weighted combination of dense + sparse scores
4. Hybrid + Reranker: Hybrid retrieval with cross-encoder reranking

Each strategy implements a common interface for fair comparison
in the evaluation pipeline.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

import chromadb

from src.chunking.strategies import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""

    content: str
    score: float
    metadata: dict
    rank: int


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Index a list of documents."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-k documents for a query."""
        pass


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval using sentence-transformers embeddings + ChromaDB.

    Encodes queries and documents into the same embedding space,
    retrieves by cosine similarity.
    """

    strategy_name = "dense"

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "dense_index",
        persist_dir: str = "data/vectorstore",
    ):
        self.embed_model = SentenceTransformer(embedding_model)
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.collection = None

    def index(self, documents: list[Document]) -> None:
        """Create or replace the ChromaDB collection with document embeddings."""
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Batch embed and add
        batch_size = 64
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.content for doc in batch]
            embeddings = self.embed_model.encode(
                texts, show_progress_bar=False
            ).tolist()

            self.collection.add(
                ids=[f"doc_{i + j}" for j in range(len(batch))],
                documents=texts,
                embeddings=embeddings,
                metadatas=[doc.metadata for doc in batch],
            )

        logger.info(f"[dense] Indexed {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self.collection is None:
            raise RuntimeError("Index not built. Call index() first.")

        query_embedding = self.embed_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results = []
        for rank, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # ChromaDB returns distance; convert to similarity score
            score = 1.0 - dist
            retrieval_results.append(
                RetrievalResult(
                    content=doc,
                    score=score,
                    metadata=meta,
                    rank=rank,
                )
            )

        return retrieval_results


class SparseRetriever(BaseRetriever):
    """
    Sparse retrieval using BM25 (Okapi).

    Pure keyword-based retrieval as a baseline comparison.
    """

    strategy_name = "sparse"

    def __init__(self):
        self.bm25 = None
        self.documents: list[Document] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def index(self, documents: list[Document]) -> None:
        self.documents = documents
        corpus = [self._tokenize(doc.content) for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"[sparse] Indexed {len(documents)} documents with BM25")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call index() first.")

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append(
                    RetrievalResult(
                        content=self.documents[idx].content,
                        score=float(scores[idx]),
                        metadata=self.documents[idx].metadata,
                        rank=rank,
                    )
                )

        return results


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining dense and sparse scores.

    Uses reciprocal rank fusion (RRF) or weighted score combination
    to merge results from both dense and sparse retrievers.
    """

    strategy_name = "hybrid"

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def index(self, documents: list[Document]) -> None:
        self.dense.index(documents)
        self.sparse.index(documents)
        logger.info(f"[hybrid] Indexed {len(documents)} documents in both retrievers")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        # Retrieve from both
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        content_scores: dict[str, dict] = {}

        for result in dense_results:
            key = result.content[:100]  # Use first 100 chars as key
            rrf_score = 1 / (k + result.rank)
            content_scores[key] = {
                "content": result.content,
                "metadata": result.metadata,
                "dense_rrf": rrf_score,
                "sparse_rrf": 0,
            }

        for result in sparse_results:
            key = result.content[:100]
            rrf_score = 1 / (k + result.rank)
            if key in content_scores:
                content_scores[key]["sparse_rrf"] = rrf_score
            else:
                content_scores[key] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "dense_rrf": 0,
                    "sparse_rrf": rrf_score,
                }

        # Compute combined scores
        scored = []
        for key, data in content_scores.items():
            combined = (
                self.dense_weight * data["dense_rrf"]
                + self.sparse_weight * data["sparse_rrf"]
            )
            scored.append((combined, data))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, data) in enumerate(scored[:top_k]):
            results.append(
                RetrievalResult(
                    content=data["content"],
                    score=score,
                    metadata=data["metadata"],
                    rank=rank,
                )
            )

        return results


class RerankedHybridRetriever(BaseRetriever):
    """
    Hybrid retrieval with cross-encoder reranking.

    First retrieves a larger candidate set via hybrid retrieval,
    then reranks using a cross-encoder for higher precision.
    """

    strategy_name = "hybrid_reranked"

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        initial_top_k: int = 20,
    ):
        self.hybrid = hybrid_retriever
        self.reranker = CrossEncoder(reranker_model)
        self.initial_top_k = initial_top_k

    def index(self, documents: list[Document]) -> None:
        self.hybrid.index(documents)
        logger.info(f"[hybrid_reranked] Index ready with reranker")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        # Get larger candidate set from hybrid retriever
        candidates = self.hybrid.retrieve(query, top_k=self.initial_top_k)

        if not candidates:
            return []

        # Rerank with cross-encoder
        pairs = [(query, c.content) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Sort by reranker score
        scored = list(zip(rerank_scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, candidate) in enumerate(scored[:top_k]):
            results.append(
                RetrievalResult(
                    content=candidate.content,
                    score=float(score),
                    metadata=candidate.metadata,
                    rank=rank,
                )
            )

        return results


class AzureSearchRetriever(BaseRetriever):
    """
    Retrieval backed by Azure AI Search instead of ChromaDB.

    The Azure counterpart to the local stack: documents are embedded (by default
    with Azure OpenAI embeddings) and pushed into an Azure AI Search index with a
    vector field; retrieval is Azure AI Search's native HYBRID query -- a vector
    search fused with BM25 keyword search in one call -- so it is the closest
    apples-to-apples comparison to the local HybridRetriever.

    Azure SDK objects are imported lazily and the client is only constructed when
    the retriever is actually used, so importing this module never requires the
    Azure packages or credentials on the local path.
    """

    strategy_name = "azure_search"

    def __init__(
        self,
        index_name: str = "earnings-filings",
        embedding_dim: int = 1536,
        hybrid: bool = True,
        embed_fn=None,
        endpoint: str | None = None,
        api_key: str | None = None,
    ):
        import os

        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.hybrid = hybrid
        # Default embedder is Azure OpenAI; injectable for tests / local vectors.
        if embed_fn is None:
            from src import backends

            embed_fn = backends.azure_openai_embed
        self._embed = embed_fn
        self._endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self._api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY")
        if not (self._endpoint and self._api_key):
            raise RuntimeError(
                "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be set for "
                "the Azure AI Search retriever (see the README's Azure section)."
            )

    def _index_client(self):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import SearchIndexClient

        return SearchIndexClient(self._endpoint, AzureKeyCredential(self._api_key))

    def _search_client(self):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        return SearchClient(
            self._endpoint, self.index_name, AzureKeyCredential(self._api_key)
        )

    def _ensure_index(self):
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        ic = self._index_client()
        try:
            ic.delete_index(self.index_name)
        except Exception:
            pass
        index = SearchIndex(
            name=self.index_name,
            fields=[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="metadata_json", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=self.embedding_dim,
                    vector_search_profile_name="hnsw-profile",
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
                profiles=[
                    VectorSearchProfile(
                        name="hnsw-profile", algorithm_configuration_name="hnsw-config"
                    )
                ],
            ),
        )
        ic.create_index(index)

    def index(self, documents: list[Document]) -> None:
        import json

        self._ensure_index()
        sc = self._search_client()
        batch = 64
        for i in range(0, len(documents), batch):
            chunk = documents[i : i + batch]
            vectors = self._embed([d.content for d in chunk])
            docs = [
                {
                    "id": f"doc_{i + j}",
                    "content": d.content,
                    "metadata_json": json.dumps(d.metadata),
                    "content_vector": vec,
                }
                for j, (d, vec) in enumerate(zip(chunk, vectors))
            ]
            sc.upload_documents(documents=docs)
        logger.info(f"[azure_search] Indexed {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        import json

        from azure.search.documents.models import VectorizedQuery

        qvec = self._embed([query])[0]
        vq = VectorizedQuery(
            vector=qvec, k_nearest_neighbors=top_k, fields="content_vector"
        )
        sc = self._search_client()
        results = sc.search(
            search_text=query if self.hybrid else None,  # hybrid = vector + BM25
            vector_queries=[vq],
            top=top_k,
            select=["content", "metadata_json"],
        )
        out = []
        for rank, r in enumerate(results):
            out.append(
                RetrievalResult(
                    content=r.get("content", ""),
                    score=float(r.get("@search.score", 0.0)),
                    metadata=json.loads(r.get("metadata_json") or "{}"),
                    rank=rank,
                )
            )
        return out


def build_retriever(strategy: str, config: dict) -> BaseRetriever:
    """
    Factory function to build a retriever from config.

    Handles the dependency chain: hybrid needs dense + sparse,
    reranked needs hybrid. `azure_search` is the Azure AI Search backend
    (selected when the config switch is on azure; see src/backends.py).
    """
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    persist_dir = config.get("vectorstore_path", "data/vectorstore")

    if strategy == "azure_search":
        az = config.get("azure_search", {})
        return AzureSearchRetriever(
            index_name=az.get("index_name", "earnings-filings"),
            embedding_dim=az.get("embedding_dim", 1536),
            hybrid=az.get("hybrid", True),
        )

    if strategy == "dense":
        return DenseRetriever(
            embedding_model=embedding_model,
            collection_name=f"dense_{config.get('collection_suffix', 'default')}",
            persist_dir=persist_dir,
        )

    elif strategy == "sparse":
        return SparseRetriever()

    elif strategy == "hybrid":
        dense = DenseRetriever(
            embedding_model=embedding_model,
            collection_name=f"hybrid_dense_{config.get('collection_suffix', 'default')}",
            persist_dir=persist_dir,
        )
        sparse = SparseRetriever()
        return HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            dense_weight=config.get("dense_weight", 0.6),
            sparse_weight=config.get("sparse_weight", 0.4),
        )

    elif strategy == "hybrid_reranked":
        dense = DenseRetriever(
            embedding_model=embedding_model,
            collection_name=f"reranked_dense_{config.get('collection_suffix', 'default')}",
            persist_dir=persist_dir,
        )
        sparse = SparseRetriever()
        hybrid = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            dense_weight=config.get("dense_weight", 0.6),
            sparse_weight=config.get("sparse_weight", 0.4),
        )
        return RerankedHybridRetriever(
            hybrid_retriever=hybrid,
            reranker_model=config.get(
                "reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2"
            ),
            initial_top_k=config.get("top_k", 20),
        )

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Options: dense, sparse, hybrid, "
            f"hybrid_reranked, azure_search"
        )
