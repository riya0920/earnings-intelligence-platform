"""
Tests for the Azure backend switch (src/backends.py) and the Azure AI Search
retriever (src/retrieval/retrievers.AzureSearchRetriever).

Everything is mocked: no Azure account, no network. These verify the WIRING --
that the switch selects the right client/retriever and that the retriever maps
Azure AI Search results into the project's RetrievalResult shape. Measuring live
retrieval quality is a separate, credentialed step documented in the README.
"""
import json
from unittest import mock

import pytest

from src import backends
from src.chunking.strategies import Document
from src.retrieval.retrievers import AzureSearchRetriever, RetrievalResult, build_retriever


# -- the switch -------------------------------------------------------------
def test_default_backend_is_local(monkeypatch):
    monkeypatch.delenv("EIP_BACKEND", raising=False)
    assert backends.active_backend() == "local"
    assert backends.is_azure() is False


def test_env_selects_azure(monkeypatch):
    monkeypatch.setenv("EIP_BACKEND", "azure")
    assert backends.active_backend() == "azure"
    assert backends.is_azure() is True


def test_config_selects_azure_when_env_absent(monkeypatch):
    monkeypatch.delenv("EIP_BACKEND", raising=False)
    assert backends.active_backend({"backend": "azure"}) == "azure"
    # env wins over config
    monkeypatch.setenv("EIP_BACKEND", "local")
    assert backends.active_backend({"backend": "azure"}) == "local"


def test_invalid_backend_raises(monkeypatch):
    monkeypatch.setenv("EIP_BACKEND", "gcp")
    with pytest.raises(ValueError):
        backends.active_backend()


# -- the chat/embeddings client factory -------------------------------------
def test_get_chat_client_local_is_openai(monkeypatch):
    monkeypatch.delenv("EIP_BACKEND", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = backends.get_chat_client()
    assert type(client).__name__ == "OpenAI"


def test_get_chat_client_azure_is_azureopenai(monkeypatch):
    monkeypatch.setenv("EIP_BACKEND", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-test")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    client = backends.get_chat_client()
    assert type(client).__name__ == "AzureOpenAI"


def test_chat_model_uses_deployment_name_on_azure(monkeypatch):
    monkeypatch.setenv("EIP_BACKEND", "azure")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "my-gpt4o-deploy")
    assert backends.chat_model("gpt-4o-mini") == "my-gpt4o-deploy"
    monkeypatch.setenv("EIP_BACKEND", "local")
    assert backends.chat_model("gpt-4o-mini") == "gpt-4o-mini"


def test_azure_client_missing_env_raises(monkeypatch):
    monkeypatch.setenv("EIP_BACKEND", "azure")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError):
        backends.get_chat_client()


# -- build_retriever routes to Azure AI Search ------------------------------
def _dummy_search_env(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "search-key")


def test_build_retriever_returns_azure_search(monkeypatch):
    _dummy_search_env(monkeypatch)
    r = build_retriever("azure_search",
                        {"azure_search": {"index_name": "idx-x", "embedding_dim": 8}})
    assert isinstance(r, AzureSearchRetriever)
    assert r.index_name == "idx-x"
    assert r.embedding_dim == 8


def test_azure_search_retriever_requires_credentials(monkeypatch):
    monkeypatch.delenv("AZURE_SEARCH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SEARCH_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        AzureSearchRetriever(embed_fn=lambda t: [[0.0]])


# -- the retriever maps Azure results correctly (mocked SDK) ----------------
def _fake_embed(dim=8):
    return lambda texts: [[0.1] * dim for _ in texts]


def test_index_uploads_documents_with_vectors(monkeypatch):
    _dummy_search_env(monkeypatch)
    r = AzureSearchRetriever(index_name="idx", embedding_dim=8, embed_fn=_fake_embed(8))
    fake_sc = mock.MagicMock()
    monkeypatch.setattr(r, "_ensure_index", lambda: None)
    monkeypatch.setattr(r, "_search_client", lambda: fake_sc)

    docs = [Document(content="alpha", metadata={"company": "AAPL"}),
            Document(content="beta", metadata={"company": "MSFT"})]
    r.index(docs)

    assert fake_sc.upload_documents.called
    uploaded = fake_sc.upload_documents.call_args.kwargs["documents"]
    assert len(uploaded) == 2
    assert uploaded[0]["content"] == "alpha"
    assert len(uploaded[0]["content_vector"]) == 8
    assert json.loads(uploaded[0]["metadata_json"])["company"] == "AAPL"


def test_retrieve_maps_search_hits_to_results(monkeypatch):
    _dummy_search_env(monkeypatch)
    r = AzureSearchRetriever(index_name="idx", embedding_dim=8, embed_fn=_fake_embed(8))
    fake_sc = mock.MagicMock()
    fake_sc.search.return_value = [
        {"content": "risk factors ...", "metadata_json": '{"company": "AAPL"}',
         "@search.score": 0.87},
        {"content": "supply chain ...", "metadata_json": '{"company": "MSFT"}',
         "@search.score": 0.61},
    ]
    monkeypatch.setattr(r, "_search_client", lambda: fake_sc)

    # avoid importing the real azure models module in the mapping path
    fake_vq = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"azure.search.documents.models": mock.MagicMock(
            VectorizedQuery=lambda **kw: fake_vq)}):
        out = r.retrieve("what are the risks?", top_k=2)

    assert [type(x).__name__ for x in out] == ["RetrievalResult", "RetrievalResult"]
    assert out[0].content == "risk factors ..."
    assert out[0].score == pytest.approx(0.87)
    assert out[0].metadata["company"] == "AAPL"
    assert out[0].rank == 0 and out[1].rank == 1
    # hybrid on by default -> keyword search_text passed alongside the vector query
    assert fake_sc.search.call_args.kwargs["search_text"] == "what are the risks?"
