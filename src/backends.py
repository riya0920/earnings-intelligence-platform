"""
Backend switch: run the LLM/generation and retrieval on either the default
local/OpenAI stack or an Azure stack (Azure OpenAI + Azure AI Search), chosen at
runtime. The existing path is untouched and remains the default.

Selection order (first that resolves wins):
    1. env  EIP_BACKEND = local | azure
    2. config  backend: local | azure   (configs/default.yaml)
    3. "local"

On Azure the OpenAI SDK's `AzureOpenAI` client is a drop-in for `OpenAI` -- same
`.chat.completions.create(...)` and `.embeddings.create(...)` surface -- with one
catch this module handles for callers: the `model=` argument must be the Azure
*deployment* name, not the model name. Use `chat_model()` / `embed_model()` to get
the right string for the active backend.

Secrets live in the environment, never in code or config:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBED_DEPLOYMENT
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX
"""
from __future__ import annotations

import os

LOCAL = "local"
AZURE = "azure"

_DEFAULT_API_VERSION = "2024-06-01"


def active_backend(config: dict | None = None) -> str:
    """Return 'local' or 'azure'."""
    env = os.getenv("EIP_BACKEND")
    if env:
        val = env.strip().lower()
    elif config and config.get("backend"):
        val = str(config["backend"]).strip().lower()
    else:
        val = LOCAL
    if val not in (LOCAL, AZURE):
        raise ValueError(f"EIP_BACKEND must be 'local' or 'azure', got {val!r}")
    return val


def is_azure(config: dict | None = None) -> bool:
    return active_backend(config) == AZURE


# --------------------------------------------------------------------------
# Chat / generation client
# --------------------------------------------------------------------------
def get_chat_client(config: dict | None = None):
    """OpenAI() on local, AzureOpenAI(...) on azure. Same call surface."""
    if is_azure(config):
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
            api_key=_require("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", _DEFAULT_API_VERSION),
        )
    from openai import OpenAI
    return OpenAI()


def chat_model(default: str, config: dict | None = None) -> str:
    """The string to pass as `model=`. On Azure that is the deployment name."""
    if is_azure(config):
        return os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", default)
    return default


# --------------------------------------------------------------------------
# Embeddings client (used by the Azure AI Search retriever)
# --------------------------------------------------------------------------
def get_embeddings_client(config: dict | None = None):
    """Same object as the chat client; separated so callers read clearly."""
    return get_chat_client(config)


def embed_model(default: str = "text-embedding-3-small",
                config: dict | None = None) -> str:
    if is_azure(config):
        return os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", default)
    return default


def azure_openai_embed(texts: list[str], config: dict | None = None) -> list[list[float]]:
    """Embed a batch of texts with Azure OpenAI (or OpenAI) embeddings."""
    client = get_embeddings_client(config)
    model = embed_model(config=config)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _require(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(
            f"{var} is not set. The Azure backend needs it -- see the "
            f"'Azure backend' section of the README for the full list."
        )
    return val
