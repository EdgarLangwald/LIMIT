"""
Embedding with transparent disk cache.

On a GPU machine / cluster: embeddings are computed and saved to cache_dir.
On a CPU machine:           cached .npy files are loaded directly.
If the cache is cold and no GPU is available, SentenceTransformer will still
run (slowly) on CPU — or pre-compute with main.py first.
"""

import gc
import hashlib
import json
import os

import numpy as np


_DEFAULT_CACHE_DIR  = os.path.join(os.path.dirname(__file__), "embeddings")
_DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Query-side instruction prefixes, keyed by HuggingFace model ID.
# Documents are always embedded without a prefix.
QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-large-en-v1.5":              "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-mistral-7b-instruct":     "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "Snowflake/snowflake-arctic-embed-l":  "Represent this sentence for searching relevant passages: ",
}


def get_query_prefix(model_name: str) -> str:
    """Return the query-side instruction prefix for model_name, or '' if none needed."""
    return QUERY_PREFIXES.get(model_name, "")


def embed_dataset(
    dataset: dict,
    model_name: str,
    query_prefix: str = "",
    cache_dir:  str = _DEFAULT_CACHE_DIR,
    models_dir: str = _DEFAULT_MODELS_DIR,
    device:     str | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed corpus and queries from a standard dataset dict.

    Args:
        dataset:      dict with keys "corpus" ({id: text}) and "queries" ({id: text}).
        query_prefix: Prepended to every query text (model-specific instruction).
                      Use get_query_prefix(model_name) to look it up automatically.

    Returns:
        (doc_embs, query_embs) — shapes (n_docs, d) and (n_queries, d), float32.
    """
    doc_embs = embed(
        list(dataset["corpus"].values()), model_name, prefix="",
        cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size,
    )
    qry_embs = embed(
        list(dataset["queries"].values()), model_name, prefix=query_prefix,
        cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size,
    )
    return doc_embs, qry_embs


def embed(
    texts: list[str],
    model_name: str,
    prefix: str = "",
    cache_dir:  str = _DEFAULT_CACHE_DIR,
    models_dir: str = _DEFAULT_MODELS_DIR,
    device:     str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Return (len(texts), d) float32 embeddings, loading from cache when available.

    Args:
        texts:      Raw texts to embed (without prefix).
        model_name: HuggingFace model ID or local alias.
        prefix:     Prepended to every text before encoding (e.g. BGE query prefix).
                    Included in the cache key so doc/query caches never collide.
        cache_dir:  Directory for .npy cache files.
        models_dir: Directory where downloaded models are saved.
        device:     "cuda", "cpu", or None (auto-detect).
        batch_size: Encoding batch size.
    """
    os.makedirs(cache_dir, exist_ok=True)

    payload = json.dumps({"model": model_name, "prefix": prefix, "texts": texts}, sort_keys=True)
    key = hashlib.md5(payload.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{key}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    local_path = os.path.join(models_dir, model_name.replace("/", "_"))
    if os.path.isdir(local_path):
        model = SentenceTransformer(local_path, device=device)
    else:
        model = SentenceTransformer(model_name, device=device)
        os.makedirs(models_dir, exist_ok=True)
        model.save(local_path)

    prefixed = [prefix + t for t in texts] if prefix else texts
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embs = np.array(embs, dtype=np.float32)
    np.save(cache_path, embs)

    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embs
