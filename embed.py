"""
Embedding with transparent disk cache.

On a GPU machine / cluster: embeddings are computed and saved to cache_dir.
On a CPU machine:           cached .npy files are loaded directly.
If the cache is cold and no GPU is available, SentenceTransformer will still
run (slowly) on CPU — or you can call this from prepare_embeddings.py first.
"""

import gc
import hashlib
import json
import os

import numpy as np


_DEFAULT_CACHE_DIR  = os.path.join(os.path.dirname(__file__), "embeddings")
_DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


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
