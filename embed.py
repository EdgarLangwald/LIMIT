"""
Embedding with structured per-dataset cache.

Cache layout (single):   embeddings/{experiment_name}/{model_name}_d.npy  /  _q.npy
Cache layout (multiple): embeddings/{experiment_name}/{model_name}/{i}_d.npy  /  _q.npy
"""

import gc
import json
import os

import numpy as np
import torch
from tqdm import tqdm

_DEFAULT_CACHE_DIR  = os.path.join(os.path.dirname(__file__), "embeddings")
_DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS: dict[str, dict] = {
    # ~0.1 GB; fast baseline, decent MTEB score
    "BGE_S": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    # ~1.3 GB; strong MTEB ~2023, good general baseline
    "BGE_L": {
        "hf_id": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    # ~0.5 GB; ModernBERT backbone, strong MTEB for its size (2024)
    "ModernBERT": {
        "hf_id": "Alibaba-NLP/gte-modernbert-base",
        "query_prefix": "query: ",
    },
    # ~1.2 GB; SOTA-tier small model, strong MTEB for size (2025)
    "Qwen0.6b": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "query_prefix": "query: ",
    },
    # ~8 GB; near SOTA on MTEB (2025)
    "Qwen4b": {
        "hf_id": "Qwen/Qwen3-Embedding-4B",
        "query_prefix": "query: ",
    },
    # ~16 GB; SOTA on MTEB at release (2025)
    "Qwen8b": {
        "hf_id": "Qwen/Qwen3-Embedding-8B",
        "query_prefix": "query: ",
    },
    # ~14 GB; was SOTA on MTEB at release (2023)
    "E5_Mistral": {
        "hf_id": "intfloat/e5-mistral-7b-instruct",
        "query_prefix": "Instruct: Retrieve the person whose profile contains the queried item.\nQuery: ",
    },
    # ~14 GB; unified embedding+generation, strong MTEB (2024)
    "GritLM": {
        "hf_id": "GritLM/GritLM-7B",
        "query_prefix": "<|user|>\nRetrieve the person whose profile contains the queried item.\n<|embed|>\n",
    },
    # ~16 GB; instruction-following retrieval based on Llama 3.1 (2024)
    "Promptriever": {
        "hf_id": "samaya-ai/promptriever-llama3.1-8b-instruct-v1",
        "query_prefix": "Retrieve the person whose profile contains the queried item.\n\n",
    },
    # ~16 GB; #1 MTEB late 2024, instruction-following with latent attention layer
    "NV_Embed": {
        "hf_id": "nvidia/NV-Embed-v2",
        "query_prefix": "Instruct: Retrieve the person whose profile contains the queried item.\nQuery: ",
    },
    # ~14 GB; strong MTEB 2024, instruction-following, Qwen2 backbone
    "GTE_Qwen2": {
        "hf_id": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "query_prefix": "Instruct: Retrieve the person whose profile contains the queried item.\nQuery: ",
    },
    # ~1 GB; MoE architecture, 475M params (305M active), strong for size (2024)
    "Nomic_MoE": {
        "hf_id": "nomic-ai/nomic-embed-text-v2-moe",
        "query_prefix": "search_query: ",
    },
    # ~2 GB; updated Arctic-L, stronger MTEB than v1 (2024)
    "Snowflake_v2": {
        "hf_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "query_prefix": "query: ",
    },
}

def _progress_path(p: str) -> str:
    return p + "_progress.json"

def _load_progress(p: str) -> dict | None:
    pp = _progress_path(p)
    if os.path.isfile(pp):
        with open(pp) as f:
            return json.load(f)
    return None

def _save_progress(p: str, data: dict) -> None:
    with open(_progress_path(p), "w") as f:
        json.dump(data, f)

def _clear_progress(p: str) -> None:
    pp = _progress_path(p)
    if os.path.isfile(pp):
        os.remove(pp)

def _is_complete(p: str) -> bool:
    return (
        os.path.isfile(p + "_d.npy")
        and os.path.isfile(p + "_q.npy")
        and not os.path.isfile(p + "_progress.json")
    )


def get_query_prefix(model_name: str) -> str:
    return MODELS[model_name]["query_prefix"]

def get_model_id(model_name: str) -> str:
    return MODELS[model_name]["hf_id"]


def _load_dataset(base: str) -> dict:
    return {
        "doc_embs": np.load(base + "_d.npy", mmap_mode="r"),
        "qry_embs": np.load(base + "_q.npy", mmap_mode="r"),
    }


def embed_dataset(
    datasets: list[dict] | dict,
    model_name: str,
    dataset_name: str,
    force: bool = False,
    batch_size: int = 64,
    device: str | None = None,
) -> list[dict] | dict:
    """
    Embed one dataset or a list of datasets.

    Returns {"doc_embs": ndarray, "qry_embs": ndarray} or a list of these.

    force=False: load from disk if present, embed and save if not.
    force=True:  always re-embed and overwrite.
    """
    single  = isinstance(datasets, dict)
    ds_list = [datasets] if single else datasets

    folder = (
        os.path.join(_DEFAULT_CACHE_DIR, dataset_name)
        if single else
        os.path.join(_DEFAULT_CACHE_DIR, dataset_name, model_name)
    )
    paths = (
        [os.path.join(folder, model_name)]
        if single else
        [os.path.join(folder, str(i)) for i in range(len(ds_list))]
    )

    missing = list(range(len(ds_list))) if force else [i for i, p in enumerate(paths) if not _is_complete(p)]

    n_cached = len(ds_list) - len(missing)
    if missing:
        cached_note = f" ({n_cached} cached)" if n_cached else ""
        print(f"Embedding {len(missing)}/{len(ds_list)} datasets{cached_note}")
    else:
        print(f"Loading {len(ds_list)} cached dataset(s)")

    if missing:
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        from sentence_transformers import SentenceTransformer
        from numpy.lib.format import open_memmap

        os.makedirs(folder, exist_ok=True)
        model_id         = get_model_id(model_name)
        model_local_path = os.path.join(_DEFAULT_MODELS_DIR, model_name)
        use_device       = device or _device
        if os.path.isdir(model_local_path):
            print(f"  model: {model_name} (local)")
            model = SentenceTransformer(model_local_path, device=use_device)
        else:
            print(f"  model: {model_name} (downloading {model_id})")
            model = SentenceTransformer(model_id, device=use_device)
            model.save(model_local_path)

        query_prefix = get_query_prefix(model_name)
        dim = model.get_sentence_embedding_dimension()

        for i in missing:
            ds        = ds_list[i]
            p         = paths[i]
            doc_texts = list(ds["corpus"].values())
            qry_texts = list(ds["queries"].values())

            progress  = None if force else _load_progress(p)
            # docs are done if: no progress file (clean between-phase state) or progress says so
            docs_done = (not force) and (progress is None or progress.get("docs_done", False)) and os.path.isfile(p + "_d.npy")
            doc_start = progress.get("doc_start", 0) if (progress and not docs_done) else 0
            qry_start = progress.get("qry_start", 0) if (progress and docs_done) else 0

            n_doc_batches = (len(doc_texts) + batch_size - 1) // batch_size
            n_qry_batches = (len(qry_texts) + batch_size - 1) // batch_size

            doc_mm = open_memmap(p + "_d.npy", dtype="float32", mode="r+" if docs_done else "w+", shape=(len(doc_texts), dim))

            if not docs_done:
                for start in tqdm(range(doc_start, len(doc_texts), batch_size),
                                  desc=f"[{i}] docs",
                                  initial=doc_start // batch_size,
                                  total=n_doc_batches):
                    batch = doc_texts[start : start + batch_size]
                    doc_mm[start : start + len(batch)] = embed(batch, model, prefix="", batch_size=batch_size)
                    _save_progress(p, {"docs_done": False, "doc_start": start + len(batch)})
                doc_mm.flush()
                _clear_progress(p)

            qry_exists = os.path.isfile(p + "_q.npy")
            qry_mm = open_memmap(p + "_q.npy", dtype="float32", mode="r+" if qry_exists else "w+", shape=(len(qry_texts), dim))

            for start in tqdm(range(qry_start, len(qry_texts), batch_size),
                              desc=f"[{i}] queries",
                              initial=qry_start // batch_size,
                              total=n_qry_batches):
                batch = qry_texts[start : start + batch_size]
                qry_mm[start : start + len(batch)] = embed(batch, model, prefix=query_prefix, batch_size=batch_size)
                _save_progress(p, {"docs_done": True, "qry_start": start + len(batch)})

            qry_mm.flush()
            del doc_mm, qry_mm
            _clear_progress(p)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = [_load_dataset(p) for p in paths]
    return results[0] if single else results


def embed(
    texts: list[str],
    model,
    prefix: str = "",
    batch_size: int = 64,
) -> np.ndarray:
    """Return (len(texts), dim) float32 embeddings using a pre-loaded model."""
    prefixed = [prefix + t for t in texts] if prefix else texts
    return np.array(
        model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
