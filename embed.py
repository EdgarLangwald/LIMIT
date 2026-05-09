"""
Embedding with structured per-dataset cache.

Cache layout:  embeddings/{experiment_name}/{model_slug}/{file}.npz
Each .npz contains doc_ids, query_ids, doc_embs, qry_embs for one dataset.
Doc ordering is shuffled with a fixed seed so any doc_ids[:n] prefix is a
random-looking sample from the full dataset — required for the n_values sweep.
"""

import gc
import hashlib
import json
import os
import tempfile

import numpy as np


_DEFAULT_CACHE_DIR  = os.path.join(os.path.dirname(__file__), "embeddings")
_DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODELS: dict[str, dict] = {
    # ~0.1 GB; fast baseline, decent MTEB score
    "BGE-small": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    # ~1.3 GB; strong MTEB ~2023, good general baseline
    "BGE-large": {
        "hf_id": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    # ~0.5 GB; ModernBERT backbone, strong MTEB for its size (2024)
    "GTE-ModernBERT": {
        "hf_id": "Alibaba-NLP/gte-modernbert-base",
        "query_prefix": "query: ",
    },
    # ~2 GB; near SOTA on MTEB at release (2024)
    "Snowflake-Arctic-L": {
        "hf_id": "Snowflake/snowflake-arctic-embed-l",
        "query_prefix": "",
    },
    # ~1.2 GB; SOTA-tier small model, strong MTEB for size (2025)
    "Qwen3-Embedding-0.6B": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "query_prefix": "query: ",
    },
    # ~8 GB; near SOTA on MTEB (2025)
    "Qwen3-Embedding-4B": {
        "hf_id": "Qwen/Qwen3-Embedding-4B",
        "query_prefix": "query: ",
    },
    # ~16 GB; SOTA on MTEB at release (2025)
    "Qwen3-Embedding-8B": {
        "hf_id": "Qwen/Qwen3-Embedding-8B",
        "query_prefix": "query: ",
    },
    # ~14 GB; was SOTA on MTEB at release (2023)
    "E5-Mistral-7B": {
        "hf_id": "intfloat/e5-mistral-7b-instruct",
        "query_prefix": "Instruct: Retrieve the person whose profile contains the queried item.\nQuery: ",
    },
    # ~14 GB; unified embedding+generation, strong MTEB (2024)
    "GritLM-7B": {
        "hf_id": "GritLM/GritLM-7B",
        "query_prefix": "<|user|>\nRetrieve the person whose profile contains the queried item.\n<|embed|>\n",
    },
    # ~16 GB; instruction-following retrieval based on Llama 3.1 (2024)
    "Promptriever-Llama3-8B": {
        "hf_id": "samaya-ai/promptriever-llama3.1-8b-instruct-v1",
        "query_prefix": "Retrieve the person whose profile contains the queried item.\n\n",
    },
    # ~16 GB; #1 MTEB late 2024, instruction-following with latent attention layer
    "NV-Embed-v2": {
        "hf_id": "nvidia/NV-Embed-v2",
        "query_prefix": "Instruct: Retrieve the person whose profile contains the queried item.\nQuery: ",
    },
    # ~14 GB; strong MTEB 2024, instruction-following, Qwen2 backbone
    "GTE-Qwen2-7B": {
        "hf_id": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "query_prefix": "Instruct: Retrieve the person whose profile contains the queried item.\nQuery: ",
    },
    # ~1 GB; MoE architecture, 475M params (305M active), strong for size (2024)
    "Nomic-Embed-v2-MoE": {
        "hf_id": "nomic-ai/nomic-embed-text-v2-moe",
        "query_prefix": "search_query: ",
    },
    # ~2 GB; updated Arctic-L, stronger MTEB than v1 (2024)
    "Snowflake-Arctic-L-v2": {
        "hf_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "query_prefix": "query: ",
    },
}


def get_query_prefix(model_name: str) -> str:
    for m in MODELS.values():
        if m["hf_id"] == model_name:
            return m["query_prefix"]
    return ""


def embed_dataset(
    dataset_or_list,
    model_name: str,
    name: str,
    cache: bool = True,
    meta: list | None = None,
    query_prefix: str | None = None,
    batch_size: int = 64,
    cache_dir:  str = _DEFAULT_CACHE_DIR,
    models_dir: str = _DEFAULT_MODELS_DIR,
    device:     str | None = None,
    seed: int = 42,
) -> dict | list[dict]:
    """
    Embed one dataset or a list of datasets, returning structured mappings.

    Each mapping is {"docs": {id: emb}, "queries": {id: emb}}.
    Doc ordering is shuffled (seed) before caching so doc_ids[:n] looks like
    a random sample — consistent between cache loads and fresh computation.

    Cache: {cache_dir}/{name}/{model_slug}/{fname}.npz per dataset.
    For a list, fnames come from meta (e.g. [1,2,...,m_max] → "1.npz", "2.npz").
    """
    single   = isinstance(dataset_or_list, dict)
    datasets = [dataset_or_list] if single else dataset_or_list
    query_prefix = query_prefix or get_query_prefix(model_name)
    model_slug   = model_name.replace("/", "_")
    file_names   = [name] if single else [str(m) for m in (meta or range(len(datasets)))]

    folder = os.path.join(cache_dir, name, model_slug)
    if cache:
        os.makedirs(folder, exist_ok=True)

    # Load cached results where available; mark the rest for computation
    loaded  = [None] * len(datasets)
    missing = []
    for i, fname in enumerate(file_names):
        path = os.path.join(folder, f"{fname}.npz")
        if cache and os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            loaded[i] = {
                "doc_ids":   list(data["doc_ids"]),
                "query_ids": list(data["query_ids"]),
                "doc_embs":  data["doc_embs"],
                "qry_embs":  data["qry_embs"],
            }
        else:
            missing.append(i)

    # Embed all missing datasets in one pass
    if missing:
        missing_ds    = [datasets[i] for i in missing]
        all_doc_texts = list({t for ds in missing_ds for t in ds["corpus"].values()})
        all_qry_texts = list({t for ds in missing_ds for t in ds["queries"].values()})

        # Phase 1: embed docs, save per-dataset doc arrays to temp files, then free
        doc_embs_all = embed(all_doc_texts, model_name, prefix="", cache_dir=None, models_dir=models_dir, device=device, batch_size=batch_size)
        doc_idx = {t: j for j, t in enumerate(all_doc_texts)}

        rng = np.random.default_rng(seed)
        tmp_paths = {}
        for i in missing:
            ds = datasets[i]
            doc_ids = list(ds["corpus"].keys())
            rng.shuffle(doc_ids)
            d_embs = np.stack([doc_embs_all[doc_idx[ds["corpus"][d]]] for d in doc_ids])
            fd, path = tempfile.mkstemp(suffix=".npz")
            os.close(fd)
            np.savez(path, doc_ids=np.array(doc_ids, dtype=object), d_embs=d_embs)
            tmp_paths[i] = path

        del doc_embs_all
        gc.collect()

        # Phase 2: embed queries, load doc arrays from temp, save final npz
        qry_embs_all = embed(all_qry_texts, model_name, prefix=query_prefix, cache_dir=None, models_dir=models_dir, device=device, batch_size=batch_size)
        qry_idx = {t: j for j, t in enumerate(all_qry_texts)}

        for i in missing:
            ds, fname = datasets[i], file_names[i]
            query_ids = list(ds["queries"].keys())
            q_embs = np.stack([qry_embs_all[qry_idx[ds["queries"][q]]] for q in query_ids])

            tmp_data = np.load(tmp_paths[i], allow_pickle=True)
            doc_ids = list(tmp_data["doc_ids"])
            d_embs  = tmp_data["d_embs"].copy()
            tmp_data.close()
            os.unlink(tmp_paths[i])

            loaded[i] = {"doc_ids": doc_ids, "query_ids": query_ids, "doc_embs": d_embs, "qry_embs": q_embs}

            if cache:
                np.savez(
                    os.path.join(folder, f"{fname}.npz"),
                    doc_ids=np.array(doc_ids, dtype=object),
                    query_ids=np.array(query_ids, dtype=object),
                    doc_embs=d_embs,
                    qry_embs=q_embs,
                )

    mappings = [
        {
            "docs":    {d: loaded[i]["doc_embs"][j]  for j, d in enumerate(loaded[i]["doc_ids"])},
            "queries": {q: loaded[i]["qry_embs"][j]  for j, q in enumerate(loaded[i]["query_ids"])},
        }
        for i in range(len(datasets))
    ]
    return mappings[0] if single else mappings


def embed(
    texts: list[str],
    model_name: str,
    prefix: str = "",
    cache_dir:  str | None = _DEFAULT_CACHE_DIR,
    models_dir: str = _DEFAULT_MODELS_DIR,
    device:     str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Return (len(texts), d) float32 embeddings.
    Pass cache_dir=None to skip disk caching (used internally by embed_dataset).
    """
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        payload    = json.dumps({"model": model_name, "prefix": prefix, "texts": texts}, sort_keys=True)
        cache_path = os.path.join(cache_dir, f"{hashlib.md5(payload.encode()).hexdigest()}.npy")
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
    embs = np.array(
        model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True),
        dtype=np.float32,
    )

    if cache_path is not None:
        np.save(cache_path, embs)

    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embs
