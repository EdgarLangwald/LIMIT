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

import numpy as np


_DEFAULT_CACHE_DIR  = os.path.join(os.path.dirname(__file__), "embeddings")
_DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-large-en-v1.5":              "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-mistral-7b-instruct":     "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "Snowflake/snowflake-arctic-embed-l":  "Represent this sentence for searching relevant passages: ",
}


def get_query_prefix(model_name: str) -> str:
    return QUERY_PREFIXES.get(model_name, "")


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

        doc_embs_all = embed(all_doc_texts, model_name, prefix="",           cache_dir=None, models_dir=models_dir, device=device, batch_size=batch_size)
        qry_embs_all = embed(all_qry_texts, model_name, prefix=query_prefix, cache_dir=None, models_dir=models_dir, device=device, batch_size=batch_size)

        text_to_doc = {t: doc_embs_all[j] for j, t in enumerate(all_doc_texts)}
        text_to_qry = {t: qry_embs_all[j] for j, t in enumerate(all_qry_texts)}

        rng = np.random.default_rng(seed)
        for i in missing:
            ds, fname = datasets[i], file_names[i]
            doc_ids   = list(ds["corpus"].keys())
            query_ids = list(ds["queries"].keys())
            rng.shuffle(doc_ids)

            d_embs = np.stack([text_to_doc[ds["corpus"][d]]  for d in doc_ids])
            q_embs = np.stack([text_to_qry[ds["queries"][q]] for q in query_ids])

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
