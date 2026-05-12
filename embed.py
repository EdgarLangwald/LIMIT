"""
Embedding with structured per-dataset cache.

Cache layout (single):   embeddings/{experiment_name}/{model_name}.npz
Cache layout (multiple): embeddings/{experiment_name}/{model_name}/{i}.npz
Each .npz contains doc_ids, query_ids, doc_embs, qry_embs for one dataset.
"""

import gc
import os

import numpy as np
import torch

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

def get_query_prefix(model_name: str) -> str:
    return MODELS[model_name]["query_prefix"]

def get_model_id(model_name: str) -> str:
    return MODELS[model_name]["hf_id"]


def _load_dataset(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    doc_ids = list(data["doc_ids"])
    qry_ids = list(data["query_ids"])
    return {
        "doc_map":  {d: i for i, d in enumerate(doc_ids)},
        "qry_map":  {q: i for i, q in enumerate(qry_ids)},
        "doc_embs": data["doc_embs"],
        "qry_embs": data["qry_embs"],
    }


def embed_dataset(
    datasets: list[dict] | dict,
    model_name: str,
    dataset_name: str,
    cache: bool = False,
    batch_size: int = 64,
    device: str | None = None,
) -> list[dict] | dict:
    """
    Embed one dataset or a list of datasets.

    Returns {"doc_map": {id: idx}, "qry_map": {id: idx}, "doc_embs": ndarray, "qry_embs": ndarray}
    or a list of these.

    cache=False: load from disk if present, embed and save if not.
    cache=True:  always re-embed and overwrite.
    """
    single  = isinstance(datasets, dict)
    ds_list = [datasets] if single else datasets

    folder = (
        os.path.join(_DEFAULT_CACHE_DIR, dataset_name)
        if single else
        os.path.join(_DEFAULT_CACHE_DIR, dataset_name, model_name)
    )
    paths = (
        [os.path.join(folder, f"{model_name}.npz")]
        if single else
        [os.path.join(folder, f"{i}.npz") for i in range(len(ds_list))]
    )

    missing = list(range(len(ds_list))) if cache else [i for i, p in enumerate(paths) if not os.path.exists(p)]

    if missing:
        os.makedirs(folder, exist_ok=True)

        from sentence_transformers import SentenceTransformer
        model_id         = get_model_id(model_name)
        model_local_path = os.path.join(_DEFAULT_MODELS_DIR, model_name)
        use_device       = device or _device
        if os.path.isdir(model_local_path):
            model = SentenceTransformer(model_local_path, device=use_device)
        else:
            model = SentenceTransformer(model_id, device=use_device)
            model.save(model_local_path)

        query_prefix = get_query_prefix(model_name)

        for i in missing:
            ds      = ds_list[i]
            doc_ids = list(ds["corpus"].keys())
            qry_ids = list(ds["queries"].keys())
            doc_embs = embed(list(ds["corpus"].values()), model, prefix="",           batch_size=batch_size)
            qry_embs = embed(list(ds["queries"].values()), model, prefix=query_prefix, batch_size=batch_size)
            np.savez(
                paths[i],
                doc_ids=np.array(doc_ids, dtype=object),
                query_ids=np.array(qry_ids, dtype=object),
                doc_embs=doc_embs,
                qry_embs=qry_embs,
            )

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
        model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True),
        dtype=np.float32,
    )
