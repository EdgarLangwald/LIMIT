"""
Pre-compute and cache all embeddings needed by evaluate.py.

Run this on a GPU machine (cluster or local) before running evaluate.py
on a CPU machine. Then copy the embeddings/ folder back.

Usage:
    python prepare_embeddings.py --model intfloat/e5-mistral-7b-instruct
    python prepare_embeddings.py --model BAAI/bge-large-en-v1.5 --n 100
    python prepare_embeddings.py --model Snowflake/snowflake-arctic-embed-l --device cuda
"""

import argparse

from custom_dataset_creation import (
    _load_notebook_globals,
    generate_disjoint_dataset,
    generate_k_shared_dataset,
    generate_names,
)
from embed import embed, _DEFAULT_CACHE_DIR, _DEFAULT_MODELS_DIR

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def prepare_disjoint(n: int, m_max: int, model_name: str, cache_dir: str, models_dir: str, device: str | None, batch_size: int) -> None:
    names = generate_names(n)
    for m in range(1, m_max + 1):
        dataset = generate_disjoint_dataset(n=n, m=m, names=names)
        doc_texts   = list(dataset["corpus"].values())
        query_texts = list(dataset["queries"].values())
        print(f"  m={m:>3}: {len(doc_texts)} docs, {len(query_texts)} queries")
        embed(doc_texts,   model_name, prefix="",           cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size)
        embed(query_texts, model_name, prefix=QUERY_PREFIX, cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for all experiments.")
    parser.add_argument("--model",      required=True,              help="HuggingFace model ID")
    parser.add_argument("--n",          type=int, default=100,      help="Number of persons (documents)")
    parser.add_argument("--m-max",      type=int, default=None,     help="Max LOI length (default: pool_size // n)")
    parser.add_argument("--cache-dir",  default=_DEFAULT_CACHE_DIR, help="Where to save .npy cache files")
    parser.add_argument("--models-dir", default=_DEFAULT_MODELS_DIR,help="Where to save downloaded models")
    parser.add_argument("--device",     default=None,               help="cuda / cpu / None (auto)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    pool_size = len(_load_notebook_globals()[0])
    m_max = args.m_max or pool_size // args.n

    print(f"Model     : {args.model}")
    print(f"n={args.n}, m_max={m_max}, device={args.device or 'auto'}")
    print(f"Cache dir : {args.cache_dir}")

    prepare_disjoint(args.n, m_max, args.model, args.cache_dir, args.models_dir, args.device, args.batch_size)
    print("Done.")


if __name__ == "__main__":
    main()
