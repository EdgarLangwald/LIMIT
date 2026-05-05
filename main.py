"""
Pre-compute and cache embeddings for any registered dataset.

Run on a GPU machine (or cluster) before running evaluate.py on CPU.
Copy the embeddings/ folder back afterward.

Usage:
    python main.py --model BAAI/bge-large-en-v1.5 --dataset disjoint --n 100 --m-max 18
    python main.py --model BAAI/bge-large-en-v1.5 --dataset steiner --n 1849
    python main.py --model BAAI/bge-large-en-v1.5 --dataset k_shared --n 100 --m 3 --k 2

Dataset kwargs are forwarded directly to the creation function.
"""

import argparse

from create_datasets import build_disjoint_dataset, generate_steiner_dataset, generate_k_shared_dataset
from embed import embed_dataset, get_query_prefix, _DEFAULT_CACHE_DIR, _DEFAULT_MODELS_DIR

DATASETS = {
    "disjoint": build_disjoint_dataset,
    "steiner":  generate_steiner_dataset,
    "k_shared": generate_k_shared_dataset,
}


def _parse_kwargs(extras: list[str]) -> dict:
    """Convert ['--n', '100', '--m-max', '18'] into {'n': 100, 'm_max': 18}."""
    kwargs: dict = {}
    it = iter(extras)
    for token in it:
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            val = next(it, None)
            if val is not None:
                try:
                    kwargs[key] = int(val)
                except ValueError:
                    try:
                        kwargs[key] = float(val)
                    except ValueError:
                        kwargs[key] = val
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,               help="HuggingFace model ID")
    parser.add_argument("--dataset",    required=True, choices=list(DATASETS))
    parser.add_argument("--cache-dir",  default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--models-dir", default=_DEFAULT_MODELS_DIR)
    parser.add_argument("--device",     default=None,                help="cuda / cpu / None (auto)")
    parser.add_argument("--batch-size", type=int, default=64)
    args, extras = parser.parse_known_args()

    dataset_kwargs = _parse_kwargs(extras)
    query_prefix = get_query_prefix(args.model)

    print(f"Model:   {args.model}  |  prefix: {query_prefix!r}")
    print(f"Dataset: {args.dataset}  kwargs: {dataset_kwargs}")

    dataset = DATASETS[args.dataset](**dataset_kwargs)
    print(f"  {len(dataset['corpus']):,} docs, {len(dataset['queries']):,} queries")

    embed_dataset(dataset, args.model, query_prefix=query_prefix,
                  cache_dir=args.cache_dir, models_dir=args.models_dir,
                  device=args.device, batch_size=args.batch_size)
    print("Done.")


if __name__ == "__main__":
    main()
