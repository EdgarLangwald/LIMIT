import os

def _load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env_file()

import numpy as np
from create_datasets import generate_steiner_dataset
from embed import embed_dataset
from evaluate import evaluate, plot_results

N = 1849
MODEL_KEYS = ["Qwen8b", "GritLM"]

print(f"Building steiner dataset n={N}...")
dataset, qrels = generate_steiner_dataset(n=N)
n_docs = len(dataset["corpus"])
n_queries = len(dataset["queries"])
print(f"  {n_docs} docs, {n_queries} queries\n")

for key in MODEL_KEYS:
    print(f"Loading embeddings for {key}...")
    embs = embed_dataset(dataset, key, dataset_name="steiner_FULL")
    doc_embs = embs["doc_embs"]
    qry_embs = embs["qry_embs"]

    print(f"Evaluating {key}...")
    results = evaluate(
        doc_embs=doc_embs,
        qry_embs=qry_embs,
        qrels=qrels,
        n_values=[2500, 7000, 21000, 65000, 190000, 569492],
        q_bs=4000,
        ks=[2, 5, 10, 50, 200, 1000]
    )

    plot_results(list(results.values()), list(results.keys()), model_name=key, show=False)
