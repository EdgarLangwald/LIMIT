import os
import torch

print("CUDA available:", torch.cuda.is_available())

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

import torch
from create_datasets import build_preference_dataset
from embed import embed_dataset
from evaluate import evaluate_preference

M = 2
N = 1849 // 2
MODEL_KEYS = ["Qwen8b", "GritLM", "Promptriever", "E5_Mistral", "Qwen4b", "Qwen0.6b", "ModernBERT", "Snowflake_v2"]

print(f"Building preference dataset n={N}, m={M}...")
dataset, qrels, sentiments = build_preference_dataset(n=N, m=M)
print(f"  {len(dataset['corpus'])} docs, {len(dataset['queries'])} queries\n")

for MODEL_KEY in MODEL_KEYS:
    FILE_NAME = f"Preference dataset/{MODEL_KEY}"

    print(f"Embedding with {MODEL_KEY}...")
    mapping = embed_dataset(dataset, MODEL_KEY, dataset_name=FILE_NAME)
    print(f"  Done.\n")

    print("Evaluating...")
    results = evaluate_preference(
        doc_embs=mapping["doc_embs"],
        qry_embs=mapping["qry_embs"],
        qrels=qrels,
        sentiments=sentiments,
        ks=[1, 2],
        neutral_ks=[2, 5],
        file_name=FILE_NAME,
    )
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
