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
from create_datasets import generate_steiner_dataset
from embed import embed_dataset, MODELS

print("CUDA available:", torch.cuda.is_available())

N = 49
MODEL_KEYS = [
    "Qwen0.6b"
]

print(f"Building steiner dataset n={N}...")
dataset, qrels = generate_steiner_dataset(n=N)
print(f"  {len(dataset['corpus'])} docs, {len(dataset['queries'])} queries\n")

for key in MODEL_KEYS:
    print(f"Embedding with {key}...")
    embed_dataset(dataset, key, dataset_name="steiner_FULL", batch_size=64)
    print(f"  Done.\n")
