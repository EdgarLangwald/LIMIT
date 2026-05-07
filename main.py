import os
import pprint

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

from create_datasets import generate_steiner_dataset
from embed import embed_dataset
from evaluate import evaluate

MODEL  = "BAAI/bge-small-en-v1.5"
N      = 61   # N % 6 = 1 or 3

print(f"=== Pipeline test: steiner n={N}, model={MODEL} ===\n")

print("1. Building dataset...")
dataset, qrels = generate_steiner_dataset(n=N)
print(f"   {len(dataset['corpus'])} docs, {len(dataset['queries'])} queries\n")

print("2. Embedding...")
mapping = embed_dataset(dataset, MODEL, name=f"steiner_n{N}")
print()

print("3. Evaluating...")
results = evaluate(mapping, qrels, recall_at=[1], n_values=[20, 80, 250])
pprint.pprint(results[0])
