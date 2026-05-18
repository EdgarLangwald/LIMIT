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

from create_datasets import increase_param
from embed import embed_dataset
from evaluate import evaluate_increase_m

MODEL_KEYS = ["Qwen0.6b", "BGE_S"]
N     = 36
M_MAX = 50

# 1. Create datasets — one per m value (shared across models)
print(f"Building loi datasets  n={N}  m=1..{M_MAX}...")
results_raw = increase_param("build_loi_dataset", "m", range(1, M_MAX + 1), n=N)
datasets    = [r[0] for r in results_raw]
meta        = list(range(1, M_MAX + 1))

for model in MODEL_KEYS:
    # 2. Embed
    print(f"\nEmbedding with {model}...")
    mappings = embed_dataset(datasets, model, dataset_name=f"loi_n{N}", batch_size=64)

    # 3. Evaluate geometry
    print(f"Evaluating {model}...")
    evaluate_increase_m(
        mappings=mappings,
        meta=meta,
        k=5,
        file_name=f"increase_m_{model}_n{N}",
    )
