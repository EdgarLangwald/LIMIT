## Overview

The goal of this project is to stress test embedding models in multiple ways and see when and how they break. The code can broadly be split into three components that each have multiple interchangeable implementations: Dataset creation -> Embedding -> Evaluation. This way, multiple tests can be done on multiple models. The embedding has to be able to run locally and on a cluster. And different metrics have to be able to evaluate performance without recomputing embeddings every time.

## Dataset Basics

LIMIT paper provides 1848 unique items and >= 1000 american names and surnames. The embeddings consist of list of items (loi) e.g. "John Smith likes apples, winter and cats". Queries ask for specific items.


## Project structure

**Python modules**
- `main.py` — CLI entry point; pre-computes and caches embeddings for any registered dataset
- `create_datasets.py` — all dataset creation functions (`build_disjoint_dataset`, `generate_k_shared_dataset`, `generate_steiner_dataset`)
- `embed.py` — embedding with transparent MD5 disk cache; `QUERY_PREFIXES` as single source of truth
- `evaluate.py` — evaluation functions (`eval_item_retrieval`, `eval_embed_distance`, `eval_retrieval_vs_n`) and plot helpers
- `name_item_pool.py` — loads `data/pool.json` (items, first names, surnames) via `load_pool()`

**Notebooks**
- `main.ipynb` — interactive experimentation and plotting

**Folders**
- `data/` — `pool.json` (1848 items, 2738 first names, 1000 surnames)
- `embeddings/` — MD5-keyed `.npy` embedding cache files
- `models/` — downloaded SentenceTransformer model weights

## Coding practices

**IMPORTANT:** Run all scripts from C:\OneDrive\Documents\Coding\.venv this venv. Also pip install into there.
Add to and edit Project structure that's **relevant for coding** as you work on this project, so it stays relevant.

## Other

Path to LIMIT paper pdf "C:\OneDrive\Documents\Papers\LIMIT DeepMind.pdf"
URL to LIMIT repo: https://github.com/google-deepmind/limit
This Claude Code session is being run from windows powershell
Documentation links for RWTH Aachen Cluster:
- Submit Jobs: https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/13ace46cfbb84e92a64c1361e0e4c104/
- Job Parameters: https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/3d20a87835db4569ad9094d91874e2b4/
- Job Management: https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/85b21b312bfb48b290043083d2a34b8f/