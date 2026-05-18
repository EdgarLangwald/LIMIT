## Overview

The goal of this project is to stress test embedding models in multiple ways and see when and how they break. The code can broadly be split into three components that each have multiple interchangeable implementations: Dataset creation -> Embedding -> Evaluation. This way, multiple tests can be done on multiple models. The embedding has to be able to run locally and on a cluster. And different metrics have to be able to evaluate performance without recomputing embeddings every time.

## Dataset Basics

LIMIT paper provides 1848 unique items and >= 1000 american names and surnames. The embeddings consist of list of items (loi) e.g. "John Smith likes apples, winter and cats". Queries ask for specific items.


## Project structure

**Python modules**
- `main.py` — Script that gets executed for any experiments from Claude Code or CLI.
- `create_datasets.py` — dataset builders: `build_loi_dataset`, `build_preference_dataset`, `build_steiner_dataset`, `increase_param`; `build_k_shared_dataset` exists but is **not used**
- `embed.py` — embedding with structured per-dataset disk cache; `MODELS` dict is single source of truth for model IDs and query prefixes
- `evaluate.py` — evaluation functions (`evaluate`, `eval_embed_distance`, `evaluate_preference`) and plot helpers (`plot_results`, `plot_embed_distance`)
- `name_item_pool.py` — loads `data/pool.json` (items, first names, surnames) via `load_pool()`

**Notebooks**
- `main.ipynb` — interactive experimentation and plotting
- `tests.ipynb` — correctness tests (qrel content, sentiments alignment, incremental accumulation, fixed_rel_size paths, item uniqueness)

**Folders**
- `data/` — `pool.json` (1848 items, 2738 first names, 1000 surnames)
- `embeddings/` — MD5-keyed `.npy` embedding cache files
- `models/` — downloaded SentenceTransformer model weights

## Coding practices

**IMPORTANT:** Use the correct venv based on `$env:COMPUTERNAME`:
- `EDGAR-PC` (home desktop): `C:\Users\edgar\Projekte\Python\Machine_Learning_venv\Scripts\python.exe`
- `EDGAR_LAPTOP` (work laptop): `C:\Users\EdgarLangwald\OneDrive - neuland AI AG\Coding\.venv\Scripts\python.exe`

Run all scripts and pip installs using the correct python.exe for the current device.
Add to and edit Project structure that's **relevant for coding** as you work on this project, so it stays relevant.

## Other

Documentation links for RWTH Aachen Cluster:
- CLAIX documentation: https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/13ace46cfbb84e92a64c1361e0e4c104/
- RWTH Login command the user will sometimes ask you for: ssh nld68820@login23-1.hpc.itc.rwth-aachen.de
- Copy embeddings to local:

work laptop:
scp -r nld68820@login23-1.hpc.itc.rwth-aachen.de:/home/nld68820/LIMIT/embeddings/ "C:\Users\EdgarLangwald\OneDrive - neuland AI AG\Dokumente\Coding\LIMIT\embeddings"

home pc:
scp -r nld68820@login23-1.hpc.itc.rwth-aachen.de:/home/nld68820/LIMIT/embeddings/ "C:\Users\edgar\Projekte\Python\LIMIT\embeddings"