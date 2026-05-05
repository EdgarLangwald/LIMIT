## Overview

The goal of this project is to understand the LIMIT paper in action. Currently I want to find out whether an embedding model can even embed a list of attributes meaningfully, i.e. if models fail due to geometric constraints or due to bad experiment design.

## Current state of project

Hypothesis: Such long lists of items (loi) are unfeasable to embed for any model.
    Question: Do document embeddings ie loi cluster more then text chunks from realistic pdf's?
    Question: Is information about every item preserved in lois embedding? I.e. can every item be retrieved?
Result: Clustering increases and recall steadily drops with increasing loi length (Recall@1 from 100% to 20% at n=100 people and m=1, ..., 18 items). Models can't embed a loi in a way that preserves each items meaning reliably. But they are still able to with m = 3.



## Other

Path to LIMIT paper pdf "C:\OneDrive\Documents\Papers\LIMIT DeepMind.pdf"
URL to LIMIT repo: https://github.com/google-deepmind/limit
This Claude Code session is being run from windows powershell
**IMPORTANT** Run all scripts from C:\OneDrive\Documents\Coding\.venv this venv. Also pip install into there
Documentation links for RWTH Aachen Cluster:
- Submit Jobs: https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/13ace46cfbb84e92a64c1361e0e4c104/
- Job Parameters: https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/3d20a87835db4569ad9094d91874e2b4/
- Job Management: https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/85b21b312bfb48b290043083d2a34b8f/

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

## Paper description for quick context

**"On the Theoretical Limitations of Embedding-Based Retrieval"** (ICLR 2026)
Weller et al., Google DeepMind & Johns Hopkins

**Core claim:** Single-vector embedding models have a hard geometric ceiling. For a given embedding dimension *d*, there exist top-k subsets of documents that *no* query vector can retrieve — not because of bad training, but because the geometry doesn't allow it. The number of representable top-k combinations is bounded by *d*.

**Theoretical basis:** Connects results from linear algebra / high-dimensional geometry (Papadimitriou & Sipser) to dense retrieval. Lower-bounds the embedding dimension needed to represent a given set of query-document relevance combinations.

**Empirical validation:** They bypass training entirely by *directly optimizing free parameter embeddings* on the test set.

**The LIMIT dataset:** A deliberately simple synthetic-to-natural dataset. Documents are short person profiles ("Jon likes Apples and Rabbits."), queries are attribute lookups ("Who likes Apples?").