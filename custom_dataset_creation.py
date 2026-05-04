"""
Generates a fully disjoint LIMIT-style IR dataset where every item belongs to
exactly one person. Each query "Who likes X?" has exactly one relevant document.

The qrels matrix (n*m x n) has one 1 per row and m ones per column — person k
owns exactly m queries, one per item they like.
"""

import functools
import nbformat
import random
import os
import networkx as nx

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "generate_limit_dataset.ipynb")


@functools.lru_cache(maxsize=None)
def _load_notebook_globals():
    """Execute notebook setup cells until items_to_like and names are available."""
    with open(NOTEBOOK_PATH) as f:
        nb = nbformat.read(f, as_version=4)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        exec(cell.source, ns)
        if all(k in ns for k in ("items_to_like", "unique_names", "unique_surnames")) and isinstance(ns["items_to_like"], list):
            break
    return ns["items_to_like"], ns["unique_names"], ns["unique_surnames"]


def generate_names(n: int, seed: int = 42) -> list[str]:
    """Generate n unique person names from the notebook name pools."""
    _, unique_names, unique_surnames = _load_notebook_globals()
    rng = random.Random(seed)
    raw = [
        f"{rng.choice(unique_names).capitalize()} {rng.choice(unique_surnames).capitalize()}"
        for _ in range(n * 3)
    ]
    names = list(dict.fromkeys(raw))[:n]
    assert len(names) == n, "Could not generate enough unique names; increase n*3 oversampling"
    return names


def generate_disjoint_dataset(
    n: int,
    m: int,
    seed: int = 42,
    single_query: bool = True,
    names: list[str] | None = None,
) -> dict:
    """
    Generate a fully disjoint LIMIT-style IR dataset.

    Args:
        n: number of people (documents)
        m: number of items each person likes
        seed: random seed (used for item assignment; name generation uses its own RNG)
        single_query: if True, one random query per person; otherwise one per item
        names: pre-generated person names; if None, generated via generate_names(n, seed)

    Returns:
        dict with keys "corpus", "queries", "qrels"
    """
    items_to_like, _, _ = _load_notebook_globals()
    pool_size = len(items_to_like)
    assert n * m <= pool_size, f"n*m={n*m} exceeds item pool size ({pool_size})"

    if names is None:
        names = generate_names(n, seed)

    # Item assignment uses its own RNG so it doesn't couple with name generation
    rng = random.Random(seed)
    pool = rng.sample(items_to_like, n * m)
    person_items = [pool[i * m:(i + 1) * m] for i in range(n)]

    corpus = {
        name: f"{name} likes {', '.join(items[:-1])} and {items[-1]}."
        for name, items in zip(names, person_items)
    }

    queries, qrels = {}, {}
    for qidx, (name, items) in enumerate(zip(names, person_items)):
        if single_query:
            item = rng.choice(items)
            qid = f"query_{qidx * m + items.index(item)}"
            queries[qid] = f"Who likes {item}?"
            qrels[qid] = {name: 1}
        else:
            for item in items:
                qid = f"query_{qidx * m + items.index(item)}"
                queries[qid] = f"Who likes {item}?"
                qrels[qid] = {name: 1}

    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def _fmt_likes(name: str, items: list[str]) -> str:
    if len(items) == 1:
        return f"{name} likes {items[0]}."
    return f"{name} likes {', '.join(items[:-1])} and {items[-1]}."


def generate_k_shared_dataset(
    n: int,
    m: int,
    k: int,
    seed: int = 42,
    names: list[str] | None = None,
) -> dict:
    """
    Generate a dataset where each item is shared by exactly k persons.

    For k=1: items are disjoint, each query has 1 relevant doc.
    For k=2: uses a random m-regular graph; each query has 2 relevant docs.

    Args:
        n: number of persons (documents)
        m: items per person (LOI length); graph degree for k=2
        k: number of persons sharing each item (= relevant docs per query)
        seed: random seed
        names: pre-generated names; if None, generated via generate_names(n, seed)

    Returns:
        dict with "corpus", "queries", "qrels"
    """
    assert k in (1, 2), f"k={k} not supported; only k=1 and k=2"
    assert (n * m) % k == 0, f"n*m must be divisible by k"

    items_pool, _, _ = _load_notebook_globals()
    n_items = n * m // k
    assert n_items <= len(items_pool), f"need {n_items} items, pool only has {len(items_pool)}"

    if names is None:
        names = generate_names(n, seed)

    rng = random.Random(seed)
    sampled_items = rng.sample(items_pool, n_items)

    if k == 1:
        person_items = [sampled_items[i * m:(i + 1) * m] for i in range(n)]
        edges = [(i, i) for i in range(n * m)]  # placeholder, not used below
    else:
        G = nx.random_regular_graph(m, n, seed=seed)
        edges = list(G.edges())
        person_items = [[] for _ in range(n)]
        for item, (u, v) in zip(sampled_items, edges):
            person_items[u].append(item)
            person_items[v].append(item)

    corpus = {name: _fmt_likes(name, items) for name, items in zip(names, person_items)}

    queries, qrels = {}, {}
    if k == 1:
        for pidx, (name, items) in enumerate(zip(names, person_items)):
            for iidx, item in enumerate(items):
                qid = f"query_{pidx * m + iidx}"
                queries[qid] = f"Who likes {item}?"
                qrels[qid] = {name: 1}
    else:
        for qidx, (item, (u, v)) in enumerate(zip(sampled_items, edges)):
            qid = f"query_{qidx}"
            queries[qid] = f"Who likes {item}?"
            qrels[qid] = {names[u]: 1, names[v]: 1}

    return {"corpus": corpus, "queries": queries, "qrels": qrels}


if __name__ == "__main__":
    dataset = generate_disjoint_dataset(n=10, m=10)
    print(f"Corpus : {len(dataset['corpus'])} docs")
    print(f"Queries: {len(dataset['queries'])}")
    for qid, q in list(dataset["queries"].items())[:3]:
        doc = list(dataset["qrels"][qid].keys())[0]
        print(f"  {q!r:40s} -> {doc}")
