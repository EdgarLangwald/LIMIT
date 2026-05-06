import random
import networkx as nx

from name_item_pool import load_pool


def generate_names(n: int, seed: int = 42) -> list[str]:
    _, first_names, surnames = load_pool()
    rng = random.Random(seed)
    raw = [
        f"{rng.choice(first_names).capitalize()} {rng.choice(surnames).capitalize()}"
        for _ in range(n * 3)
    ]
    names = list(dict.fromkeys(raw))[:n]
    assert len(names) == n, "Not enough unique names; increase oversampling factor"
    return names


def _fmt_likes(name: str, items: list[str]) -> str:
    if len(items) == 1:
        return f"{name} likes {items[0]}."
    return f"{name} likes {', '.join(items[:-1])} and {items[-1]}."


def build_disjoint_dataset(n: int, m_max: int | None = None, seed: int = 42) -> tuple[list[dict], list[dict], list[int]]:
    """
    For each m in 1..m_max, generate a disjoint dataset where n persons each like m unique items.
    Returns (datasets, qrels_list, m_values).
    """
    items_pool, _, _ = load_pool()
    if m_max is None:
        m_max = len(items_pool) // n
    assert n * m_max <= len(items_pool), f"n*m_max={n*m_max} exceeds item pool size ({len(items_pool)})"

    names = generate_names(n, seed)
    datasets, qrels_list, meta = [], [], []

    for m in range(1, m_max + 1):
        rng = random.Random(seed)
        pool = rng.sample(items_pool, n * m)
        person_items = [pool[i * m:(i + 1) * m] for i in range(n)]
        corpus  = {name: _fmt_likes(name, items) for name, items in zip(names, person_items)}
        queries, qrels = {}, {}
        for pidx, (name, items) in enumerate(zip(names, person_items)):
            for iidx, item in enumerate(items):
                qid = f"query_{pidx * m + iidx}"
                queries[qid] = f"Who likes {item}?"
                qrels[qid]   = {name: 1}
        datasets.append({"corpus": corpus, "queries": queries})
        qrels_list.append(qrels)
        meta.append(m)

    return datasets, qrels_list, meta


def generate_k_shared_dataset(
    n: int,
    m: int,
    k: int,
    seed: int = 42,
    names: list[str] | None = None,
) -> tuple[dict, dict]:
    """
    Generate a dataset where each item is shared by exactly k persons.
    Returns (dataset, qrels).

    k=1: disjoint (each query has 1 relevant doc).
    k=2: uses a random m-regular graph (each query has 2 relevant docs).
    """
    assert k in (1, 2), f"k={k} not supported; only k=1 and k=2"
    assert (n * m) % k == 0, "n*m must be divisible by k"

    items_pool, _, _ = load_pool()
    n_items = n * m // k
    assert n_items <= len(items_pool), f"need {n_items} items, pool only has {len(items_pool)}"

    if names is None:
        names = generate_names(n, seed)

    rng = random.Random(seed)
    sampled_items = rng.sample(items_pool, n_items)

    if k == 1:
        person_items = [sampled_items[i * m:(i + 1) * m] for i in range(n)]
        edges = []
    else:
        G = nx.random_regular_graph(m, n, seed=seed)
        edges = list(G.edges())
        person_items = [[] for _ in range(n)]
        for item, (u, v) in zip(sampled_items, edges):
            person_items[u].append(item)
            person_items[v].append(item)

    corpus  = {name: _fmt_likes(name, items) for name, items in zip(names, person_items)}
    queries, qrels = {}, {}

    if k == 1:
        for pidx, (name, items) in enumerate(zip(names, person_items)):
            for iidx, item in enumerate(items):
                qid = f"query_{pidx * m + iidx}"
                queries[qid] = f"Who likes {item}?"
                qrels[qid]   = {name: 1}
    else:
        for qidx, (item, (u, v)) in enumerate(zip(sampled_items, edges)):
            qid = f"query_{qidx}"
            queries[qid] = f"Who likes {item}?"
            qrels[qid]   = {names[u]: 1, names[v]: 1}

    return {"corpus": corpus, "queries": queries}, qrels


def generate_steiner_dataset(n: int = 1849, seed: int = 42) -> tuple[dict, dict]:
    """
    Generate a Steiner Triple System dataset with AND pair queries.
    Returns (dataset, qrels).

    Constructs STS(n) so every pair of items appears in exactly one document.
    Each document's 3 pairs become queries "Who likes X and Y?" with exactly one
    relevant document each. n must be ≡ 1 or 3 (mod 6).
    """
    assert n % 6 in (1, 3), f"STS({n}) does not exist: n must be ≡ 1 or 3 (mod 6)"
    items_pool, _, _ = load_pool()
    assert n <= len(items_pool), f"n={n} exceeds item pool size ({len(items_pool)})"

    def _sts_indices(n: int) -> list[tuple[int, int, int]]:
        """Bose (n≡3 mod 6) / Skolem (n≡1 mod 6) construction. Adapted from SageMath."""
        if n % 6 == 3:
            t = (n - 3) // 6
            sz = 2 * t + 1
            T = lambda x, y: x + sz * y
            raw = (
                [[T(i,0), T(i,1), T(i,2)] for i in range(sz)] +
                [[T(i,k), T(j,k), T(((t+1)*(i+j)) % sz, (k+1)%3)]
                 for k in range(3) for i in range(sz) for j in range(sz) if i != j]
            )
        else:
            t = (n - 1) // 6
            two_t = 2 * t

            def T(x, y):
                return n - 1 if (x, y) == (-1, -1) else x + y * two_t

            def L(i, j):
                l1 = (i + j) % (two_t)
                return l1 // 2 if l1 % 2 == 0 else t + (l1 - 1) // 2

            raw = (
                [[T(i,0), T(i,1), T(i,2)] for i in range(t)] +
                [[T(-1,-1), T(i,k), T(i - t, (k+1)%3)] for i in range(t, two_t) for k in range(3)] +
                [[T(i,k), T(j,k), T(L(i,j), (k+1)%3)] for k in range(3) for i in range(two_t) for j in range(i+1, two_t)]
            )
        return [tuple(sorted(fs)) for fs in {frozenset(triple) for triple in raw}]

    rng    = random.Random(seed)
    items  = rng.sample(items_pool, n)
    triples = _sts_indices(n)
    names  = generate_names(len(triples), seed)

    corpus, queries, qrels = {}, {}, {}
    for doc_idx, (a, b, c) in enumerate(triples):
        ia, ib, ic = items[a], items[b], items[c]
        doc_id = f"person_{doc_idx}"
        corpus[doc_id] = _fmt_likes(names[doc_idx], [ia, ib, ic])
        for q_offset, (x, y) in enumerate([(ia, ib), (ia, ic), (ib, ic)]):
            qid = f"query_{doc_idx * 3 + q_offset}"
            queries[qid] = f"Who likes {x} and {y}?"
            qrels[qid]   = {doc_id: 1}

    return {"corpus": corpus, "queries": queries}, qrels
