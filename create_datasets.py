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


def _fmt_dislikes(name: str, items: list[str]) -> str:
    if len(items) == 1:
        return f"{name} dislikes {items[0]}."
    return f"{name} dislikes {', '.join(items[:-1])} and {items[-1]}."


def build_disjoint_dataset(n: int, m: int, seed: int = 42) -> tuple[dict, list[list[int]]]:
    """Generate a dataset where n persons each like m unique items (disjoint across people)."""
    items_pool, _, _ = load_pool()
    assert n * m <= len(items_pool), f"n*m={n*m} exceeds item pool size ({len(items_pool)})"

    names = generate_names(n, seed)
    rng = random.Random(seed)
    pool = rng.sample(items_pool, n * m)
    person_items = [pool[i * m:(i + 1) * m] for i in range(n)]
    corpus = {name: _fmt_likes(name, items) for name, items in zip(names, person_items)}
    queries, qrels = {}, []
    for pidx, (name, items) in enumerate(zip(names, person_items)):
        for iidx, item in enumerate(items):
            qid = f"query_{pidx * m + iidx}"
            queries[qid] = f"Who likes {item}?"
            qrels.append([pidx])
    return {"corpus": corpus, "queries": queries}, qrels


def _try_greedy_preference(
    n: int, m: int, items_pool: list, rng: random.Random
) -> tuple[list[list], list[list], bool]:
    """
    Attempt to assign m positive and m negative items to each of n people such that
    no person's positive and negative lists overlap, and every item from the sampled
    pool appears exactly once as a positive and once as a negative.

    Returns (pos_lists, neg_lists, success). On failure (last person can't be placed),
    success=False and the lists contain n-1 entries.
    """
    items = rng.sample(items_pool, n * m)
    pos_pool = list(items)
    neg_pool = list(items)
    pos_lists: list[list] = []
    neg_lists: list[list] = []

    for _ in range(n - 1):
        pos_sample = rng.sample(pos_pool, m)
        pos_set = set(pos_sample)
        for x in pos_sample:
            pos_pool.remove(x)
        pos_lists.append(pos_sample)

        available = [x for x in neg_pool if x not in pos_set]
        neg_sample = rng.sample(available, m)
        for x in neg_sample:
            neg_pool.remove(x)
        neg_lists.append(neg_sample)

    last_pos = list(pos_pool)
    last_pos_set = set(last_pos)
    free = [x for x in neg_pool if x not in last_pos_set]

    if len(free) >= m:
        pos_lists.append(last_pos)
        neg_lists.append(rng.sample(free, m))
        return pos_lists, neg_lists, True

    # Atomic swap: find person X whose neg list doesn't conflict with last_pos,
    # and whose pos list doesn't conflict with neg_pool (so X can take neg_pool).
    neg_pool_set = set(neg_pool)
    for i in range(n - 1):
        if set(neg_lists[i]).isdisjoint(last_pos_set) and neg_pool_set.isdisjoint(set(pos_lists[i])):
            pos_lists.append(last_pos)
            neg_lists.append(neg_lists[i])
            neg_lists[i] = list(neg_pool)
            return pos_lists, neg_lists, True

    return pos_lists, neg_lists, False


def build_preference_dataset(
    n: int,
    m: int,
    seed: int = 42,
) -> tuple[dict, list[list[int]], list[dict]]:
    """
    Each person likes m items and dislikes m other items (no overlap).
    Every item appears exactly once as liked and once as disliked across all people.

    Generates 3 queries per item:
      - "Who likes X?"                  → qrels: [liker_idx]
      - "Who dislikes X?"               → qrels: [disliker_idx]
      - "Who has a preference about X?" → qrels: [liker_idx, disliker_idx]

    Returns (dataset, qrels, sentiments) where sentiments is a list aligned with
    queries, each entry: {type, liker_idx, disliker_idx}.
    """
    items_pool, _, _ = load_pool()
    assert n * m <= len(items_pool), f"n*m={n*m} exceeds pool size ({len(items_pool)})"
    assert n >= 2

    names = generate_names(n, seed)
    pos_lists = neg_lists = None
    success = False

    for attempt in range(5):
        rng = random.Random(seed + attempt)
        pos_lists, neg_lists, success = _try_greedy_preference(n, m, items_pool, rng)
        if success:
            break

    n_actual = len(pos_lists)
    if not success:
        print(f"Warning: m={m}: could not place last person after 5 attempts, using {n_actual}/{n} people")

    actual_names = names[:n_actual]
    corpus = {
        name: _fmt_likes(name, pos) + " " + _fmt_dislikes(name, neg)
        for name, pos, neg in zip(actual_names, pos_lists, neg_lists)
    }

    item_to_liker    = {item: name for name, pos in zip(actual_names, pos_lists) for item in pos}
    item_to_disliker = {item: name for name, neg in zip(actual_names, neg_lists) for item in neg}
    all_items = [item for pos in pos_lists for item in pos]

    name_to_idx = {name: i for i, name in enumerate(actual_names)}
    queries: dict[str, str] = {}
    qrels: list[list[int]] = []
    sentiments: list[dict] = []

    for qid_idx, item in enumerate(all_items):
        liker, disliker = item_to_liker[item], item_to_disliker[item]
        liker_idx, disliker_idx = name_to_idx[liker], name_to_idx[disliker]
        base = qid_idx * 3

        queries[f"query_{base}"]     = f"Who likes {item}?"
        qrels.append([liker_idx])
        sentiments.append({"type": "like", "liker_idx": liker_idx, "disliker_idx": disliker_idx})

        queries[f"query_{base + 1}"] = f"Who dislikes {item}?"
        qrels.append([disliker_idx])
        sentiments.append({"type": "dislike", "liker_idx": liker_idx, "disliker_idx": disliker_idx})

        queries[f"query_{base + 2}"] = f"Who has a preference about {item}?"
        qrels.append([liker_idx, disliker_idx])
        sentiments.append({"type": "neutral", "liker_idx": liker_idx, "disliker_idx": disliker_idx})

    return {"corpus": corpus, "queries": queries}, qrels, sentiments


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
    queries, qrels = {}, []

    if k == 1:
        for pidx, (name, items) in enumerate(zip(names, person_items)):
            for iidx, item in enumerate(items):
                qid = f"query_{pidx * m + iidx}"
                queries[qid] = f"Who likes {item}?"
                qrels.append([pidx])
    else:
        for qidx, (item, (u, v)) in enumerate(zip(sampled_items, edges)):
            qid = f"query_{qidx}"
            queries[qid] = f"Who likes {item}?"
            qrels.append([u, v])

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

    corpus, queries, qrels = {}, {}, []
    for doc_idx, (a, b, c) in enumerate(triples):
        ia, ib, ic = items[a], items[b], items[c]
        doc_id = f"person_{doc_idx}"
        corpus[doc_id] = _fmt_likes(names[doc_idx], [ia, ib, ic])
        for q_offset, (x, y) in enumerate([(ia, ib), (ia, ic), (ib, ic)]):
            qid = f"query_{doc_idx * 3 + q_offset}"
            queries[qid] = f"Who likes {x} and {y}?"
            qrels.append([doc_idx])

    return {"corpus": corpus, "queries": queries}, qrels 

def increase_param(fn_name: str, param: str, values, **kwargs) -> list:
    """
    Call fn_name repeatedly with `param` set to each value in `values`,
    passing any additional `kwargs` through unchanged each call.
    Returns a list of the function's return values.
    """
    _registry = {
        "build_disjoint_dataset": build_disjoint_dataset,
        "build_preference_dataset": build_preference_dataset,
        "generate_k_shared_dataset": generate_k_shared_dataset,
        "generate_steiner_dataset": generate_steiner_dataset,
    }
    fn = _registry[fn_name]
    return [fn(**{param: v}, **kwargs) for v in values]