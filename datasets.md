# Dataset Designs

## Shared Building Blocks

**Pool** (`data/pool.json`): 1848 items (things people can "like"), 2738 first names, 1000 surnames.

**Document format** (`_fmt_likes`): `"Alice Smith likes apples, cats and winter."` — a person-profile sentence.

**Query format**: `"Who likes X?"` or `"Who likes X and Y?"` — attribute lookup questions.

**The retrieval task**: embed all documents and all queries, then rank documents by cosine similarity to each query. Correct retrieval means the right person lands at rank 1 (or in top-k).

---

## 1. `increase_m(n, m_max)`

**What is tested:** Does recall degrade as a person's profile grows longer? Each person likes exactly `m` items, and `m` sweeps from 1 to `m_max`. The core LOI-length stress test.

**Constraint:** Items are disjoint — no two people share an item at any given `m`. Each query therefore has exactly **one** correct answer.

**Algorithm:**

```
for m in 1..m_max:
    sample n*m unique items from pool (no repeats)
    split into n chunks of size m → each chunk belongs to one person
    documents: "Name likes item_1, ..., item_m."
    queries:   for every (person, item) pair → "Who likes item?" → {person: 1}
    namespace: prefix all IDs with "m{m}/" so all slices live in one dict
```

All `m` slices use the **same** `n` names (same people, different item assignments at each LOI length). The merged dict has `n * m_max` documents and `n * (1 + 2 + ... + m_max)` queries total.

**Used by:** `eval_item_retrieval` (recall vs. m) and `eval_embed_distance` (geometry vs. m).



---

## 2. `generate_k_shared_dataset(n, m, k)`

**What is tested:** Multi-relevant-doc retrieval. When an item is owned by exactly `k` people, can the model retrieve all `k` correct documents? Also used to study how recall degrades as corpus size `n` grows.

**k = 1 — disjoint:** Identical logic to one slice of `build_disjoint_dataset`. `n*m` items divided equally, each query has 1 relevant doc.

**k = 2 — shared via a regular graph:**

Each item is co-owned by exactly 2 people. The sharing structure is determined by a random **m-regular graph** on `n` nodes (every node = person, every edge = shared item). Because each node has degree `m`, every person ends up liking exactly `m` items.

```
G = random m-regular graph on n nodes
sample n*m/2 items (one per edge)
for each (item, edge (u, v)):
    person_items[u].append(item)
    person_items[v].append(item)
documents: "Name likes item_1, ..., item_m."
queries:   "Who likes item?" → {names[u]: 1, names[v]: 1}
```

The item count is `n*m/2` because each item covers 2 people. Each query has exactly **2** relevant documents.

**Used by:** `eval_retrieval_vs_n` — sweeps corpus size `n` with fixed `m` and `k` to show how recall scales with the number of distractors.

---

## 3. `generate_steiner_dataset(n)`

**What is tested:** Pair-conjunction queries — `"Who likes X and Y?"` — where exactly one document contains both items. Tests whether models can handle conjunctive attribute lookups that require joint representation.

**Key structure — Steiner Triple System STS(n):** A collection of 3-element subsets ("triples") of `{0, ..., n−1}` such that **every pair of elements appears in exactly one triple**. This exists iff `n ≡ 1 or 3 (mod 6)`. With `n` items there are `n(n−1)/6` triples.

Default `n = 1849` → 569,492 documents.

**Why STS?** The STS property guarantees that for any two items `(X, Y)` there is a unique document containing both. So every pair query has exactly **one** correct answer — no ambiguity, clean single-answer retrieval.

**Algorithm:**

```
triples = STS_construction(n)        # Bose (n≡3 mod 6) or Skolem (n≡1 mod 6)
items = random_sample(pool, n)       # assign one pool item to each index 0..n-1
names = generate_names(len(triples))

for each triple (a, b, c):
    doc:     "Name likes item_a, item_b and item_c."
    queries: "Who likes item_a and item_b?" → {doc: 1}
             "Who likes item_a and item_c?" → {doc: 1}
             "Who likes item_b and item_c?" → {doc: 1}
```

Each document produces 3 queries (one per pair), so there are `n(n−1)/2` queries total.

**Bose vs. Skolem construction** (both are classical combinatorial designs):
- `n ≡ 3 (mod 6)` — Bose: sets `t = (n−3)/6`, works over `Z_{2t+1} × Z_3`, generates triples via a cyclic difference construction.
- `n ≡ 1 (mod 6)` — Skolem: sets `t = (n−1)/6`, similar modular arithmetic extended with one "infinity" point `n−1`.

Both constructions produce exactly `n(n−1)/6` triples covering every pair exactly once, verified by deduplication with `frozenset`.
