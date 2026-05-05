"""
Evaluates item retrieval on the disjoint LIMIT-style dataset.

For each m (LOI length), generates n=100 persons each liking m disjoint items,
then tests whether an embedding model can answer "Who likes X?" by cosine similarity.

Model: BAAI/bge-large-en-v1.5 — asymmetric retrieval, query-side instruction prefix.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from create_datasets import build_disjoint_dataset, generate_k_shared_dataset
from name_item_pool import load_pool
from embed import embed, embed_dataset, get_query_prefix, _DEFAULT_CACHE_DIR, _DEFAULT_MODELS_DIR

MODEL_NAME = "BAAI/bge-large-en-v1.5"


def eval_item_retrieval(
    n: int = 100,
    m_max: int | None = None,
    model_name: str = MODEL_NAME,
    query_prefix: str | None = None,
    batch_size: int = 64,
    cache_dir:  str = _DEFAULT_CACHE_DIR,
    models_dir: str = _DEFAULT_MODELS_DIR,
    device:     str | None = None,
) -> dict[int, dict]:
    """
    Evaluate retrieval recall for m = 1..m_max on the disjoint dataset.
    Embeddings for all m values are loaded in a single pass.

    Returns dict mapping m -> {"recall@1", "recall@5", "mrr", "n_queries"}.
    """
    query_prefix = query_prefix or get_query_prefix(model_name)
    if m_max is None:
        m_max = len(load_pool()[0]) // n

    dataset = build_disjoint_dataset(n=n, m_max=m_max)
    doc_embs, qry_embs = embed_dataset(
        dataset, model_name, query_prefix=query_prefix,
        cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size,
    )

    doc_ids = list(dataset["corpus"].keys())
    qry_ids = list(dataset["queries"].keys())
    doc_pos = {d: i for i, d in enumerate(doc_ids)}
    qry_pos = {q: i for i, q in enumerate(qry_ids)}
    qrels   = dataset["qrels"]

    results: dict[int, dict] = {}
    print(f"Model : {model_name}")
    print(f"n     : {n} documents")
    print(f"{'m':>4}  {'recall@1':>9}  {'recall@5':>9}  {'mrr':>7}  queries")
    print("-" * 50)

    for m in range(1, m_max + 1):
        prefix = f"m{m}/"
        m_qids = [q for q in qry_ids if q.startswith(prefix)]
        m_dids = [d for d in doc_ids if d.startswith(prefix)]

        qi = np.array([qry_pos[q] for q in m_qids])
        di = np.array([doc_pos[d] for d in m_dids])
        m_doc_local = {d: j for j, d in enumerate(m_dids)}

        scores = qry_embs[qi] @ doc_embs[di].T
        rel_local  = np.array([m_doc_local[next(iter(qrels[q]))] for q in m_qids])
        rel_scores = scores[np.arange(len(m_qids)), rel_local]
        ranks      = (scores > rel_scores[:, None]).sum(axis=1) + 1

        total = len(m_qids)
        results[m] = {
            "recall@1":  float(np.mean(ranks == 1)),
            "recall@5":  float(np.mean(ranks <= 5)),
            "mrr":       float(np.mean(1.0 / ranks)),
            "n_queries": total,
        }
        r = results[m]
        print(f"{m:>4}  {r['recall@1']:>9.3f}  {r['recall@5']:>9.3f}  {r['mrr']:>7.3f}  {total}")

    return results


def eval_embed_distance(
    n: int = 100,
    m_max: int | None = None,
    k: int = 5,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
    cache_dir:  str = _DEFAULT_CACHE_DIR,
    models_dir: str = _DEFAULT_MODELS_DIR,
    device:     str | None = None,
) -> dict[int, dict]:
    """
    For each LOI length m, embed n documents and measure three geometric properties:

      mean_nn_dist  — average euclidean distance to each doc's nearest neighbour.
                      Shrinks as embeddings collapse toward one another.
      topk_gap      — average gap between the k-th and (k+1)-th nearest-neighbour
                      distances per doc.  A vanishing gap means the k boundary
                      dissolves and retrieval can no longer distinguish rank k from k+1.
      anisotropy    — average pairwise cosine similarity (off-diagonal).
                      Higher values mean embeddings are concentrated in a narrow cone
                      rather than spread across the sphere.
    """
    if m_max is None:
        m_max = len(load_pool()[0]) // n

    dataset = build_disjoint_dataset(n=n, m_max=m_max)
    doc_ids   = list(dataset["corpus"].keys())
    doc_texts = [dataset["corpus"][d] for d in doc_ids]
    doc_pos   = {d: i for i, d in enumerate(doc_ids)}

    embs = embed(doc_texts, model_name, prefix="", cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size)

    results: dict[int, dict] = {}
    print(f"Model : {model_name}")
    print(f"n     : {n} documents  |  k = {k}")
    print(f"{'m':>4}  {'mean_nn_dist':>13}  {'topk_gap':>10}  {'anisotropy':>11}")
    print("-" * 48)

    for m in range(1, m_max + 1):
        prefix  = f"m{m}/"
        m_dids  = [d for d in doc_ids if d.startswith(prefix)]
        di      = np.array([doc_pos[d] for d in m_dids])
        m_embs  = embs[di]  # (n, d), unit vectors
        n_m     = len(m_dids)

        cos_sim = m_embs @ m_embs.T  # (n_m, n_m)
        dists   = np.sqrt(np.clip(2.0 - 2.0 * cos_sim, 0.0, None))

        np.fill_diagonal(dists, np.inf)
        sorted_dists = np.sort(dists, axis=1)

        mean_nn_dist = float(sorted_dists[:, 0].mean())
        topk_gap     = float(np.mean(sorted_dists[:, k] - sorted_dists[:, k - 1]))
        off_diag     = cos_sim[~np.eye(n_m, dtype=bool)]
        anisotropy   = float(off_diag.mean())

        results[m] = {
            "mean_nn_dist": mean_nn_dist,
            "topk_gap": topk_gap,
            "anisotropy": anisotropy,
        }
        r = results[m]
        print(f"{m:>4}  {r['mean_nn_dist']:>13.4f}  {r['topk_gap']:>10.4f}  {r['anisotropy']:>11.4f}")

    return results


def plot_embed_distance(results: dict[int, dict]) -> None:
    ms           = sorted(results)
    mean_nn      = [results[m]["mean_nn_dist"] for m in ms]
    topk_gap     = [results[m]["topk_gap"]     for m in ms]
    anisotropy   = [results[m]["anisotropy"]   for m in ms]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(ms, mean_nn, marker="o", color="steelblue")
    axes[0].set_title("Mean nearest-neighbour distance")
    axes[0].set_xlabel("m (items per person)")
    axes[0].set_ylabel("Euclidean distance")

    axes[1].plot(ms, topk_gap, marker="s", color="darkorange")
    axes[1].set_title("Top-k gap (rank k vs k+1)")
    axes[1].set_xlabel("m (items per person)")
    axes[1].set_ylabel("Distance gap")

    axes[2].plot(ms, anisotropy, marker="^", color="seagreen")
    axes[2].set_title("Anisotropy (avg pairwise cos-sim)")
    axes[2].set_xlabel("m (items per person)")
    axes[2].set_ylabel("Cosine similarity")

    for ax in axes:
        ax.set_xticks(ms)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_results(results: dict[int, dict]) -> None:
    ms      = sorted(results)
    r1      = [results[m]["recall@1"] for m in ms]
    r5      = [results[m]["recall@5"] for m in ms]
    mrr     = [results[m]["mrr"]      for m in ms]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ms, r1,  marker="o", label="Recall@1")
    ax.plot(ms, r5,  marker="s", label="Recall@5")
    ax.plot(ms, mrr, marker="^", label="MRR")
    ax.set_xlabel("m (items per person)")
    ax.set_ylabel("Score")
    ax.set_title("Item retrieval performance vs. LOI length")
    ax.set_xticks(ms)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def eval_retrieval_vs_n(
    m: int = 3,
    k: int = 2,
    n: int | None = None,
    n_values: list[int] | None = None,
    model_name: str = MODEL_NAME,
    query_prefix: str | None = None,
    batch_size: int = 64,
    cache_dir:  str = _DEFAULT_CACHE_DIR,
    models_dir: str = _DEFAULT_MODELS_DIR,
    device:     str | None = None,
) -> dict[int, dict]:
    """
    Evaluate retrieval performance for increasing corpus size, with fixed m and k.

    If n_values is provided, evaluates exactly those n values.
    Otherwise sweeps all even n from n_min to n (inclusive), where n defaults to
    the maximum the item pool allows: pool_size * k // m floored to even.

    Metrics per query:
      recall@1  — fraction of relevant docs at rank 1
      recall@k  — fraction of relevant docs in top-k
      mrr       — 1 / rank of the highest-ranked relevant doc
    """
    query_prefix = query_prefix or get_query_prefix(model_name)

    pool_size = len(load_pool()[0])

    if n_values is not None:
        sweep = n_values
    else:
        if n is None:
            n = (pool_size * k // m) // 2 * 2
        n_min = (m // 2 + 1) * 2
        sweep = range(n_min, n + 1, 2)

    results: dict[int, dict] = {}
    print(f"Model : {model_name}")
    print(f"m={m}, k={k}")
    print(f"{'n':>6}  {'recall@2':>9}  {'recall@5':>9}  {'recall@10':>10}  queries")
    print("-" * 52)

    for n in sweep:
        dataset = generate_k_shared_dataset(n=n, m=m, k=k)
        corpus    = dataset["corpus"]
        queries   = dataset["queries"]
        qrels     = dataset["qrels"]

        doc_ids   = list(corpus.keys())
        doc_texts = [corpus[d] for d in doc_ids]
        doc_idx   = {d: i for i, d in enumerate(doc_ids)}

        query_ids   = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        doc_embs = embed(doc_texts,   model_name, prefix="",           cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size)
        qry_embs = embed(query_texts, model_name, prefix=query_prefix, cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size)

        scores = qry_embs @ doc_embs.T  # (n_queries, n_docs)

        r2_hits, r5_hits, r10_hits = [], [], []
        for qi, qid in enumerate(query_ids):
            rel_idxs = [doc_idx[name] for name in qrels[qid]]
            rel_scores = scores[qi, rel_idxs]
            ranks = np.array([(scores[qi] > s).sum() + 1 for s in rel_scores])
            n_rel = len(rel_idxs)
            r2_hits.append((ranks <= 2).sum() / n_rel)
            r5_hits.append((ranks <= 5).sum() / n_rel)
            r10_hits.append((ranks <= 10).sum() / n_rel)

        total = len(query_ids)
        r2  = float(np.mean(r2_hits))
        r5  = float(np.mean(r5_hits))
        r10 = float(np.mean(r10_hits))
        results[n] = {"recall@2": r2, "recall@5": r5, "recall@10": r10, "n_queries": total}
        print(f"{n:>6}  {r2:>9.3f}  {r5:>9.3f}  {r10:>10.3f}  {total}")

    return results


def plot_retrieval_vs_n(results: dict[int, dict], m: int, k: int) -> None:
    ns  = sorted(results)
    r2  = [results[n]["recall@2"]  for n in ns]
    r5  = [results[n]["recall@5"]  for n in ns]
    r10 = [results[n]["recall@10"] for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, r2,  marker="o", label="Recall@2  (both in top-2)")
    ax.plot(ns, r5,  marker="s", label="Recall@5  (both in top-5)")
    ax.plot(ns, r10, marker="^", label="Recall@10 (both in top-10)")
    ax.set_xlabel("n (corpus size)")
    ax.set_ylabel("Score")
    ax.set_title(f"Retrieval vs corpus size  (m={m}, k={k})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n = 50
    pool_size = len(load_pool()[0])
    print(f"Item pool size: {pool_size}, max m for n={n}: {pool_size // n}")

    dist_results = eval_embed_distance(n=n)
    plot_embed_distance(dist_results)

    retrieval_results = eval_item_retrieval(n=n)
    plot_results(retrieval_results)
