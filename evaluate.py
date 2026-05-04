"""
Evaluates item retrieval on the disjoint LIMIT-style dataset.

For each m (LOI length), generates n=100 persons each liking m disjoint items,
then tests whether an embedding model can answer "Who likes X?" by cosine similarity.

Model: BAAI/bge-large-en-v1.5 — asymmetric retrieval, query-side instruction prefix.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from custom_dataset_creation import generate_disjoint_dataset, generate_names, generate_k_shared_dataset, _load_notebook_globals
from embed import embed, _DEFAULT_CACHE_DIR, _DEFAULT_MODELS_DIR

MODEL_NAME = "BAAI/bge-large-en-v1.5"
# BGE asymmetric retrieval: queries get this prefix, documents get none
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def eval_item_retrieval(
    n: int = 100,
    m: int | None = None,
    model_name: str = MODEL_NAME,
    query_prefix: str = QUERY_PREFIX,
    batch_size: int = 64,
    single_query: bool = False,
    cache_dir:  str = _DEFAULT_CACHE_DIR,
    models_dir: str = _DEFAULT_MODELS_DIR,
    device:     str | None = None,
) -> dict[int, dict]:
    """
    Evaluate retrieval performance for increasing list-of-items (LOI) length.

    Args:
        n:          Number of persons (documents) per experiment.
        m_values:   LOI lengths to test. Defaults to 1..18.
        model_name: HuggingFace SentenceTransformer model.
        batch_size: Encoding batch size.

    Returns:
        dict mapping m -> {"recall@1", "recall@5", "mrr", "n_queries"}
    """
    if m is None:
        pool_size = len(_load_notebook_globals()[0])
        m = pool_size // n

    names = generate_names(n)
    results: dict[int, dict] = {}

    print(f"Model : {model_name}")
    print(f"n     : {n} documents")
    print(f"{'m':>4}  {'recall@1':>9}  {'recall@5':>9}  {'mrr':>7}  queries")
    print("-" * 50)

    for m in range(1, m + 1):
        dataset = generate_disjoint_dataset(n=n, m=m, names=names, single_query=single_query)
        corpus  = dataset["corpus"]   # {name: "Name likes ..."}
        queries = dataset["queries"]  # {qid: "Who likes X?"}
        qrels   = dataset["qrels"]    # {qid: {name: 1}}

        doc_ids   = list(corpus.keys())
        doc_texts = [corpus[d] for d in doc_ids]
        doc_idx   = {d: i for i, d in enumerate(doc_ids)}

        query_ids   = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]

        doc_embs = embed(doc_texts,   model_name, prefix="",           cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size)
        qry_embs = embed(query_texts, model_name, prefix=query_prefix, cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size)

        # (n_queries × n_docs) cosine similarity matrix
        scores = qry_embs @ doc_embs.T

        rel_indices = np.array(
            [doc_idx[next(iter(qrels[qid]))] for qid in query_ids]
        )
        # rank of the relevant doc = number of docs that scored strictly higher + 1
        rel_scores = scores[np.arange(len(query_ids)), rel_indices]
        ranks = (scores > rel_scores[:, None]).sum(axis=1) + 1  # (n_queries,)

        total = len(query_ids)
        r1      = float(np.mean(ranks == 1))
        r5      = float(np.mean(ranks <= 5))
        mrr_sum = float(np.sum(1.0 / ranks))
        results[m] = {
            "recall@1": r1,
            "recall@5": r5,
            "mrr":      mrr_sum / total,
            "n_queries": total,
        }
        r = results[m]
        print(f"{m:>4}  {r['recall@1']:>9.3f}  {r['recall@5']:>9.3f}  {r['mrr']:>7.3f}  {total}")

    return results


def eval_embed_distance(
    n: int = 100,
    m: int | None = None,
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
    if m is None:
        pool_size = len(_load_notebook_globals()[0])
        m = pool_size // n

    names = generate_names(n)
    results: dict[int, dict] = {}

    print(f"Model : {model_name}")
    print(f"n     : {n} documents  |  k = {k}")
    print(f"{'m':>4}  {'mean_nn_dist':>13}  {'topk_gap':>10}  {'anisotropy':>11}")
    print("-" * 48)

    for m in range(1, m + 1):
        dataset = generate_disjoint_dataset(n=n, m=m, names=names)
        doc_texts = list(dataset["corpus"].values())

        embs = embed(doc_texts, model_name, prefix="", cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size)
        # (n, d), unit vectors

        # Cosine similarity via dot product (embeddings are already normalised)
        cos_sim = embs @ embs.T  # (n, n)

        # Euclidean distance on the unit sphere: sqrt(2 - 2·cos)
        dists = np.sqrt(np.clip(2.0 - 2.0 * cos_sim, 0.0, None))

        # Exclude self-distance from neighbour statistics
        np.fill_diagonal(dists, np.inf)
        sorted_dists = np.sort(dists, axis=1)  # ascending; col 0 = nearest neighbour

        mean_nn_dist = float(sorted_dists[:, 0].mean())
        # gap between rank-k and rank-(k+1) neighbour (0-indexed: cols k-1 and k)
        topk_gap = float(np.mean(sorted_dists[:, k] - sorted_dists[:, k - 1]))

        # Anisotropy: average off-diagonal cosine similarity
        off_diag = cos_sim[~np.eye(n, dtype=bool)]
        anisotropy = float(off_diag.mean())

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
    query_prefix: str = QUERY_PREFIX,
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
    pool_size = len(_load_notebook_globals()[0])

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
    pool_size = len(_load_notebook_globals()[0])
    print(f"Item pool size: {pool_size}, max m for n={n}: {pool_size // n}")

    dist_results = eval_embed_distance(n=n)
    plot_embed_distance(dist_results)

    retrieval_results = eval_item_retrieval(n=n)
    plot_results(retrieval_results)
