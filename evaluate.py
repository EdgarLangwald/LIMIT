import numpy as np
import matplotlib.pyplot as plt
from create_datasets import build_disjoint_dataset
from name_item_pool import load_pool
from embed import embed, embed_dataset, get_query_prefix, _DEFAULT_CACHE_DIR, _DEFAULT_MODELS_DIR

MODEL_NAME = "BAAI/bge-large-en-v1.5"


def retrieval_metrics(
    scores: np.ndarray,
    relevant_indices: list[list[int]],
    ks: list[int] = [1, 5],
) -> dict[str, float]:
    """
    scores:           (n_queries, n_docs) cosine similarity matrix
    relevant_indices: per-query list of relevant doc column indices
    Returns recall@k for each k in ks, plus mrr.
    """
    recall_hits = {k: [] for k in ks}
    mrr_vals = []
    for qi, rel in enumerate(relevant_indices):
        row = scores[qi]
        rel_scores = row[np.array(rel)]
        ranks = (row[None, :] > rel_scores[:, None]).sum(axis=1) + 1
        mrr_vals.append(float(1.0 / ranks.min()))
        for k in ks:
            recall_hits[k].append(float((ranks <= k).mean()))
    out = {f"recall@{k}": float(np.mean(recall_hits[k])) for k in ks}
    out["mrr"] = float(np.mean(mrr_vals))
    return out


def evaluate(
    mapping_or_list,
    qrels_or_list,
    recall_at: list[int] = [1, 5],
    n_values: list[int] | None = None,
) -> list[dict]:
    """
    Compute retrieval metrics from structured mappings returned by embed_dataset.

    mapping_or_list: {"docs": {id: emb}, "queries": {id: emb}} or a list of these
    qrels_or_list:   matching {query_id: {doc_id: 1}} or list of these
    recall_at:       recall cutoffs
    n_values:        if provided, evaluate at each corpus size by restricting to the
                     first n doc IDs (shuffled by embed_dataset) and filtering to
                     queries whose relevant docs are all within that prefix.
                     Each list element is then a dict keyed by n instead of a flat metrics dict.

    Returns a list parallel to the input.
    """
    mappings   = [mapping_or_list] if isinstance(mapping_or_list, dict) else mapping_or_list
    qrels_list = [qrels_or_list]   if isinstance(qrels_or_list,   dict) else qrels_or_list

    results = []
    for mapping, qrels in zip(mappings, qrels_list):
        doc_ids   = list(mapping["docs"].keys())        # ["John Smith", "Betty Rose", ...]
        query_ids = list(mapping["queries"].keys())     # ["query_31", "query_14", ...]

        doc_embs    = np.stack([mapping["docs"][d]    for d in doc_ids])
        qry_embs    = np.stack([mapping["queries"][q] for q in query_ids])
        scores_full = qry_embs @ doc_embs.T           # (n_queries, n_docs)

        if n_values is None:
            doc_pos = {d: i for i, d in enumerate(doc_ids)}                 # {"John Smith": 0, "Betty Rose": 1, ...}
            rel_idx = [[doc_pos[d] for d in qrels[q]] for q in query_ids]   # qrels["query_31"] = {"Alice Jones": 1} <- or multiple
                                                                            # rel_idx[0] = [doc_pos[d] for d in {"Alice Jones": 1}] = [254]
            r = retrieval_metrics(scores_full, rel_idx, recall_at)
            r["n_queries"] = len(query_ids)
            results.append(r)
        else:
            qry_pos   = {q: i for i, q in enumerate(query_ids)}
            n_results = {}
            for n in sorted(n_values):
                subset_ids = set(doc_ids[:n])
                subset_pos = {d: i for i, d in enumerate(doc_ids[:n])}
                valid_qids = [q for q in query_ids if all(d in subset_ids for d in qrels[q])]
                if not valid_qids:
                    continue
                qi_rows    = np.array([qry_pos[q] for q in valid_qids])
                scores_sub = scores_full[qi_rows, :n]
                rel_idx    = [[subset_pos[d] for d in qrels[q]] for q in valid_qids]
                r = retrieval_metrics(scores_sub, rel_idx, recall_at)
                r["n_queries"] = len(valid_qids)
                n_results[n]   = r
            results.append(n_results)

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

    datasets, _, meta = build_disjoint_dataset(n=n, m_max=m_max)
    mappings = embed_dataset(datasets, model_name, name=f"disjoint_n{n}", meta=meta, cache=True, cache_dir=cache_dir, models_dir=models_dir, device=device, batch_size=batch_size)

    print(f"Model : {model_name}")
    print(f"n     : {n} documents  |  k = {k}")
    print(f"{'m':>4}  {'mean_nn_dist':>13}  {'topk_gap':>10}  {'anisotropy':>11}")
    print("-" * 48)

    results: dict[int, dict] = {}
    for m, mapping in zip(meta, mappings):
        doc_ids = list(mapping["docs"].keys())
        m_embs  = np.stack([mapping["docs"][d] for d in doc_ids])
        n_m     = len(doc_ids)

        cos_sim = m_embs @ m_embs.T
        dists   = np.sqrt(np.clip(2.0 - 2.0 * cos_sim, 0.0, None))
        np.fill_diagonal(dists, np.inf)
        sorted_dists = np.sort(dists, axis=1)

        mean_nn_dist = float(sorted_dists[:, 0].mean())
        topk_gap     = float(np.mean(sorted_dists[:, k] - sorted_dists[:, k - 1]))
        anisotropy   = float(cos_sim[~np.eye(n_m, dtype=bool)].mean())

        results[m] = {"mean_nn_dist": mean_nn_dist, "topk_gap": topk_gap, "anisotropy": anisotropy}
        r = results[m]
        print(f"{m:>4}  {r['mean_nn_dist']:>13.4f}  {r['topk_gap']:>10.4f}  {r['anisotropy']:>11.4f}")

    return results


def plot_results(results: list[dict], meta: list) -> None:
    ks  = sorted(int(k.split("@")[1]) for k in results[0] if k.startswith("recall@"))
    mrr = [r["mrr"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, k in enumerate(ks):
        ax.plot(meta, [r[f"recall@{k}"] for r in results], marker="os^Dv"[i % 5], label=f"Recall@{k}")
    ax.plot(meta, mrr, marker="x", linestyle="--", label="MRR")
    ax.set_xlabel("parameter")
    ax.set_ylabel("Score")
    ax.set_xticks(meta)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_embed_distance(results: dict[int, dict]) -> None:
    ms         = sorted(results)
    mean_nn    = [results[m]["mean_nn_dist"] for m in ms]
    topk_gap   = [results[m]["topk_gap"]     for m in ms]
    anisotropy = [results[m]["anisotropy"]   for m in ms]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(ms, mean_nn,    marker="o", color="steelblue");  axes[0].set_title("Mean nearest-neighbour distance");   axes[0].set_ylabel("Euclidean distance")
    axes[1].plot(ms, topk_gap,   marker="s", color="darkorange"); axes[1].set_title("Top-k gap (rank k vs k+1)");          axes[1].set_ylabel("Distance gap")
    axes[2].plot(ms, anisotropy, marker="^", color="seagreen");   axes[2].set_title("Anisotropy (avg pairwise cos-sim)");  axes[2].set_ylabel("Cosine similarity")
    for ax in axes:
        ax.set_xlabel("m (items per person)")
        ax.set_xticks(ms)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()