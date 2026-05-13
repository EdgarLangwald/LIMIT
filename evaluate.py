import numpy as np
import matplotlib.pyplot as plt
from create_datasets import build_disjoint_dataset
from name_item_pool import load_pool
from embed import embed_dataset

MODEL_NAME = "BGE_L"


def evaluate(
    doc_embs,          # (n_docs, dim), mmap-compatible
    qry_embs,          # (n_queries, dim), mmap-compatible
    qrels,             # list[list[int]] — relevant doc indices per query, aligned with qry_embs rows
    n_values,          # list[int]
    q_bs,              # query batch size
    doc_bs: int = 4096,
    seed:   int = 42,
    ks:     list[int] = [1, 5],
) -> dict[int, dict]:
    # CSR layout — precomputed once outside the n loop
    rel_lens = np.array([len(r) for r in qrels], dtype=np.int64)
    rel_ptr  = np.empty(len(qrels) + 1, dtype=np.int64)
    rel_ptr[0] = 0
    np.cumsum(rel_lens, out=rel_ptr[1:])
    rel_flat = np.concatenate([np.asarray(r, dtype=np.int64) for r in qrels])

    rng      = np.random.default_rng(seed)
    shuffled = rng.permutation(len(doc_embs))

    results = {}
    for n in sorted(n_values):
        doc_subs = np.sort(shuffled[:n])    # sort for sequential mmap reads

        in_subset           = np.zeros(len(doc_embs), dtype=bool)
        in_subset[doc_subs] = True
        counts              = np.add.reduceat(in_subset[rel_flat].view(np.uint8), rel_ptr[:-1])
        valid_qs            = np.where(counts == rel_lens)[0]
        if not valid_qs.size:
            continue

        recall_hits = {k: [] for k in ks}
        mrr_vals    = []

        for i in range(0, len(valid_qs), q_bs):
            q_batch  = valid_qs[i : i + q_bs]                        # (q_bs,)
            q_vecs   = np.asarray(qry_embs[q_batch])                 # (q_bs, dim)
            rel_docs = [rel_flat[rel_ptr[qi] : rel_ptr[qi + 1]] for qi in q_batch]

            r_scores_b = [doc_embs[rel] @ q_vecs[bi] for bi, rel in enumerate(rel_docs)]
            better     = [np.zeros(len(r), dtype=np.int64) for r in rel_docs]

            for start in range(0, n, doc_bs):
                chunk = doc_embs[doc_subs[start : start + doc_bs]]   # (doc_bs, dim)
                s     = q_vecs @ chunk.T                              # (q_bs, doc_bs)
                for bi, (r_s, bet) in enumerate(zip(r_scores_b, better)):
                    bet += (s[bi][None, :] > r_s[:, None]).sum(axis=1)

            for bet, r_s in zip(better, r_scores_b):
                ranks = bet + 1
                mrr_vals.append(float(1.0 / ranks.min()))
                for k in ks:
                    recall_hits[k].append(float((ranks <= k).mean()))

        out = {f"recall@{k}": float(np.mean(recall_hits[k])) for k in ks}
        out["mrr"]       = float(np.mean(mrr_vals))
        out["n_queries"] = len(valid_qs)
        results[n]       = out

    return results


def eval_embed_distance(
    n: int = 100,
    m_max: int | None = None,
    k: int = 5,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
    device: str | None = None,
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
    mappings = embed_dataset(datasets, model_name, dataset_name=f"disjoint_n{n}", force=False,device=device, batch_size=batch_size)

    print(f"Model : {model_name}")
    print(f"n     : {n} documents  |  k = {k}")
    print(f"{'m':>4}  {'mean_nn_dist':>13}  {'topk_gap':>10}  {'anisotropy':>11}")
    print("-" * 48)

    results: dict[int, dict] = {}
    for m, mapping in zip(meta, mappings):
        m_embs = mapping["doc_embs"]
        n_m    = len(m_embs)

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