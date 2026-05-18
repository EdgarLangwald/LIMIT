import json
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from create_datasets import increase_param
from name_item_pool import load_pool
from embed import embed_dataset

def evaluate(
    doc_embs,               # (n_docs, dim), mmap-compatible
    qry_embs,               # (n_queries, dim), mmap-compatible
    qrels,                  # list[list[int]] — relevant doc indices per query, aligned with qry_embs rows
    n_values,               # list[int]
    q_bs,                   # query batch size
    doc_bs:   int = 4096,
    seed:     int = 42,
    ks:       list[int] = [1, 5],
    fixed_rel_size: int | None = None,  # set to fixed rel-docs-per-query for vectorised path; None for variable
    save_json: str | bool = False,      # path to accumulating JSON, or True for "results.json"
    device:   str | None = None,        # e.g. "cuda", "cuda:1", "cpu"; None → auto-detect
) -> dict[int, dict]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    t0 = time.time()
    rel_lens = np.array([len(r) for r in qrels], dtype=np.int64)
    rel_ptr  = np.empty(len(qrels) + 1, dtype=np.int64)
    rel_ptr[0] = 0
    np.cumsum(rel_lens, out=rel_ptr[1:])
    rel_flat = np.concatenate([np.asarray(r, dtype=np.int64) for r in qrels])

    rng      = np.random.default_rng(seed)
    shuffled = rng.permutation(len(doc_embs))

    # For each query: smallest n at which all its relevant docs are in shuffled[:n]
    pos           = np.empty(len(doc_embs), dtype=np.int64)
    pos[shuffled] = np.arange(len(doc_embs), dtype=np.int64)
    valid_at_n    = np.maximum.reduceat(pos[rel_flat], rel_ptr[:-1]) + 1
    print(f"  [pre-work] CSR + permutation + valid_at_n: {time.time() - t0:.2f}s")

    # Precompute relevant-doc scores for all queries once (not per-n)
    n_queries = len(qrels)
    t_pre = time.time()
    if fixed_rel_size is not None:
        rel_idx_all  = rel_flat.reshape(n_queries, fixed_rel_size)
        r_scores_all = torch.empty((n_queries, fixed_rel_size), dtype=torch.float32, device=dev)
        for i in range(0, n_queries, q_bs):
            q_end    = min(i + q_bs, n_queries)
            q_vecs   = torch.from_numpy(np.asarray(qry_embs[i:q_end])).to(dev)
            rel_docs = torch.from_numpy(np.asarray(doc_embs[rel_idx_all[i:q_end]])).to(dev)
            r_scores_all[i:q_end] = torch.einsum('qrd,qd->qr', rel_docs, q_vecs)
        better_all = torch.zeros((n_queries, fixed_rel_size), dtype=torch.int64, device=dev)
    else:
        r_scores_flat = torch.empty(len(rel_flat), dtype=torch.float32, device=dev)
        for i in range(0, n_queries, q_bs):
            q_end  = min(i + q_bs, n_queries)
            q_vecs = torch.from_numpy(np.asarray(qry_embs[i:q_end])).to(dev)
            for bi, qi in enumerate(range(i, q_end)):
                sl      = slice(int(rel_ptr[qi]), int(rel_ptr[qi + 1]))
                rel_doc = torch.from_numpy(np.asarray(doc_embs[rel_flat[sl]])).to(dev)
                r_scores_flat[sl] = rel_doc @ q_vecs[bi]
        better_flat = torch.zeros(len(rel_flat), dtype=torch.int64, device=dev)
    print(f"  [pre-work] relevant-doc scores: {time.time() - t_pre:.2f}s")

    results = {}
    n_prev  = 0
    for n in sorted(n_values):
        t_n = time.time()

        # Only score docs added since the previous checkpoint
        new_docs = np.sort(shuffled[n_prev:n])  # sorted for sequential mmap reads

        for start in range(0, len(new_docs), doc_bs):
            chunk = torch.from_numpy(np.asarray(doc_embs[new_docs[start:start + doc_bs]])).to(dev)
            for i in range(0, n_queries, q_bs):
                q_end  = min(i + q_bs, n_queries)
                q_vecs = torch.from_numpy(np.asarray(qry_embs[i:q_end])).to(dev)
                s      = q_vecs @ chunk.T                                     # (q_bs, chunk_bs) on GPU
                if fixed_rel_size is not None:
                    better_all[i:q_end] += (s[:, None, :] > r_scores_all[i:q_end, :, None]).sum(dim=2)
                else:
                    for bi, qi in enumerate(range(i, q_end)):
                        r_s = r_scores_flat[rel_ptr[qi]:rel_ptr[qi + 1]]
                        better_flat[rel_ptr[qi]:rel_ptr[qi + 1]] += (s[bi][None, :] > r_s[:, None]).sum(dim=1)

        n_prev   = n
        valid_qs = np.where(valid_at_n <= n)[0]
        if not valid_qs.size:
            continue

        if fixed_rel_size is not None:
            valid_qs_t  = torch.from_numpy(valid_qs).to(dev)
            ranks       = better_all[valid_qs_t] + 1                        # (valid_qs, R) on GPU
            mrr_vals    = (1.0 / ranks.min(dim=1).values).tolist()
            recall_hits = {k: (ranks <= k).float().mean(dim=1).tolist() for k in ks}
        else:
            mrr_vals    = []
            recall_hits = {k: [] for k in ks}
            for qi in valid_qs:
                ranks = better_flat[rel_ptr[qi]:rel_ptr[qi + 1]] + 1
                mrr_vals.append(float(1.0 / ranks.min()))
                for k in ks:
                    recall_hits[k].append(float((ranks <= k).float().mean()))

        out              = {f"recall@{k}": float(np.mean(recall_hits[k])) for k in ks}
        out["mrr"]       = float(np.mean(mrr_vals))
        out["n_queries"] = len(valid_qs)
        results[n]       = out
        print(f"  [n={n}] done in {time.time() - t_n:.1f}s  ({len(valid_qs)} queries)")

        if save_json:
            json_file = save_json if isinstance(save_json, str) else "results.json"
            existing = {}
            if os.path.exists(json_file):
                with open(json_file) as f:
                    existing = json.load(f)
            existing[str(n)] = out
            with open(json_file, "w") as f:
                json.dump(existing, f, indent=2)

    return results


def eval_embed_distance(
    n: int = 100,
    m_max: int | None = None,
    k: int = 5,
    model_name: str = "model",
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

    results_raw = increase_param("build_disjoint_dataset", "m", range(1, m_max + 1), n=n)
    datasets = [r[0] for r in results_raw]
    meta = list(range(1, m_max + 1))
    mappings = embed_dataset(datasets, model_name, dataset_name=f"disjoint_n{n}", force=False, device=device, batch_size=batch_size)

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


def evaluate_preference(
    doc_embs,           # (n_docs, dim)
    qry_embs,           # (n_queries, dim), aligned with qrels and sentiments
    qrels,              # list[list[int]]
    sentiments,         # list[dict] aligned with qrels; each has {type, liker_idx, disliker_idx}
    ks: list[int] = [1, 2],
    margin: float = 0.05,
    device: str | None = None,
    file_name: str | None = None,
) -> dict:
    """
    Evaluate a preference dataset on three metrics:
      a) recall@k per query type (like / dislike / neutral)
      b) sentiment capture rate with margin: fraction of like-queries where
         score(liker) > score(disliker) + margin, and likewise for dislike-queries
      c) bias of neutral queries: mean(score(liker) - score(disliker)) and the
         fraction of neutral queries where the liker scores higher
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    doc_t = torch.from_numpy(np.asarray(doc_embs)).to(dev)   # (n_docs, dim)
    qry_t = torch.from_numpy(np.asarray(qry_embs)).to(dev)   # (n_queries, dim)
    S = qry_t @ doc_t.T                                       # (n_queries, n_docs)
    n_docs = doc_t.shape[0]

    like_qs    = [i for i, s in enumerate(sentiments) if s["type"] == "like"]
    dislike_qs = [i for i, s in enumerate(sentiments) if s["type"] == "dislike"]
    neutral_qs = [i for i, s in enumerate(sentiments) if s["type"] == "neutral"]

    out = {}

    # a) recall@k per type
    for type_name, qs in [("like", like_qs), ("dislike", dislike_qs), ("neutral", neutral_qs)]:
        if not qs:
            continue
        for k in ks:
            recalls = []
            for qi in qs:
                rel = qrels[qi]
                topk = torch.topk(S[qi], min(k, n_docs)).indices.cpu().numpy()
                recalls.append(sum(1 for r in rel if r in topk) / len(rel))
            out[f"recall@{k}_{type_name}"] = float(np.mean(recalls))

    # b) sentiment capture with margin
    like_capture = [
        float(S[qi, sentiments[qi]["liker_idx"]].item() > S[qi, sentiments[qi]["disliker_idx"]].item() + margin)
        for qi in like_qs
    ]
    dislike_capture = [
        float(S[qi, sentiments[qi]["disliker_idx"]].item() > S[qi, sentiments[qi]["liker_idx"]].item() + margin)
        for qi in dislike_qs
    ]
    if like_capture:
        out["sentiment_capture_like"] = float(np.mean(like_capture))
    if dislike_capture:
        out["sentiment_capture_dislike"] = float(np.mean(dislike_capture))

    # c) neutral query bias
    if neutral_qs:
        bias = np.array([
            S[qi, sentiments[qi]["liker_idx"]].item() - S[qi, sentiments[qi]["disliker_idx"]].item()
            for qi in neutral_qs
        ])
        out["neutral_bias_mean"] = float(bias.mean())          # + → liker-favored
        out["neutral_bias_liker_frac"] = float((bias > 0).mean())  # fraction preferring liker

    if file_name:
        _plot_preference_results(out, ks, margin, file_name)
        with open(f"{file_name}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {file_name}.png / {file_name}.json")

    return out


def _plot_preference_results(out: dict, ks: list[int], margin: float, file_name: str) -> None:
    types = [t for t in ("like", "dislike", "neutral") if any(f"recall@{k}_{t}" in out for k in ks)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(file_name)

    # a) recall@k grouped by query type
    x = np.arange(len(types))
    width = 0.8 / len(ks)
    for i, k in enumerate(ks):
        vals = [out.get(f"recall@{k}_{t}", 0.0) for t in types]
        axes[0].bar(x + (i - (len(ks) - 1) / 2) * width, vals, width, label=f"recall@{k}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(types)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Recall@k by query type")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # b) sentiment capture
    cap_labels = [l for l in ("like", "dislike") if f"sentiment_capture_{l}" in out]
    cap_vals = [out[f"sentiment_capture_{l}"] for l in cap_labels]
    axes[1].bar(cap_labels, cap_vals, color=["steelblue", "coral"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Sentiment capture (margin={margin})")
    axes[1].grid(True, alpha=0.3)

    # c) neutral bias
    bias_labels = ["bias mean", "liker frac"]
    bias_vals = [out.get("neutral_bias_mean", 0.0), out.get("neutral_bias_liker_frac", 0.0)]
    axes[2].bar(bias_labels, bias_vals, color=["seagreen" if v >= 0 else "crimson" for v in bias_vals])
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Neutral query bias (+ = liker-favored)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{file_name}.png", dpi=150)
    plt.close(fig)


def plot_results(results: list[dict], meta: list, file_name: str = "", show: bool = True) -> None:
    ks  = sorted(int(k.split("@")[1]) for k in results[0] if k.startswith("recall@"))
    mrr = [r["mrr"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, k in enumerate(ks):
        ax.plot(meta, [r[f"recall@{k}"] for r in results], marker="os^Dv"[i % 5], label=f"Recall@{k}")
    ax.plot(meta, mrr, marker="x", linestyle="--", label="MRR")
    ax.set_xlabel("n")
    ax.set_ylabel("Score")
    ax.set_xticks(meta)
    ax.set_xticklabels([f"{n/1000:.0f}k" if n >= 1000 else str(n) for n in meta])
    ax.set_ylim(0, 1.05)
    if file_name:
        ax.set_title(file_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        fname = f"{file_name}.png" if file_name else "results.png"
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved plot: {fname}")


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