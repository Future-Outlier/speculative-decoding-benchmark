"""Turn S1 results into the five figures + a paired-bootstrap summary table.

Usage: python ssd_stale_exp/plots.py --results <dir with s1_*.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALGOS = ("dflash", "dspark")
MODES = ("fresh", "gap", "self_kv")
COLORS = {"fresh": "#2266aa", "gap": "#cc5522", "self_kv": "#228855"}
K = 7


PREFIX = "s1"


def load_rounds(results: Path, algo: str, mode: str) -> list[dict]:
    path = results / f"{PREFIX}_{algo}_{mode}.rounds.jsonl"
    rows = []
    with path.open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def per_sample_tau(rows: list[dict]) -> dict[int, float]:
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r["uid"]][0] += r["committed"]
        agg[r["uid"]][1] += 1
    return {uid: c / n for uid, (c, n) in agg.items()}


def bootstrap_diff(
    tau_a: dict[int, float], tau_b: dict[int, float], iters: int = 4000
) -> tuple[float, float, float]:
    uids = sorted(set(tau_a) & set(tau_b))
    diffs = [tau_a[u] - tau_b[u] for u in uids]
    point = sum(diffs) / len(diffs)
    rng = random.Random(0)
    stats = []
    for _ in range(iters):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    return point, stats[int(0.025 * iters)], stats[int(0.975 * iters)]


def pooled_tau(rows: list[dict]) -> float:
    return sum(r["committed"] for r in rows) / len(rows)


def alpha_k(rows: list[dict]) -> list[float | None]:
    out = []
    for k in range(K):
        prop = sum(1 for r in rows if r["effective_proposal_len"] > k)
        acc = sum(1 for r in rows if r["accepted_draft"] > k)
        out.append(acc / prop if prop else None)
    return out


def pmf(rows: list[dict]) -> list[float]:
    hist = [0] * (K + 2)
    for r in rows:
        hist[min(r["committed"], K + 1)] += 1
    n = len(rows)
    return [c / n for c in hist]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="s1")
    args = ap.parse_args()
    global PREFIX
    PREFIX = args.prefix
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {
        (a, m): load_rounds(results, a, m) for a in ALGOS for m in MODES
    }

    # 1. paired tau with bootstrap CI ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    table_lines = []
    for ax, algo in zip(axes, ALGOS):
        taus = [pooled_tau(data[(algo, m)]) for m in MODES]
        ax.bar(MODES, taus, color=[COLORS[m] for m in MODES], width=0.6)
        for i, t in enumerate(taus):
            ax.text(i, t + 0.04, f"{t:.2f}", ha="center", fontsize=9)
        fresh_ps = per_sample_tau(data[(algo, "fresh")])
        for m in ("gap", "self_kv"):
            d, lo, hi = bootstrap_diff(fresh_ps, per_sample_tau(data[(algo, m)]))
            table_lines.append(
                f"{algo:7s} fresh-vs-{m:8s} dtau={d:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]"
            )
        ax.set_title(algo)
        ax.set_ylabel("tau (tokens/round)")
    fig.suptitle("Accepted length: fresh vs stale (S1, Qwen3-4B, temp 1.0)")
    fig.tight_layout()
    fig.savefig(out / "tau_paired.png", dpi=150)

    # 2. alpha_k ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        for m in MODES:
            ys = alpha_k(data[(algo, m)])
            ax.plot(range(1, K + 1), ys, marker="o", label=m, color=COLORS[m])
        ax.set_title(algo)
        ax.set_xlabel("position k")
        ax.set_ylabel("alpha_k = P(accept k | reached k)")
        ax.legend()
    fig.suptitle("Per-position cumulative acceptance")
    fig.tight_layout()
    fig.savefig(out / "alpha_k.png", dpi=150)

    # 3. committed-length PMF --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        for m in MODES:
            ys = pmf(data[(algo, m)])
            ax.plot(range(len(ys)), ys, marker="s", label=m, color=COLORS[m])
        ax.set_title(algo)
        ax.set_xlabel("committed tokens per round")
        ax.set_ylabel("P(L = l)")
        ax.legend()
    fig.suptitle("Committed-length distribution (drives SSD fan-out)")
    fig.tight_layout()
    fig.savefig(out / "pmf.png", dpi=150)

    # 4. correction recovery-rank CDF ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    ks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    for ax, algo in zip(axes, ALGOS):
        for m in MODES:
            ranks = [
                r["recovery_rank"]
                for r in data[(algo, m)]
                if r["recovery_kind"] == "correction"
            ]
            ys = [sum(1 for x in ranks if x <= k) / len(ranks) for k in ks]
            ax.plot(ks, ys, marker="o", label=m, color=COLORS[m])
        ax.set_xscale("log")
        ax.set_title(algo)
        ax.set_xlabel("fan-out per depth F (top-F)")
        ax.set_ylabel("P(correction token in top-F)")
        ax.legend()
    fig.suptitle("Recovery-token coverage vs fan-out (SSD p_hit driver)")
    fig.tight_layout()
    fig.savefig(out / "rank_cdf.png", dpi=150)

    # 5. tau vs staleness tokens -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        for m in ("gap", "self_kv"):
            groups = defaultdict(list)
            for r in data[(algo, m)]:
                groups[r["missing_fresh_rows"]].append(r["committed"])
            xs = sorted(k for k in groups if len(groups[k]) >= 20)
            ys = [sum(groups[x]) / len(groups[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=m, color=COLORS[m])
        fresh_tau = pooled_tau(data[(algo, "fresh")])
        ax.axhline(fresh_tau, ls="--", color=COLORS["fresh"], label="fresh mean")
        ax.set_title(algo)
        ax.set_xlabel("missing fresh rows at propose (prev round l)")
        ax.set_ylabel("tau")
        ax.legend()
    fig.suptitle("Degradation vs hole size (groups with n>=20)")
    fig.tight_layout()
    fig.savefig(out / "tau_vs_staleness.png", dpi=150)

    report = "\n".join(table_lines)
    (out / "paired_bootstrap.txt").write_text(report + "\n")
    print(report)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
