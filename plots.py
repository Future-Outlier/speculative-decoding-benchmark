"""Turn proxy results into four figures and a cluster-bootstrap summary.

Usage: python plots.py --results <dir with s1_*.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from .benchmark_stats import (  # type: ignore[import-not-found]
        add_effective_committed,
        paired_cluster_bootstrap_diff,
        paired_cluster_bootstrap_ratio,
        pooled_value_per_verification,
        validate_paired_samples,
    )
except ImportError:
    from benchmark_stats import (
        add_effective_committed,
        paired_cluster_bootstrap_diff,
        paired_cluster_bootstrap_ratio,
        pooled_value_per_verification,
        validate_paired_samples,
    )

ALGOS = ("dflash", "dspark")
MODES = ("fresh", "gap", "self_kv")
COLORS = {"fresh": "#2266aa", "gap": "#cc5522", "self_kv": "#228855"}
K = 7


PREFIX = "s1"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def load_arm(
    results: Path, algo: str, mode: str
) -> tuple[list[dict], dict[int, dict], dict]:
    stem = f"{PREFIX}_{algo}_{mode}"
    rows = load_jsonl(results / f"{stem}.rounds.jsonl")
    samples = {
        int(row["uid"]): row
        for row in load_jsonl(results / f"{stem}.samples.jsonl")
    }
    summary = json.loads((results / f"{stem}.summary.json").read_text())
    max_new_tokens = int(summary["max_new_tokens"])
    derived_rows = add_effective_committed(rows, samples, max_new_tokens)
    for original, derived in zip(rows, derived_rows):
        if (
            "committed_within_limit" in original
            and int(original["committed_within_limit"])
            != int(derived["committed_within_limit"])
        ):
            raise ValueError(
                f"{stem}: uid={original['uid']} round={original['round_idx']} "
                "has inconsistent committed_within_limit"
            )
    rows = derived_rows

    effective_by_uid = defaultdict(int)
    for row in rows:
        effective_by_uid[int(row["uid"])] += int(row["committed_within_limit"])
    for uid, sample in samples.items():
        returned_after_initial = max(0, int(sample["n_output_tokens"]) - 1)
        if effective_by_uid[uid] != returned_after_initial:
            raise ValueError(
                f"{stem}: uid={uid} effective committed={effective_by_uid[uid]} "
                f"but returned after initial token={returned_after_initial}"
            )
    if summary.get("algo") != algo or summary.get("mode") != mode:
        raise ValueError(
            f"{stem}: summary identity is algo={summary.get('algo')}, "
            f"mode={summary.get('mode')}"
        )
    return rows, samples, summary


def prefix_survival(rows: list[dict]) -> list[float | None]:
    out = []
    for k in range(K):
        prop = sum(1 for r in rows if r["effective_proposal_len"] > k)
        acc = sum(1 for r in rows if r["accepted_draft"] > k)
        out.append(acc / prop if prop else None)
    return out


def pmf(rows: list[dict], value_key: str) -> list[float]:
    hist = [0] * (K + 2)
    for r in rows:
        hist[min(r[value_key], K + 1)] += 1
    n = len(rows)
    return [c / n for c in hist]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="s1")
    ap.add_argument(
        "--metric",
        choices=("accepted_length", "returned_tokens"),
        default="accepted_length",
        help="accepted_length is the perfect-coverage surrogate quality metric; "
        "returned_tokens includes the finite-request boundary",
    )
    args = ap.parse_args()
    global PREFIX
    PREFIX = args.prefix
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    value_key = (
        "committed" if args.metric == "accepted_length" else "committed_within_limit"
    )
    metric_label = (
        "unclipped accepted length / verification"
        if args.metric == "accepted_length"
        else "returned tokens / verification"
    )
    metric_short_label = (
        "accepted length" if args.metric == "accepted_length" else "returned tokens"
    )

    data = {}
    reference_samples = None
    reference_summary = None
    common_summary_fields = (
        "stage",
        "target",
        "target_revision",
        "temperature",
        "max_new_tokens",
        "decode_seed",
        "subset_seed",
        "manifest_sha256",
        "dtype",
        "gpu_name",
        "torch_version",
        "transformers_version",
        "confidence_threshold",
        "deepspec_git_sha",
        "benchmark_source_sha256",
    )
    for algo in ALGOS:
        algo_draft = None
        for mode in MODES:
            rows, samples, summary = load_arm(results, algo, mode)
            if reference_samples is None:
                reference_samples = samples
                reference_summary = summary
            else:
                validate_paired_samples(reference_samples, samples)
                for field in common_summary_fields:
                    if summary.get(field) != reference_summary.get(field):
                        raise ValueError(
                            f"incompatible arm summary field {field}: "
                            f"{summary.get(field)!r} != {reference_summary.get(field)!r}"
                        )
            draft_identity = (summary.get("draft"), summary.get("draft_revision"))
            if algo_draft is None:
                algo_draft = draft_identity
            elif draft_identity != algo_draft:
                raise ValueError(f"{algo} modes use different draft revisions")
            data[(algo, mode)] = rows

    # 1. paired tau with bootstrap CI ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    table_lines = []
    for ax, algo in zip(axes, ALGOS):
        taus = [
            pooled_value_per_verification(data[(algo, m)], value_key)
            for m in MODES
        ]
        ax.bar(MODES, taus, color=[COLORS[m] for m in MODES], width=0.6)
        for i, t in enumerate(taus):
            ax.text(i, t + 0.04, f"{t:.2f}", ha="center", fontsize=9)
        for m in ("gap", "self_kv"):
            d, lo, hi = paired_cluster_bootstrap_diff(
                data[(algo, "fresh")],
                data[(algo, m)],
                value_key=value_key,
            )
            retention, retention_lo, retention_hi = paired_cluster_bootstrap_ratio(
                data[(algo, "fresh")],
                data[(algo, m)],
                value_key=value_key,
            )
            table_lines.append(
                f"{algo:7s} fresh-vs-{m:8s} pooled_delta={d:+.3f}  "
                f"95%CI [{lo:+.3f}, {hi:+.3f}]  "
                f"quality_retention={retention:.3f} "
                f"95%CI [{retention_lo:.3f}, {retention_hi:.3f}]"
            )
        nonboundary_fresh = [
            row for row in data[(algo, "fresh")]
            if int(row["boundary_excess_tokens"]) == 0
        ]
        nonboundary_self = [
            row for row in data[(algo, "self_kv")]
            if int(row["boundary_excess_tokens"]) == 0
        ]
        sensitivity, sensitivity_lo, sensitivity_hi = (
            paired_cluster_bootstrap_ratio(
                nonboundary_fresh,
                nonboundary_self,
                value_key=value_key,
            )
        )
        table_lines.append(
            f"{algo:7s} self_kv/fresh excluding boundary-affected rounds "
            f"quality_retention={sensitivity:.3f} "
            f"95%CI [{sensitivity_lo:.3f}, {sensitivity_hi:.3f}]"
        )
        ax.set_title(algo)
        ax.set_ylabel(metric_label)
    fig.suptitle(
        "Perfect-coverage retained-mask-KV quality surrogate (Qwen3-4B)"
    )
    fig.tight_layout()
    fig.savefig(out / "tau_paired.png", dpi=150)

    # 2. cumulative prefix survival --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        for m in MODES:
            ys = prefix_survival(data[(algo, m)])
            ax.plot(range(1, K + 1), ys, marker="o", label=m, color=COLORS[m])
        ax.set_title(algo)
        ax.set_xlabel("position k")
        ax.set_ylabel("P(first k accepted | effective proposal reaches k)")
        ax.legend()
    fig.suptitle("Conditional cumulative accepted-prefix survival")
    fig.tight_layout()
    fig.savefig(out / "alpha_k.png", dpi=150)

    # 3. committed-length PMF --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        for m in MODES:
            ys = pmf(data[(algo, m)], value_key)
            ax.plot(range(len(ys)), ys, marker="s", label=m, color=COLORS[m])
        ax.set_title(algo)
        ax.set_xlabel(metric_short_label)
        ax.set_ylabel("P(L = l)")
        ax.legend()
    fig.suptitle(f"Distribution of {metric_label}")
    fig.tight_layout()
    fig.savefig(out / "pmf.png", dpi=150)

    # 4. association with missing target-feature rows --------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        for m in ("gap", "self_kv"):
            groups = defaultdict(list)
            for r in data[(algo, m)]:
                groups[r["missing_fresh_rows"]].append(
                    r[value_key]
                )
            xs = sorted(k for k in groups if len(groups[k]) >= 20)
            ys = [sum(groups[x]) / len(groups[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=m, color=COLORS[m])
        fresh_tau = pooled_value_per_verification(
            data[(algo, "fresh")], value_key
        )
        ax.axhline(fresh_tau, ls="--", color=COLORS["fresh"], label="fresh mean")
        ax.set_title(algo)
        ax.set_xlabel("missing target-feature rows\n(previous-round outcome)")
        ax.set_ylabel(metric_label)
        ax.legend()
    fig.suptitle("Association with missing target-feature rows (n>=20 groups)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "tau_vs_staleness.png", dpi=150)

    report = "\n".join(table_lines)
    (out / "paired_bootstrap.txt").write_text(report + "\n")
    print(report)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
