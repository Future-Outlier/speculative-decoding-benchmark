"""Run one (algo, mode, stage) staleness experiment on a single device.

Usage (repo root on PYTHONPATH):
  python ssd_stale_exp/runner.py --algo dflash --mode gap --stage s1 \
      --target Qwen/Qwen3-4B --draft deepseek-ai/dflash_qwen3_4b_block7 \
      --out-dir /results

Stages:
  s0  correctness: few prompts, temperature 0, max 64 new tokens; the
      committed output of EVERY mode must equal target-only greedy.
  s1  cheap signal: manifest prompts (default 32/task x 3 tasks),
      temperature 1.0, max 256 new tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from deepspec.data.parser import encode_chat_messages  # noqa: E402
from deepspec.eval.base_evaluator import (  # noqa: E402
    assert_no_final_target_layer,
    generate_decoding_sample,
    resolve_stop_token_ids,
    trim_output_ids,
)
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # noqa: E402
from deepspec.utils import seed_all  # noqa: E402

from ssd_stale_exp.manifest import build_manifest  # noqa: E402
from ssd_stale_exp.stale_core import MODES, StaleController  # noqa: E402

ALGO_DRAFTS_4B = {
    "dflash": "deepseek-ai/dflash_qwen3_4b_block7",
    "dspark": "deepseek-ai/dspark_qwen3_4b_block7",
}


def build_models(
    target_name: str,
    draft_name: str,
    device: torch.device,
    dtype,
    target_revision: str | None = None,
    draft_revision: str | None = None,
):
    target = (
        AutoModelForCausalLM.from_pretrained(
            target_name,
            dtype=dtype,
            attn_implementation="sdpa",
            revision=target_revision,
        )
        .to(device)
        .eval()
    )
    draft = (
        Qwen3DSparkModel.from_pretrained(
            draft_name,
            dtype=dtype,
            attn_implementation="sdpa",
            revision=draft_revision,
        )
        .to(device)
        .eval()
    )
    assert_no_final_target_layer(target, draft.target_layer_ids)
    tokenizer = AutoTokenizer.from_pretrained(target_name, revision=target_revision)
    return target, draft, tokenizer


@torch.inference_mode()
def target_greedy(target, input_ids, max_new_tokens, stop_token_ids):
    from transformers import DynamicCache

    device = input_ids.device
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    position_ids = torch.arange(max_len, device=device).unsqueeze(0)
    cache = DynamicCache()
    out = target(
        input_ids=input_ids,
        position_ids=position_ids[:, :n_in],
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    ids = [input_ids]
    cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
    ids.append(cur)
    stop = torch.tensor(stop_token_ids, device=device) if stop_token_ids else None
    pos = n_in
    while pos + 1 < max_len:
        if stop is not None and bool(torch.isin(cur, stop).any()):
            break
        out = target(
            input_ids=cur,
            position_ids=position_ids[:, pos : pos + 1],
            past_key_values=cache,
            use_cache=True,
        )
        cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
        ids.append(cur)
        pos += 1
    output_ids = torch.cat(ids, dim=1)
    return trim_output_ids(output_ids, n_in, stop_token_ids)


@torch.inference_mode()
def run_one_sample(
    *,
    target,
    draft,
    input_ids,
    mode,
    temperature,
    max_new_tokens,
    confidence_threshold,
    stop_token_ids,
    device,
):
    controller = StaleController(
        model=draft,
        mode=mode,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    def init_context(*, initial_output, output_ids, position_ids, num_input_tokens):
        del output_ids, position_ids
        controller.init_prompt(initial_output, num_input_tokens)
        return controller

    def propose(*, context, output_ids, position_ids, start, stop_token_ids=None):
        del context
        return controller.propose(
            output_ids=output_ids,
            position_ids=position_ids,
            start=start,
            stop_token_ids=stop_token_ids,
        )

    def update(context, verification):
        del context
        controller.update(verification)

    def post_verify(proposal, verification):
        controller.post_verify(proposal, verification)

    result = generate_decoding_sample(
        target_model=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        max_proposal_tokens=int(draft.block_size),
        temperature=temperature,
        stop_token_ids=stop_token_ids,
        init_context=init_context,
        propose=propose,
        update=update,
        post_verify=post_verify,
    )
    return result, controller


def summarize(round_rows: list[dict], sample_rows: list[dict], K: int) -> dict:
    live = [r for r in round_rows]
    n_rounds = len(live)
    if n_rounds == 0:
        return {"rounds": 0}
    committed = sum(r["committed"] for r in live)
    proposed = sum(r["effective_proposal_len"] for r in live)
    tau = committed / n_rounds
    e_lens = [r["e_len_analytic"] for r in live if r["e_len_analytic"] is not None]
    tau_analytic = sum(e_lens) / len(e_lens) if e_lens else None
    # cumulative per-position acceptance, repo-style accept_rate@k
    prop_at = [0] * K
    acc_at = [0] * K
    for r in live:
        for k in range(K):
            if r["effective_proposal_len"] > k:
                prop_at[k] += 1
            if r["accepted_draft"] > k:
                acc_at[k] += 1
    alpha = [
        (acc_at[k] / prop_at[k]) if prop_at[k] else None for k in range(K)
    ]
    hist = [0] * (K + 2)
    for r in live:
        hist[min(r["committed"], K + 1)] += 1
    pmf = [round(c / n_rounds, 5) for c in hist]
    ranks = [
        r["recovery_rank"] for r in live if r["recovery_kind"] == "correction"
    ]
    rank_cdf = {}
    for topk in (1, 2, 4, 8, 16):
        if ranks:
            rank_cdf[f"top{topk}"] = sum(1 for x in ranks if x <= topk) / len(ranks)
    kinds = {}
    for r in live:
        kinds[r["recovery_kind"]] = kinds.get(r["recovery_kind"], 0) + 1
    t_prop = sorted(r["t_propose_ms"] for r in live)
    t_round = sorted(r["t_round_ms"] for r in live)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    stale_groups: dict[int, list[float]] = {}
    for r in live:
        stale_groups.setdefault(r["missing_fresh_rows"], []).append(r["committed"])
    tau_by_staleness = {
        str(k): {"n": len(v), "tau": sum(v) / len(v)}
        for k, v in sorted(stale_groups.items())
    }
    total_out = sum(s["n_output_tokens"] for s in sample_rows)
    total_wall = sum(s["wall_s"] for s in sample_rows)
    return {
        "rounds": n_rounds,
        "samples": len(sample_rows),
        "tau": round(tau, 4),
        "tau_analytic": round(tau_analytic, 4) if tau_analytic else None,
        "verify_rate": round(committed / (proposed + n_rounds), 4),
        "accept_rate_at_k": [round(a, 4) if a is not None else None for a in alpha],
        "committed_hist": hist,
        "committed_pmf": pmf,
        "recovery_kinds": kinds,
        "correction_rank_cdf": rank_cdf,
        "tau_by_staleness_tokens": tau_by_staleness,
        "t_propose_ms_p50": pct(t_prop, 0.5),
        "t_round_ms_p50": pct(t_round, 0.5),
        "t_round_ms_p90": pct(t_round, 0.9),
        "output_tokens": total_out,
        "tokens_per_s": round(total_out / total_wall, 2) if total_wall else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGO_DRAFTS_4B), required=True)
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--stage", choices=["s0", "s1", "s1t0"], required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft", default=None)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "ssd_stale_exp" / "results"))
    ap.add_argument("--decode-seed", type=int, default=980406)
    ap.add_argument("--subset-seed", type=int, default=20260805)
    ap.add_argument("--samples-per-task", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="fp32 for S0: batched-verify vs serial-greedy argmax must agree; "
        "bf16 tie-breaking flips make strict equality unreliable.",
    )
    ap.add_argument("--target-revision", default=None)
    ap.add_argument("--draft-revision", default=None)
    args = ap.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        print("[runner] FATAL: --require-cuda set but CUDA unavailable", flush=True)
        sys.exit(3)

    stage_defaults = {
        "s0": {"samples_per_task": 2, "max_new_tokens": 64, "temperature": 0.0},
        "s1": {"samples_per_task": 32, "max_new_tokens": 256, "temperature": 1.0},
        # Same measurement as s1 but greedy — no correctness oracle here
        # (that is s0's job); bf16 like deployment.
        "s1t0": {"samples_per_task": 32, "max_new_tokens": 256, "temperature": 0.0},
    }
    cfg = stage_defaults[args.stage]
    n_per_task = args.samples_per_task or cfg["samples_per_task"]
    max_new = args.max_new_tokens or cfg["max_new_tokens"]
    temperature = cfg["temperature"]
    draft_name = args.draft or ALGO_DRAFTS_4B[args.algo]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(
        f"[runner] algo={args.algo} mode={args.mode} stage={args.stage} "
        f"device={device} target={args.target} draft={draft_name}",
        flush=True,
    )
    target, draft, tokenizer = build_models(
        args.target,
        draft_name,
        device,
        dtype,
        target_revision=args.target_revision,
        draft_revision=args.draft_revision,
    )
    stop_token_ids = resolve_stop_token_ids(target, tokenizer)
    K = int(draft.block_size)

    manifest = build_manifest(
        dataset_root=str(REPO_ROOT / "eval_datasets"),
        samples_per_task=n_per_task,
        subset_seed=args.subset_seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.stage}_{args.algo}_{args.mode}"
    rounds_path = out_dir / f"{tag}.rounds.jsonl"
    samples_path = out_dir / f"{tag}.samples.jsonl"

    round_rows: list[dict] = []
    sample_rows: list[dict] = []
    mismatches = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with rounds_path.open("w") as rf, samples_path.open("w") as sf:
        for uid, item in enumerate(manifest):
            seed_all(args.decode_seed + uid)
            messages = [{"role": "user", "content": item["prompt"]}]
            input_ids = encode_chat_messages(
                tokenizer,
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            result, controller = run_one_sample(
                target=target,
                draft=draft,
                input_ids=input_ids,
                mode=args.mode,
                temperature=temperature,
                max_new_tokens=max_new,
                confidence_threshold=args.confidence_threshold,
                stop_token_ids=stop_token_ids,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall = time.perf_counter() - t0

            greedy_match = None
            if args.stage == "s0":
                seed_all(args.decode_seed + uid)
                ref = target_greedy(target, input_ids, max_new, stop_token_ids)
                greedy_match = bool(
                    ref.shape == result.output_ids.shape
                    and torch.equal(ref, result.output_ids)
                )
                if not greedy_match:
                    mismatches.append(
                        {
                            "uid": uid,
                            "dataset": item["dataset"],
                            "ref": ref[0, input_ids.shape[1]:].tolist(),
                            "got": result.output_ids[0, input_ids.shape[1]:].tolist(),
                        }
                    )

            out_tokens = result.output_ids[0, input_ids.shape[1]:].tolist()
            srow = {
                "uid": uid,
                "dataset": item["dataset"],
                "row_idx": item["row_idx"],
                "prompt_sha1": item["prompt_sha1"],
                "seed": args.decode_seed + uid,
                "n_input_tokens": int(input_ids.shape[1]),
                "n_output_tokens": int(result.num_output_tokens),
                "output_sha1": hashlib.sha1(
                    json.dumps(out_tokens).encode()
                ).hexdigest()[:12],
                "rounds": len(controller.rounds),
                "wall_s": round(wall, 3),
                "greedy_match": greedy_match,
            }
            sample_rows.append(srow)
            sf.write(json.dumps(srow) + "\n")
            for r in controller.rounds:
                row = {"uid": uid, **r.__dict__}
                round_rows.append(row)
                rf.write(json.dumps(row) + "\n")
            if (uid + 1) % 8 == 0:
                done = sum(s["n_output_tokens"] for s in sample_rows)
                print(
                    f"[runner] {uid + 1}/{len(manifest)} samples, "
                    f"{done} tokens, {time.perf_counter() - t0:.1f}s last",
                    flush=True,
                )

    import subprocess as sp

    import transformers

    try:
        git_sha = sp.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            encoding="utf-8",
        ).strip()
    except Exception:
        git_sha = "unknown"
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()[:16]

    summary = summarize(round_rows, sample_rows, K)
    summary.update(
        {
            "algo": args.algo,
            "mode": args.mode,
            "stage": args.stage,
            "target": args.target,
            "draft": draft_name,
            "target_revision": args.target_revision,
            "draft_revision": args.draft_revision,
            "temperature": temperature,
            "max_new_tokens": max_new,
            "decode_seed": args.decode_seed,
            "subset_seed": args.subset_seed,
            "manifest_sha256": manifest_sha,
            "device": str(device),
            "dtype": str(dtype),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "repo_git_sha": git_sha,
        }
    )
    if device.type == "cuda":
        summary["peak_mem_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 2**30, 2
        )
    if args.stage == "s0":
        summary["greedy_match_all"] = all(
            s["greedy_match"] for s in sample_rows
        )
        summary["mismatch_count"] = len(mismatches)
        if mismatches:
            (out_dir / f"{tag}.mismatches.json").write_text(
                json.dumps(mismatches, indent=2)
            )
    (out_dir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
    print("[summary] " + json.dumps(summary), flush=True)
    if args.stage == "s0" and not summary["greedy_match_all"]:
        print("[runner] S0 FAILED: outputs diverge from target-only greedy", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
