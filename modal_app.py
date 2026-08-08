"""Modal app for the SSD staleness experiment (S0 correctness, S1 signal).

Usage:
  modal run speculative-decoding-benchmark/modal_app.py --action download
  modal run speculative-decoding-benchmark/modal_app.py --action s0
  modal run speculative-decoding-benchmark/modal_app.py --action s1
  modal run speculative-decoding-benchmark/modal_app.py --action report

Budget model:
  - cumulative cost persists in /results/costlog.json
  - every run PRE-CHARGES a worst-case reserve before launching, then adjusts
    to actual afterwards, so hangs / OOMs / Modal timeouts stay accounted for
  - a run only starts if (spent + its full reserve) fits inside BUDGET_USD
  - the estimate includes configured GPU/CPU/memory and a regional safety factor
  - each subprocess has its own hard timeout; any failure aborts the stage
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "ssd-stale-exp"
GPU_KIND = os.environ.get("MODAL_GPU", "A10")
# Official base rates, modal.com/pricing (fetched 2026-08-08), USD/s.
GPU_RATE_USD_S = {
    "A10": 0.000306,
    "L4": 0.000222,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}
CPU_CORES = 4
MEMORY_GIB = 16
CPU_RATE_USD_CORE_S = 0.0000131
MEMORY_RATE_USD_GIB_S = 0.00000222
REGIONAL_SAFETY_FACTOR = 1.75
HARD_BUDGET_USD = 3.0
BUDGET_USD = float(os.environ.get("STALE_BUDGET_USD", str(HARD_BUDGET_USD)))
if not 0.0 < BUDGET_USD <= HARD_BUDGET_USD:
    raise ValueError(
        f"STALE_BUDGET_USD must be in (0, {HARD_BUDGET_USD}], got {BUDGET_USD}"
    )
RATE_USD_S = (
    GPU_RATE_USD_S[GPU_KIND]
    + CPU_CORES * CPU_RATE_USD_CORE_S
    + MEMORY_GIB * MEMORY_RATE_USD_GIB_S
) * REGIONAL_SAFETY_FACTOR
DOWNLOAD_RATE_USD_S = (
    CPU_CORES * CPU_RATE_USD_CORE_S + 8 * MEMORY_RATE_USD_GIB_S
) * REGIONAL_SAFETY_FACTOR

# Per-run worst-case wall-clock reserve (also the subprocess timeout).
RUN_RESERVE_S = {"s0": 300, "s1": 600, "s1t0": 600}
DOWNLOAD_RESERVE_S = 1800

TARGET = "Qwen/Qwen3-4B"
DRAFTS = {
    "dflash": "deepseek-ai/dflash_qwen3_4b_block7",
    "dspark": "deepseek-ai/dspark_qwen3_4b_block7",
}
MODES = ("fresh", "gap", "self_kv")

LOCAL_BENCHMARK = Path(__file__).resolve().parent
LOCAL_DEEPSPEC = LOCAL_BENCHMARK.parent


def _local_git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        encoding="utf-8",
    ).strip()


def _local_git_dirty(path: Path, pathspec: tuple[str, ...] = ()) -> bool:
    cmd = ["git", "-C", str(path), "status", "--porcelain"]
    if pathspec:
        cmd.extend(["--", *pathspec])
    return bool(
        subprocess.check_output(
            cmd,
            encoding="utf-8",
        ).strip()
    )


DEEPSPEC_GIT_SHA = _local_git_sha(LOCAL_DEEPSPEC)
BENCHMARK_GIT_SHA = _local_git_sha(LOCAL_BENCHMARK)
DEEPSPEC_GIT_DIRTY = _local_git_dirty(
    LOCAL_DEEPSPEC,
    ("deepspec", "eval_datasets", "requirements.txt", "pyproject.toml"),
)
BENCHMARK_GIT_DIRTY = _local_git_dirty(LOCAL_BENCHMARK)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.10.2",
        "numpy==2.4.4",
        "safetensors==0.7.0",
        "sentencepiece==0.2.1",
        "prettytable==3.17.0",
        "typing_extensions==4.15.0",
        "tqdm==4.67.3",
        "PyYAML==6.0.3",
        "hf_transfer",
    )
    .env(
        {
            "PYTHONPATH": "/repo",
            "DEEPSPEC_ROOT": "/repo",
            "HF_HOME": "/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_dir(
        str(LOCAL_DEEPSPEC / "deepspec"),
        remote_path="/repo/deepspec",
        ignore=[
            "**/__pycache__/**",
        ],
    )
    .add_local_dir(
        str(LOCAL_DEEPSPEC / "eval_datasets"),
        remote_path="/repo/eval_datasets",
    )
    .add_local_dir(
        str(LOCAL_BENCHMARK),
        remote_path="/repo/benchmark",
        ignore=[
            ".git",
            ".git/**",
            "**/__pycache__/**",
            "results/**",
            "figs/**",
        ],
    )
)

hf_vol = modal.Volume.from_name("deepspec-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ssd-stale-results", create_if_missing=True)

COSTLOG = Path("/results/costlog.json")
PINS = Path("/hf/pinned_revisions.json")


def _read_costlog() -> dict:
    # A reused Modal container does not automatically observe commits made by
    # another container.  This refreshes the volume before every guard read.
    results_vol.reload()
    if COSTLOG.exists():
        return json.loads(COSTLOG.read_text())
    return {"usd": 0.0, "entries": []}


def _append_costlog(kind: str, usd: float) -> dict:
    log = _read_costlog()
    log["usd"] = round(log["usd"] + usd, 4)
    log["entries"].append(
        {"kind": kind, "usd": round(usd, 4), "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    )
    COSTLOG.write_text(json.dumps(log, indent=2))
    results_vol.commit()
    return log


@app.function(
    image=image,
    volumes={"/hf": hf_vol, "/results": results_vol},
    timeout=DOWNLOAD_RESERVE_S,
    cpu=CPU_CORES,
    memory=8192,
    max_containers=1,
)
def download_models():
    from huggingface_hub import HfApi, snapshot_download

    reserve = DOWNLOAD_RESERVE_S * DOWNLOAD_RATE_USD_S
    log = _read_costlog()
    if log["usd"] + reserve > BUDGET_USD:
        raise RuntimeError(
            f"download estimate ${reserve:.2f} would exceed ${BUDGET_USD:.2f} budget"
        )
    _append_costlog("reserve:download", reserve)
    t0 = time.monotonic()
    try:
        api = HfApi()
        pins = {}
        for repo in [TARGET, *DRAFTS.values()]:
            sha = api.model_info(repo).sha
            print(f"[download] {repo} @ {sha}", flush=True)
            snapshot_download(repo, revision=sha)
            pins[repo] = sha
        PINS.write_text(json.dumps(pins, indent=2))
        hf_vol.commit()
        print(f"[download] done, pinned: {json.dumps(pins)}", flush=True)
    except Exception:
        raise
    else:
        elapsed = time.monotonic() - t0
        _append_costlog(
            "adjust:download", elapsed * DOWNLOAD_RATE_USD_S - reserve
        )


@app.function(
    image=image,
    gpu=GPU_KIND,
    volumes={"/hf": hf_vol, "/results": results_vol},
    timeout=2 * 3600,
    cpu=CPU_CORES,
    memory=MEMORY_GIB * 1024,
    max_containers=1,
)
def run_stage(
    stage: str,
    algos: list[str],
    modes: list[str],
    run_id: str,
    samples_per_task: int | None,
    max_new_tokens: int | None,
) -> dict:
    if stage not in ("s0", "s1", "s1t0"):
        raise ValueError(f"unknown stage: {stage}")
    if not algos or len(set(algos)) != len(algos) or not set(algos) <= set(DRAFTS):
        raise ValueError(f"algos must be a nonempty unique subset of {tuple(DRAFTS)}")
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODES):
        raise ValueError(f"modes must be a nonempty unique subset of {MODES}")
    if samples_per_task is not None and samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    results_vol.reload()
    hf_vol.reload()
    if not PINS.exists():
        raise FileNotFoundError(
            "run --action download first (pinned_revisions.json missing)"
        )
    pins = json.loads(PINS.read_text())
    required_pins = {TARGET, *(DRAFTS[algo] for algo in algos)}
    missing_pins = sorted(required_pins - set(pins))
    if missing_pins:
        raise ValueError(f"pinned revisions missing: {missing_pins}")

    allowed_run_id = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not run_id or any(ch not in allowed_run_id for ch in run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    out_dir = Path("/results/runs") / run_id / stage
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing artifacts: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    reserve_s = RUN_RESERVE_S[stage]
    reserve_usd = reserve_s * RATE_USD_S
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"

    statuses = []
    aborted = False
    for algo in algos:
        if aborted:
            break
        for mode in modes:
            log = _read_costlog()
            if log["usd"] + reserve_usd > BUDGET_USD:
                statuses.append(
                    {"algo": algo, "mode": mode, "status": "SKIPPED_BUDGET"}
                )
                print(
                    f"[budget] ${log['usd']:.2f} + reserve ${reserve_usd:.2f} "
                    f">= ${BUDGET_USD} — stop",
                    flush=True,
                )
                aborted = True
                break
            # Pre-charge the reserve; adjust to actual afterwards. A hang or
            # kill leaves the reserve charged (safe overcount).
            _append_costlog(f"reserve:{run_id}:{stage}:{algo}:{mode}", reserve_usd)
            cmd = [
                sys.executable,
                "/repo/benchmark/runner.py",
                "--algo", algo,
                "--mode", mode,
                "--stage", stage,
                "--target", TARGET,
                "--draft", DRAFTS[algo],
                "--target-revision", pins[TARGET],
                "--draft-revision", pins[DRAFTS[algo]],
                "--deepspec-git-sha", DEEPSPEC_GIT_SHA,
                "--benchmark-git-sha", BENCHMARK_GIT_SHA,
                "--require-revisions",
                "--require-cuda",
                "--dtype", "fp32" if stage == "s0" else "bf16",
                "--out-dir", str(out_dir),
            ]
            if DEEPSPEC_GIT_DIRTY:
                cmd.append("--deepspec-git-dirty")
            if BENCHMARK_GIT_DIRTY:
                cmd.append("--benchmark-git-dirty")
            if samples_per_task is not None:
                cmd.extend(["--samples-per-task", str(samples_per_task)])
            if max_new_tokens is not None:
                cmd.extend(["--max-new-tokens", str(max_new_tokens)])
            print(f"[run] {' '.join(cmd)}", flush=True)
            t0 = time.monotonic()
            try:
                proc = subprocess.run(cmd, env=env, timeout=reserve_s)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = -9
            dt = time.monotonic() - t0
            actual_usd = dt * RATE_USD_S
            # Persist runner artifacts before the ledger helper reloads the
            # shared volume to observe commits from other containers.
            results_vol.commit()
            _append_costlog(
                f"adjust:{run_id}:{stage}:{algo}:{mode}",
                round(actual_usd - reserve_usd, 4),
            )
            status = {0: "OK", -9: "TIMEOUT"}.get(code, f"EXIT_{code}")
            statuses.append(
                {"algo": algo, "mode": mode, "status": status, "seconds": round(dt, 1)}
            )
            if code != 0:
                print(f"[run] failure ({status}) — aborting stage", flush=True)
                aborted = True
                break

    log = _read_costlog()
    summaries = {}
    for p in sorted(out_dir.glob("*.summary.json")):
        summaries[p.stem] = json.loads(p.read_text())
    ok = all(s["status"] == "OK" for s in statuses) and len(statuses) == len(
        algos
    ) * len(modes)
    report = {
        "stage": stage,
        "run_id": run_id,
        "ok": ok,
        "statuses": statuses,
        "estimated_cumulative_usd": log["usd"],
        "budget_usd": BUDGET_USD,
        "hard_budget_usd": HARD_BUDGET_USD,
        "estimate_rate_usd_s": RATE_USD_S,
        "estimate_regional_safety_factor": REGIONAL_SAFETY_FACTOR,
        "gpu": GPU_KIND,
        "pins": pins,
        "summaries": summaries,
    }
    (out_dir / "stage_report.json").write_text(json.dumps(report, indent=2))
    results_vol.commit()
    print("[stage_report] " + json.dumps(report), flush=True)
    return report


@app.function(image=image, volumes={"/results": results_vol}, timeout=300)
def report() -> dict:
    out = {"costlog": _read_costlog(), "stages": {}}
    runs_root = Path("/results/runs")
    if runs_root.exists():
        for p in sorted(runs_root.glob("*/*/stage_report.json"))[-20:]:
            out["stages"][str(p.parent.relative_to(runs_root))] = json.loads(
                p.read_text()
            )
    print(json.dumps(out)[:30000], flush=True)
    return out


@app.local_entrypoint()
def main(
    action: str = "s0",
    algos: str = "dflash,dspark",
    modes: str = "",
    run_id: str = "",
    samples_per_task: int | None = None,
    max_new_tokens: int | None = None,
):
    algo_list = [a for a in algos.split(",") if a]
    mode_list = [m for m in modes.split(",") if m] or list(MODES)
    if action == "download":
        download_models.remote()
    elif action in ("s0", "s1", "s1t0"):
        if (
            not algo_list
            or len(set(algo_list)) != len(algo_list)
            or not set(algo_list) <= set(DRAFTS)
        ):
            raise SystemExit(
                f"algos must be a nonempty unique subset of {tuple(DRAFTS)}"
            )
        if (
            not mode_list
            or len(set(mode_list)) != len(mode_list)
            or not set(mode_list) <= set(MODES)
        ):
            raise SystemExit(
                f"modes must be a nonempty unique subset of {MODES}"
            )
        if samples_per_task is not None and samples_per_task <= 0:
            raise SystemExit("samples-per-task must be positive")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise SystemExit("max-new-tokens must be positive")
        resolved_run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        rep = run_stage.remote(
            action,
            algo_list,
            mode_list,
            resolved_run_id,
            samples_per_task,
            max_new_tokens,
        )
        print(json.dumps(rep.get("statuses", []), indent=2))
        print(
            "conservative estimated cumulative cost: "
            f"${rep.get('estimated_cumulative_usd')} / budget ${rep.get('budget_usd')}"
        )
        if not rep.get("ok"):
            raise SystemExit(1)
    elif action == "report":
        report.remote()
    else:
        raise SystemExit(f"unknown action {action}")
