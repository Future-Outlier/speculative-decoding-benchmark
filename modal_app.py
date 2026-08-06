"""Modal app for the SSD staleness experiment (S0 correctness, S1 signal).

Usage:
  modal run ssd_stale_exp/modal_app.py --action download
  modal run ssd_stale_exp/modal_app.py --action s0
  modal run ssd_stale_exp/modal_app.py --action s1
  modal run ssd_stale_exp/modal_app.py --action report

Budget model (post codex review):
  - cumulative cost persists in /results/costlog.json
  - every run PRE-CHARGES a worst-case reserve before launching, then adjusts
    to actual afterwards, so hangs / OOMs / Modal timeouts stay accounted for
  - a run only starts if (spent + its full reserve) fits inside BUDGET_USD
  - GPU rate is inflated by OVERHEAD to cover CPU/mem/cold-start
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
# Official on-demand rates, modal.com/pricing (fetched 2026-08-05), USD/h.
GPU_RATE_USD_H = {
    "A10": 1.10,
    "L4": 0.80,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
}
BUDGET_USD = float(os.environ.get("STALE_BUDGET_USD", "8.0"))
OVERHEAD = 1.15  # CPU/mem/cold-start on top of the GPU rate
RATE = GPU_RATE_USD_H[GPU_KIND] * OVERHEAD

# Per-run worst-case wall-clock reserve (also the subprocess timeout).
RUN_RESERVE_S = {"s0": 1200, "s1": 3600, "s1t0": 3600}

TARGET = "Qwen/Qwen3-4B"
DRAFTS = {
    "dflash": "deepseek-ai/dflash_qwen3_4b_block7",
    "dspark": "deepseek-ai/dspark_qwen3_4b_block7",
}
MODES = ("fresh", "gap", "self_kv")

LOCAL_REPO = Path(__file__).resolve().parent.parent

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
    .env({"PYTHONPATH": "/repo", "HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        str(LOCAL_REPO),
        remote_path="/repo",
        ignore=[
            ".git",
            ".git/**",
            "**/__pycache__/**",
            ".venv/**",
            "assets/**",
            "ssd_stale_exp/results/**",
        ],
    )
)

hf_vol = modal.Volume.from_name("deepspec-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ssd-stale-results", create_if_missing=True)

COSTLOG = Path("/results/costlog.json")
PINS = Path("/hf/pinned_revisions.json")


def _read_costlog() -> dict:
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


@app.function(image=image, volumes={"/hf": hf_vol}, timeout=3600, cpu=4, memory=8192)
def download_models():
    from huggingface_hub import HfApi, snapshot_download

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


@app.function(
    image=image,
    gpu=GPU_KIND,
    volumes={"/hf": hf_vol, "/results": results_vol},
    timeout=2 * 3600,
    memory=16384,
    max_containers=1,
)
def run_stage(stage: str, algos: list[str], modes: list[str]) -> dict:
    assert stage in ("s0", "s1", "s1t0")
    assert PINS.exists(), "run --action download first (pinned_revisions.json missing)"
    pins = json.loads(PINS.read_text())

    out_dir = Path(f"/results/{stage}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():  # never mix artifacts across invocations
        if old.is_file():
            old.unlink()
    results_vol.commit()

    reserve_s = RUN_RESERVE_S[stage]
    reserve_usd = (reserve_s / 3600) * RATE
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"

    statuses = []
    aborted = False
    for algo in algos:
        if aborted:
            break
        for mode in modes:
            log = _read_costlog()
            if log["usd"] + reserve_usd >= BUDGET_USD:
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
            _append_costlog(f"reserve:{stage}:{algo}:{mode}", reserve_usd)
            cmd = [
                sys.executable,
                "/repo/ssd_stale_exp/runner.py",
                "--algo", algo,
                "--mode", mode,
                "--stage", stage,
                "--target", TARGET,
                "--draft", DRAFTS[algo],
                "--target-revision", pins[TARGET],
                "--draft-revision", pins[DRAFTS[algo]],
                "--require-cuda",
                "--dtype", "fp32" if stage == "s0" else "bf16",
                "--out-dir", str(out_dir),
            ]
            print(f"[run] {' '.join(cmd)}", flush=True)
            t0 = time.monotonic()
            try:
                proc = subprocess.run(cmd, env=env, timeout=reserve_s)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = -9
            dt = time.monotonic() - t0
            actual_usd = (dt / 3600) * RATE
            _append_costlog(
                f"adjust:{stage}:{algo}:{mode}", round(actual_usd - reserve_usd, 4)
            )
            status = {0: "OK", -9: "TIMEOUT"}.get(code, f"EXIT_{code}")
            statuses.append(
                {"algo": algo, "mode": mode, "status": status, "seconds": round(dt, 1)}
            )
            results_vol.commit()
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
        "ok": ok,
        "statuses": statuses,
        "total_usd": log["usd"],
        "budget_usd": BUDGET_USD,
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
    for stage in ("s0", "s1"):
        p = Path(f"/results/{stage}/stage_report.json")
        if p.exists():
            out["stages"][stage] = json.loads(p.read_text())
    print(json.dumps(out)[:30000], flush=True)
    return out


@app.local_entrypoint()
def main(action: str = "s0", algos: str = "dflash,dspark", modes: str = ""):
    algo_list = [a for a in algos.split(",") if a]
    mode_list = [m for m in modes.split(",") if m] or list(MODES)
    if action == "download":
        download_models.remote()
    elif action in ("s0", "s1", "s1t0"):
        rep = run_stage.remote(action, algo_list, mode_list)
        print(json.dumps(rep.get("statuses", []), indent=2))
        print(f"total spent: ${rep.get('total_usd')} / budget ${rep.get('budget_usd')}")
        if not rep.get("ok"):
            raise SystemExit(1)
    elif action == "report":
        report.remote()
    else:
        raise SystemExit(f"unknown action {action}")
