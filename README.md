# Speculative Decoding Benchmark

How much does **stale target-hidden-state conditioning** — the situation an
SSD-style ([Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251))
overlap system puts its draft model in — degrade the accepted length of
**DFlash** and **DSpark** block drafts?

Measured on [DeepSpec](https://github.com/sgl-project/DeepSpec) released
checkpoints, target `Qwen/Qwen3-4B`, chain drafting, rejection sampling,
temperature 1.0, 96 prompts (GSM8K / HumanEval / MT-Bench x 32), 256 new
tokens, bf16, 1x A10 on Modal. Total compute cost of everything in this repo:
**$0.78**.

## The three arms

Same prompts, same per-sample seeds; the only difference is *when* the
target's hidden states (the "context features" both drafts condition on)
reach the draft:

| arm | meaning |
|---|---|
| `fresh` | control — stock DeepSpec behavior: features of newly committed tokens are appended to the draft context immediately after each verify |
| `gap` | pessimistic stale (T1) — features arrive **one round late**; while drafting round *r+1* the tokens committed in round *r* have **no rows at all** in the draft context (a moving hole of ~τ tokens) |
| `self_kv` | optimistic/leaky stale (T2) — the hole is filled with the draft's **own proposal K/V rows** for exactly one round, then surgically replaced by the late fresh features |

A real SSD hit-path sits between T1 and T2. Empirically the two arms nearly
coincide, so the bracket is tight.

## Result

![tau](figs/tau_paired.png)

| algo | fresh τ | gap τ | self_kv τ | stale/fresh | Δτ (fresh−gap, 95% CI) |
|---|---|---|---|---|---|
| DFlash | 4.35 | 2.55 | 2.59 | **0.59** | +2.04 [+1.89, +2.20] |
| DSpark | 4.98 | 2.94 | 3.07 | **0.59–0.62** | +2.21 [+2.07, +2.36] |

Per domain (τ_stale/τ_fresh): math 57–60%, code 54–58%, chat 61–66% — the
strongest domains lose the most.

**Break-even analysis.** With p_hit = 1, SSD beats synchronous SD iff
τ_stale/τ_fresh > t_verify/(t_verify+t_draft). Measured on A10:
t_draft ≈ 7–9 ms, t_verify ≈ 36 ms → threshold ≈ **0.80–0.83**, versus
measured **0.59**. For these one-forward block drafts the acceptance loss
(−41%) is roughly twice the drafting latency SSD could hide (~20% of the
cycle), so **zero-shot stale conditioning does not pay** in this setting —
even with a perfect pre-speculation cache.

Implications:

1. To make SSD work with feature-conditioned block drafts, **retrain the
   draft with lagged features** (the DeepSpec training pipeline can produce
   exactly this) or use a feature-free draft (the SSD paper's own choice).
2. `self_kv` buys almost nothing over `gap` (+0.04 / +0.13 τ): the draft's
   own mask-conditioned K/V is a poor substitute for target features.
3. Degradation grows monotonically with hole size (`figs/tau_vs_staleness.png`),
   ~2.7 τ at 1 missing row down to ~1.x at 4+.

More figures: `figs/alpha_k.png` (per-position acceptance),
`figs/pmf.png` (committed-length distribution — the fan-out planning input),
`figs/rank_cdf.png` (correction-token top-F coverage — the p_hit driver).

## Correctness gates (why you can trust the numbers)

1. CPU smoke test with tiny random models: 18 combos, all three arms produce
   **token-identical output to target-only greedy** at temperature 0
   (rejection sampling stays exact under any conditioning).
2. S0 on GPU (fp32): 6/6 runs token-identical to target-only greedy with the
   real 4B checkpoints. (In bf16 strict equality fails for a benign reason:
   batched-verify vs serial-greedy reduction order flips near-tie argmaxes.)
3. Every run pins HF revisions, runs offline on GPU, and logs
   seeds/manifest hash/versions in its summary (see `results/*/*.summary.json`).
4. Reviewed by an independent model (Codex, max reasoning) before any GPU
   spend; its P0/P1 findings (budget pre-charge accounting, artifact
   isolation, CUDA fail-fast, revision pinning) are incorporated.

## Reproduce

This code imports the `deepspec` package; clone it *inside* a DeepSpec
checkout:

```bash
git clone https://github.com/sgl-project/DeepSpec
cd DeepSpec && pip install -r requirements.txt
git clone https://github.com/Future-Outlier/speculative-decoding-benchmark ssd_stale_exp

python ssd_stale_exp/smoke_test.py                       # CPU gate
modal run ssd_stale_exp/modal_app.py --action download   # pin + cache models
modal run ssd_stale_exp/modal_app.py --action s0         # fp32 correctness
modal run ssd_stale_exp/modal_app.py --action s1         # the measurement
python ssd_stale_exp/plots.py --results <s1 dir> --out figs
```

Budget guard: cumulative cost is pre-charged per run into the results
volume's `costlog.json`; stages refuse to start work past
`STALE_BUDGET_USD` (default $8).

## Caveats

1. Zero-shot staleness: the drafts were trained with fresh features; a
   stale-trained draft is the obvious next experiment, not covered here.
2. bsz=1, HuggingFace SDPA stack; t_draft/t_verify ratios differ on
   optimized serving stacks and larger targets — recompute the break-even
   with your own timings.
3. At temperature 1.0 the fresh/stale arms follow different trajectories
   after the first divergence; per-round analytic acceptance
   (`E[L] = 1 + Σ Π min(1, p/q)`) matches the sampled τ to <0.02, so
   sampling noise is not driving the gap.
4. MT-Bench uses first turns only (DeepSpec eval convention).

## References

- DSpark: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147) ·
  DFlash: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) ·
  EAGLE-3: [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) ·
  SSD: [arXiv:2603.03251](https://arxiv.org/abs/2603.03251)
- Code baseline: [sgl-project/DeepSpec](https://github.com/sgl-project/DeepSpec) (MIT)
- Checkpoints: `deepseek-ai/{dflash,dspark}_qwen3_4b_block7`, pinned
  revisions in `results/*/stage_report.json`
