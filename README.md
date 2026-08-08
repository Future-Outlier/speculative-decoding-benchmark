# Perfect-coverage retained-mask-KV draft-quality surrogate

## Scope and conclusion

This repository is an early accuracy gate for integrating DFlash/DSpark with
Speculative Speculative Decoding (SSD).  It sequentially replays the one outcome
that was actually realized under perfect coverage and measures draft acceptance
for one explicit retained-mask-KV surrogate when target features are one round
late.  It intentionally does not claim that this surrogate is the eventual SSD
adapter, and does not implement SSD scheduling or overlap yet.

Here, "accuracy" means **draft acceptance quality**, not final output accuracy.
Rejection sampling makes final output follow the target distribution regardless
of draft quality.  The primary metric is bonus-inclusive accepted length:

```text
tau = accepted draft tokens + one correction/bonus token
quality retention = tau_retained_mask_kv / tau_fresh
```

The checked-in artifacts support one narrow result: on these checkpoints, both
one-round-late treatments substantially reduce accepted length.  They do not
establish that:

- either treatment is a lower or upper bound on a real SSD hit path;
- a finite-fan-out SSD cache has the same conditional accepted length;
- SSD throughput is monotone in cache hit rate or is optimized at `p_hit=0`;
- a correction-token rank CDF equals SSD cache-hit probability.

Those stronger claims appeared in an earlier revision and were incorrect.

## Source basis

The method was audited against the directly governing primary sources:

- [SSD v3](https://arxiv.org/abs/2603.03251) and
  [official code](https://github.com/tanishqkumar/ssd/tree/d7eb8fa0edb77a6d0876af1903367b9bb82f54e7)
- [DFlash v2](https://arxiv.org/abs/2602.06036) and
  [official code](https://github.com/z-lab/dflash/tree/94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756)
- [DSpark v1](https://arxiv.org/abs/2607.05147)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192) and
  [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [EAGLE-3](https://arxiv.org/abs/2503.01840), because SSD Appendix E discusses
  EAGLE activation availability
- [DeepSpec](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2)
  at `005e03b81cec38b7da6399833d609ee89a2587f2`

The checked-in model pins are:

| model | Hugging Face revision |
|---|---|
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `deepseek-ai/dflash_qwen3_4b_block7` | `02d530b7962ea1412beaf41a05c0b8e36d5f9b1d` |
| `deepseek-ai/dspark_qwen3_4b_block7` | `3457dff1417cb84927f6098a5fcb7cee85c934b7` |

The historical Modal image excluded `.git`, so its benchmark/DeepSpec code SHA
was recorded as `unknown`.  These artifacts therefore cannot prove the exact
code revision that generated them.  New runs record both Git SHAs, dirty-state
flags, imported source paths, and a benchmark-source digest.

## Treatments

All arms use the same prompts and per-sample seeds.  DFlash is the DeepSpec
checkpoint with `markov_rank=0`; DSpark uses its Markov head.  The DSpark
confidence scheduler is disabled (`confidence_threshold=0`), matching the
paper's fixed-block raw-draft-quality evaluation rather than scheduled serving.
This is a DeepSpec DFlash-derived checkpoint/path, not the original DFlash
reference backend; its block geometry and temperature-1 verifier are not an
official-backend replication.  The DSpark result covers its default Markov
checkpoint, not the paper's RNN variant.

| arm | intervention |
|---|---|
| `fresh` | Stock DeepSpec: newly committed target features are available to the next draft round. |
| `gap` | Omission ablation: the previous round's target-feature rows are absent for one draft round. |
| `self_kv` | Candidate perfect-coverage surrogate: previous masked-block proposal K/V rows are retained for the realized outcome, then replaced when target features arrive. |

Sequential realized-outcome replay avoids materializing unused branches and is
equivalent to selecting a precomputed perfect-coverage branch only when branch
state and randomness are isolated.  Calling `self_kv` the selected branch
additionally assumes that the intended adapter uses these retained
mask-conditioned K/V rows as its unavailable-feature surrogate.  That adapter
contract has not yet been validated against an all-branches implementation.
`gap` and `self_kv` are not a bracket.  DFlash and
DSpark inject target features as per-layer K/V, whereas SSD Appendix E discusses
EAGLE draft activations.  No paper theorem, code invariant, or experiment
establishes an ordering between an omitted row, this retained-KV surrogate, and
a future integrated adapter.

## Draft-quality result

Configuration: `Qwen3-4B`, block size 7, chain drafting, batch size 1,
temperature 1, 96 prompts (GSM8K/HumanEval/MT-Bench, 32 each), maximum 256 new
tokens, bf16, A10 according to the S1 per-run summaries.

The primary metric is the paper-aligned, unclipped accepted outcome produced by
each verification.  Terminal-round continuations are retained because this is
a draft-quality simulation: they are valid target-verified continuation
outcomes even when the finite harness does not return them to the caller.
Uncertainty uses prompts as paired clusters and recomputes the displayed pooled
ratio-of-sums in every bootstrap draw.

| algorithm | fresh tau | self_kv tau | self/fresh, paired-cluster 95% CI | gap tau (diagnostic) |
|---|---:|---:|---:|---:|
| DFlash | 4.352 | 2.591 | 0.595 [0.578, 0.612] | 2.553 |
| DSpark | 4.975 | 3.073 | 0.618 [0.600, 0.634] | 2.937 |

Under the perfect-coverage retained-mask-KV assumptions, `self_kv` is the
candidate quality estimate.  It retains only 59.5% (DFlash) and 61.8% (DSpark)
of fresh accepted length.  Using the old single-GPU component timings, an idealized
`p_hit=1` gate would require roughly 83% and 80%, respectively.  Therefore the
released fresh-trained checkpoints **fail this early quality gate for this
specific retained-KV surrogate**; this does not rule out a better surrogate or
lag-aware retraining.

For finite-request harness accounting, returned tokens per verification are
also recorded separately: DFlash `4.299 / 2.532 / 2.565`, DSpark
`4.915 / 2.911 / 3.045`.  They are not the primary draft-accuracy metric.
Excluding every boundary-affected round gives nearly identical `self/fresh`
retention (`0.594` DFlash, `0.618` DSpark), so the conclusion is not a terminal-
round artifact.

Temperature 0 gives similar ratios at the two temperatures that were tested:

| algorithm | fresh tau | gap tau | self_kv tau | gap/fresh | self/fresh |
|---|---:|---:|---:|---:|---:|
| DFlash | 4.639 | 2.708 | 2.785 | 0.584 | 0.600 |
| DSpark | 5.108 | 3.013 | 3.168 | 0.590 | 0.620 |

The previous temperature-0 recovery-rank figure was invalid: at temperature 0,
`draft_probs` is one-hot, so every rejected non-argmax token ties at the
computed rank 2.  Those figures have been removed.  A valid SSD metric needs
raw pre-temperature logits, exclusion of the already sampled token, rejection
depth, full-accept bonus outcomes, and an actual cache membership test.

The missing-feature-row plot is also observational.  Its x-axis is determined
by the previous round's accepted length, so it is not a randomized causal dose;
the corrected plot labels it as an association and makes no monotonicity claim.

## What the SSD equation requires

SSD Theorem 7 is

```text
R_SSD = [p_hit E_hit + (1 - p_hit) E_miss]
        / [p_hit max(1, T_primary) + (1 - p_hit)(1 + T_backup)]
```

For this perfect-coverage simulation, `p_hit=1`, so averaging every realized
outcome is a coherent quality estimate under the stated surrogate contract.  The
`p_hit=1` break-even algebra in `breakeven_plot.py` is valid for that special
case.  It becomes invalid to hold this unconditional mean fixed and then claim
throughput is monotone for arbitrary finite-fan-out `p_hit`: a real cache selects
outcomes, so hit status and next-round draft quality can be correlated.

The existing timing is likewise not SSD timing.  It is single-GPU Hugging Face
SDPA component timing and omits parts of the local callback lifecycle as well
as SSD branch generation, cache lookup, communication, glue/extend, fallback,
and real target/draft overlap.  New summaries separate propose, verify, and
update observations, but deliberately do not call their sum a full cycle.

## Staged validation path

This repository implements Stage A only.  The remaining stages are:

1. **Stage A — perfect-coverage surrogate quality:** fresh vs the specified stale surrogate,
   accepted length, prefix survival, and exactness gates;
2. **Stage B — finite-fan-out accuracy:** a cache keyed by realized
   `(accepted_length, recovery_token)` outcomes, per-depth raw-logit fan-out
   with the sampled token excluded, plus the
   full-accept bonus outcome;
3. **Stage C — integrated performance:** branch-isolated KV lifecycle,
   primary/backup, glue/extend, rollback, communication, and actual overlap.

Across all stages the implementation must retain:

1. an explicit DFlash/DSpark stale-feature surrogate contract and branch-
   isolated KV lifecycle;
2. `p_hit`, `E[tau | hit]`, and `E[tau | miss]` joined to the next round once
   Stage B exists;
3. for DSpark serving, confidence calibration under the stale treatment and
   its multi-request, load-aware scheduler rather than fixed `K` only.

Stage A is enough to answer the user's current question: whether stale-path
draft quality is promising enough to justify Stage B/C implementation.

## Reproduce the proxy analysis

Clone this repository under any directory name inside DeepSpec:

```bash
git clone https://github.com/deepseek-ai/DeepSpec
cd DeepSpec
git checkout 005e03b81cec38b7da6399833d609ee89a2587f2
pip install -r requirements.txt
pip install modal==1.5.3
git clone https://github.com/Future-Outlier/speculative-decoding-benchmark

python speculative-decoding-benchmark/test_benchmark_stats.py
python speculative-decoding-benchmark/smoke_test.py
python speculative-decoding-benchmark/plots.py \
  --results speculative-decoding-benchmark/results/s1 \
  --out /tmp/ssd-stale-proxy-figs
```

The runner asserts that the imported controller comes from this clone, avoiding
the previous hard-coded `ssd_stale_exp` path that could silently execute a
sibling copy.

For Modal, the code refuses `STALE_BUDGET_USD > 3`, conservatively accounts for
configured GPU/CPU/memory with a regional safety factor, includes model download
in its ledger, refreshes reused Volumes before guard reads, and writes immutable
run-specific output directories.  The ledger assumes a single experiment
orchestrator; Modal Volumes do not provide an atomic cross-function lock.  This
is still a resource-cost estimate, not the Modal invoice.  A Modal workspace or
environment budget is the authoritative hard billing backstop.

```bash
STALE_BUDGET_USD=3 modal run \
  speculative-decoding-benchmark/modal_app.py \
  --action download

STALE_BUDGET_USD=3 modal run \
  speculative-decoding-benchmark/modal_app.py \
  --action s0 --run-id proxy-s0
```

The checked-in ledger reached an estimated cumulative `$1.4395` after S0, S1,
and S1-temperature-0.  The old `$0.78 for everything` statement referred only
to an earlier partial ledger and was not an actual billing measurement.

## Correctness boundary

Conditioning changes proposal distribution `q`, but lossless speculative
sampling remains valid when verification uses that same `q`, acceptance
`min(1, p/q)`, and residual correction proportional to `(p-q)^+`.  DeepSpec
does so.  This is distribution-exact in the mathematical algorithm, subject to
floating-point and explicit numerical guards (`clamp_min(1e-8)` and residual
fallback).  The current S0 gate checks greedy token equality; it is not a
temperature-1 distributional proof.

Historical artifact caveats:

- S0's stage report says A10 while its individual summaries say L40S.
- Historical code SHA is unknown, although model revisions and software
  versions were recorded.
- MT-Bench uses first turns only, following the DeepSpec evaluation convention.
