"""CPU smoke test with tiny random models — run BEFORE spending GPU money.

Checks, for each mode in {fresh, gap, self_kv} x {markov on/off}:
  1. temperature 0: committed output identical to target-only greedy
     (lossless property must hold under any treatment)
  2. controller invariants hold (asserted inside stale_core)
  3. gap logs missing_fresh_rows > 0 from round 2; self_kv hole_rows == 0
  4. temperature 1 runs end-to-end without error

Run:  python ssd_stale_exp/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402

from deepspec.eval.base_evaluator import resolve_stop_token_ids  # noqa: E402
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # noqa: E402
from deepspec.utils import seed_all  # noqa: E402

from ssd_stale_exp.runner import run_one_sample, target_greedy  # noqa: E402

VOCAB = 257
HID = 64
BLOCK = 4


def tiny_target() -> Qwen3ForCausalLM:
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HID,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg)
    return model.eval()


def tiny_draft(markov: bool, confidence: bool = False) -> Qwen3DSparkModel:
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HID,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
    )
    cfg.target_layer_ids = [0, 2]
    cfg.block_size = BLOCK
    cfg.mask_token_id = VOCAB - 1
    cfg.num_anchors = 8
    cfg.markov_rank = 8 if markov else 0
    if markov:
        cfg.markov_head_type = "vanilla"
    cfg.enable_confidence_head = confidence
    if confidence:
        assert markov
        cfg.confidence_head_with_markov = True
    model = Qwen3DSparkModel(cfg)
    return model.eval()


def main():
    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    seed_all(0)
    target = tiny_target()

    class _Tok:
        eos_token_id = 0

    stop_token_ids = resolve_stop_token_ids(target, _Tok())
    prompts = [
        torch.randint(1, VOCAB - 2, (1, n)) for n in (11, 23, 5)
    ]

    failures = 0
    for markov, confidence in ((False, False), (True, False), (True, True)):
        seed_all(1)
        draft = tiny_draft(markov, confidence)
        for temperature, max_new in ((0.0, 24), (1.0, 24)):
            refs = []
            if temperature == 0.0:
                for p in prompts:
                    refs.append(target_greedy(target, p, max_new, stop_token_ids))
            for mode in ("fresh", "gap", "self_kv"):
                for i, p in enumerate(prompts):
                    seed_all(100 + i)
                    result, ctrl = run_one_sample(
                        target=target,
                        draft=draft,
                        input_ids=p,
                        mode=mode,
                        temperature=temperature,
                        max_new_tokens=max_new,
                        confidence_threshold=0.0,
                        stop_token_ids=stop_token_ids,
                        device=device,
                    )
                    if temperature == 0.0:
                        ok = torch.equal(refs[i], result.output_ids)
                        if not ok:
                            failures += 1
                            print(
                                f"FAIL greedy-mismatch markov={markov} mode={mode} "
                                f"prompt#{i}: ref={refs[i][0, p.shape[1]:].tolist()} "
                                f"got={result.output_ids[0, p.shape[1]:].tolist()}"
                            )
                    rounds = ctrl.rounds
                    if mode == "gap" and len(rounds) > 1:
                        if not any(r.missing_fresh_rows > 0 for r in rounds[1:]):
                            failures += 1
                            print(f"FAIL gap has no missing rows markov={markov}")
                    if mode == "self_kv":
                        # hole can be exactly 1 right after a full-accept
                        # round (see stale_core.update), never more.
                        if any(r.hole_rows not in (0, 1) for r in rounds):
                            failures += 1
                            print(f"FAIL self_kv hole>1 markov={markov}")
                    if mode == "fresh":
                        if any(r.missing_fresh_rows != 0 for r in rounds):
                            failures += 1
                            print(f"FAIL fresh missing!=0 markov={markov}")
                print(
                    f"ok markov={int(markov)} temp={temperature} mode={mode:8s} "
                    f"rounds={len(ctrl.rounds)} "
                    f"tau={sum(r.committed for r in ctrl.rounds) / max(1, len(ctrl.rounds)):.2f}"
                )

    if failures:
        print(f"SMOKE FAILED: {failures} failures")
        sys.exit(1)
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
