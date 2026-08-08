"""One-round-late feature treatments for DSpark/DFlash block drafts.

Modes
-----
fresh    stock behavior: after each verify, the newly committed tokens'
         target features are appended to the draft context immediately.
         Delegates the block forward to the repo's forward_dspark_draft_block
         so it is bit-identical to the stock eval path.
gap      omission proxy: feature chunks arrive one round late.
         While drafting round r+1 the context has NO rows for the tokens
         committed in round r (a positional hole of ell_r tokens); those
         rows are appended one round later.
self_kv  retained-mask-KV proxy: the hole is filled with the draft's
         own proposal K/V rows (mask-conditioned) for exactly one round,
         after which they are surgically removed and replaced by the
         late-arriving fresh feature rows.

Invariants asserted every round:
  cache.get_seq_length() == controller.rows
  (self_kv) at most one outstanding pseudo span; spans disjoint by position.

The draft_probs used for verification are always the probabilities of the
proposal sampled under the treatment.  This preserves the speculative-
sampling distribution in exact arithmetic; the implementation still has the
floating-point guards used by DeepSpec.  At temperature 0, every mode must
match target-only greedy output.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from deepspec.eval.base_evaluator import DraftProposal, VerificationResult
from deepspec.eval.dspark.draft_ops import (
    build_dspark_proposal,
    forward_dspark_draft_block,
)
from deepspec.modeling.dspark.common import extract_context_feature

MODES = ("fresh", "gap", "self_kv")


# ---------------------------------------------------------------------------
# DynamicCache internals (transformers 5.x: cache.layers[i].keys/.values;
# 4.x fallback: cache.key_cache[i]/value_cache[i]).
# ---------------------------------------------------------------------------

def _num_cache_layers(cache: DynamicCache) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    raise RuntimeError("Unsupported DynamicCache internals")


def _get_kv(cache: DynamicCache, idx: int):
    if hasattr(cache, "layers"):
        layer = cache.layers[idx]
        return layer.keys, layer.values
    return cache.key_cache[idx], cache.value_cache[idx]


def _set_kv(cache: DynamicCache, idx: int, k: torch.Tensor, v: torch.Tensor):
    if hasattr(cache, "layers"):
        cache.layers[idx].keys = k
        cache.layers[idx].values = v
    else:
        cache.key_cache[idx] = k
        cache.value_cache[idx] = v


def remove_cache_rows(cache: DynamicCache, offset: int, n: int) -> None:
    """Surgically remove rows [offset, offset+n) from every layer."""
    if n <= 0:
        return
    for i in range(_num_cache_layers(cache)):
        k, v = _get_kv(cache, i)
        assert k.shape[-2] >= offset + n, (
            f"remove_cache_rows: layer {i} has {k.shape[-2]} rows, "
            f"need [{offset}, {offset + n})"
        )
        _set_kv(
            cache,
            i,
            torch.cat([k[:, :, :offset], k[:, :, offset + n:]], dim=-2),
            torch.cat([v[:, :, :offset], v[:, :, offset + n:]], dim=-2),
        )
    # Keep any length counters coherent with the tensors.
    new_len = None
    for attr in ("_seen_tokens", "seen_tokens"):
        if hasattr(cache, attr):
            if new_len is None:
                k0, _ = _get_kv(cache, 0)
                new_len = k0.shape[-2]
            setattr(cache, attr, new_len)
    if hasattr(cache, "layers"):
        for i in range(_num_cache_layers(cache)):
            layer = cache.layers[i]
            if hasattr(layer, "cumulative_length"):
                k, _ = _get_kv(cache, i)
                layer.cumulative_length = k.shape[-2]


# ---------------------------------------------------------------------------
# Block forward with explicit (possibly non-contiguous) position ids.
# The attention applies RoPE per-row from position_ids (Q takes the trailing
# q_len entries: see deepspec/modeling/dspark/qwen3/modeling.py:37-38), so a
# positional hole between ctx rows and block rows is well-defined.
# ---------------------------------------------------------------------------

def forward_block_explicit(
    model,
    *,
    draft_input_ids: torch.Tensor,
    ctx_feats: torch.Tensor,          # [1, n_ctx, feat_dim] (n_ctx may be 0)
    ctx_positions: torch.Tensor,      # [1, n_ctx]
    block_positions: torch.Tensor,    # [1, K]
    past_key_values: DynamicCache,
    crop_to: int | None,
) -> torch.Tensor:
    position_ids = torch.cat([ctx_positions, block_positions], dim=1)
    block_hidden = model._forward_backbone(
        target_hidden_states=ctx_feats,
        noise_embedding=model.embed_tokens(draft_input_ids),
        position_ids=position_ids,
        attention_mask=None,
        past_key_values=past_key_values,
        use_cache=True,
        is_causal=False,
    )
    if crop_to is not None:
        past_key_values.crop(crop_to)
    return block_hidden


@dataclass
class RoundLog:
    round_idx: int
    start_pos: int
    proposal_len: int
    effective_proposal_len: int
    accepted_draft: int
    committed: int
    missing_fresh_rows: int      # positions with no fresh feature row at propose
    hole_rows: int               # no-row positions (gap: ==missing; self_kv: 0 or 1)
    rho: list[float]
    e_len_analytic: float | None
    recovery_kind: str           # "correction" | "bonus" | "eos"
    t_propose_ms: float
    t_round_ms: float
    t_update_ms: float = 0.0


class StaleController:
    """Owns the draft KV cache + feature-delivery schedule for one sample."""

    def __init__(
        self,
        model,
        mode: str,
        temperature: float,
        confidence_threshold: float,
        device: torch.device,
    ):
        assert mode in MODES, mode
        self.model = model
        self.mode = mode
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.K = int(model.block_size)

        self.cache = DynamicCache()
        self.rows = 0                    # rows currently in cache (bookkeeping)
        self.fresh_covered = 0           # positions [0, fresh_covered) have real feature rows
        self.ready: tuple[torch.Tensor, int] | None = None
        self.hold: tuple[torch.Tensor, int] | None = None
        self.pseudo: dict | None = None          # {"off": int, "n": int, "pos": int}
        self.pending_block: dict | None = None   # set at propose (self_kv only)
        self.round_idx = 0
        self.cur_start = -1
        self.last_committed = 0
        self._t_propose_start = 0.0
        self._t_propose_ms = 0.0
        self.rounds: list[RoundLog] = []
        self.feat_dim: int | None = None

    # -- generate_decoding_sample callbacks --------------------------------

    def init_prompt(self, initial_output, num_input_tokens: int) -> None:
        feats = extract_context_feature(
            initial_output.hidden_states,
            self.model.target_layer_ids,
        )
        assert feats.shape[1] == num_input_tokens
        self.feat_dim = int(feats.shape[-1])
        # Prompt features are never stale (SSD prefill happens before overlap).
        self.ready = (feats, 0)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def propose(
        self,
        *,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids=None,
    ) -> DraftProposal:
        del stop_token_ids
        self._sync()
        self._t_propose_start = time.perf_counter()
        K = self.K
        model = self.model
        self.cur_start = int(start)

        draft_input_ids = torch.full(
            (output_ids.size(0), K),
            int(model.mask_token_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]

        if self.mode == "fresh":
            feats, p0 = self.ready
            assert p0 == self.rows and p0 + feats.shape[1] == start, (
                f"fresh alignment broken: p0={p0} rows={self.rows} "
                f"n={feats.shape[1]} start={start}"
            )
            self.ready = None
            block_hidden = forward_dspark_draft_block(
                model,
                draft_input_ids=draft_input_ids,
                position_ids=position_ids,
                past_key_values_draft=self.cache,
                target_hidden_states=feats,
                start=start,
                block_size=K,
            )
            self.rows = start
            self.fresh_covered = start
        else:
            if self.ready is not None:
                feats, p0 = self.ready
                self.ready = None
                n = feats.shape[1]
                ctx_positions = torch.arange(
                    p0, p0 + n, device=self.device
                ).unsqueeze(0)
            else:
                assert self.feat_dim is not None
                feats = torch.zeros(
                    (1, 0, self.feat_dim),
                    dtype=next(model.parameters()).dtype,
                    device=self.device,
                )
                n = 0
                ctx_positions = torch.zeros(
                    (1, 0), dtype=torch.long, device=self.device
                )
            block_positions = torch.arange(
                start, start + K, device=self.device
            ).unsqueeze(0)
            rows_before = self.rows
            crop_to = rows_before + n if self.mode == "gap" else None
            block_hidden = forward_block_explicit(
                model,
                draft_input_ids=draft_input_ids,
                ctx_feats=feats,
                ctx_positions=ctx_positions,
                block_positions=block_positions,
                past_key_values=self.cache,
                crop_to=crop_to,
            )
            if n > 0:
                self.fresh_covered = p0 + n
            if self.mode == "gap":
                self.rows = rows_before + n
            else:  # self_kv keeps block rows until update()
                self.rows = rows_before + n + K
                self.pending_block = {"off": rows_before + n, "pos": start}

        assert self.cache.get_seq_length() == self.rows, (
            f"cache rows {self.cache.get_seq_length()} != bookkeeping {self.rows}"
        )

        proposal = build_dspark_proposal(
            model=model,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=K,
            temperature=self.temperature,
            confidence_threshold=self.confidence_threshold,
        )
        self._sync()
        self._t_propose_ms = (time.perf_counter() - self._t_propose_start) * 1e3
        return proposal

    def update(self, verification: VerificationResult) -> None:
        self._sync()
        t_update_start = time.perf_counter()
        a = int(verification.accepted_draft_tokens)
        ell = a + 1
        feats = extract_context_feature(
            verification.target_output.hidden_states,
            self.model.target_layer_ids,
        )[:, :ell, :]
        chunk = (feats, self.cur_start)

        if self.mode == "fresh":
            self.ready = chunk
        else:
            assert self.ready is None
            self.ready, self.hold = self.hold, chunk

        if self.mode == "self_kv":
            pb = self.pending_block
            assert pb is not None
            # Block rows exist for positions cur_start..cur_start+K-1 only.
            # On a full accept (a == K) the last committed position has no
            # self-KV row, leaving a one-row hole until the fresh chunk lands.
            keep = min(a + 1, self.K)
            self.cache.crop(pb["off"] + keep)
            self.rows = pb["off"] + keep
            new_pseudo = {"off": pb["off"], "n": keep, "pos": pb["pos"]}
            if self.pseudo is not None:
                remove_cache_rows(self.cache, self.pseudo["off"], self.pseudo["n"])
                self.rows -= self.pseudo["n"]
                if new_pseudo["off"] > self.pseudo["off"]:
                    new_pseudo["off"] -= self.pseudo["n"]
            self.pseudo = new_pseudo
            self.pending_block = None
            assert self.cache.get_seq_length() == self.rows

        self.last_committed = ell
        self._sync()
        # Keep cache surgery/feature-update time separate from propose+verify.
        # The interval between callbacks and post-verify diagnostics is not
        # captured, so these component timers must not be called a full cycle.
        if self.rounds:
            self.rounds[-1].t_update_ms = round(
                (time.perf_counter() - t_update_start) * 1e3, 3
            )

    def post_verify(
        self,
        proposal: DraftProposal,
        verification: VerificationResult,
    ) -> None:
        self._sync()
        t_round_ms = (time.perf_counter() - self._t_propose_start) * 1e3
        a = int(verification.accepted_draft_tokens)
        m = int(proposal.draft_token_count)
        eff = int(verification.effective_proposal_length)

        rho_list: list[float] = []
        e_len = None
        if m > 0 and proposal.draft_probs is not None:
            proposed = proposal.verify_input_ids[:, 1:]
            tgt = torch.gather(
                verification.target_probs[:, :-1, :],
                -1,
                proposed.unsqueeze(-1),
            ).squeeze(-1)
            drf = torch.gather(
                proposal.draft_probs,
                -1,
                proposed.unsqueeze(-1),
            ).squeeze(-1).clamp_min(1e-8)
            rho = torch.clamp(tgt / drf, max=1.0)[0, :eff]
            rho_list = [round(float(x), 6) for x in rho.tolist()]
            if not verification.terminated_by_stop_token:
                e_len = float(1.0 + rho.cumprod(0).sum().item())

        if verification.terminated_by_stop_token:
            kind = "eos"
        elif a < m:
            kind = "correction"
        else:
            kind = "bonus"

        if self.mode == "fresh":
            missing = 0
        else:
            missing = self.cur_start - self.fresh_covered
        if self.mode == "self_kv":
            covered = self.fresh_covered
            if self.pseudo is not None:
                covered = max(covered, self.pseudo["pos"] + self.pseudo["n"])
            hole = self.cur_start - covered
        else:
            hole = missing

        self.rounds.append(
            RoundLog(
                round_idx=self.round_idx,
                start_pos=self.cur_start,
                proposal_len=m,
                effective_proposal_len=eff,
                accepted_draft=a,
                committed=a + (0 if verification.terminated_by_stop_token else 1),
                missing_fresh_rows=int(missing),
                hole_rows=int(hole),
                rho=rho_list,
                e_len_analytic=e_len,
                recovery_kind=kind,
                t_propose_ms=round(self._t_propose_ms, 3),
                t_round_ms=round(t_round_ms, 3),
            )
        )
        self.round_idx += 1
