"""Statistics shared by the benchmark runner and plotting script.

The primary Stage-A estimand is bonus-inclusive accepted length per target
verification.  Output tokens within the finite generation limit are retained
as a secondary harness-work metric.  Prompts are the independent sampling
units, so uncertainty is estimated by paired cluster bootstrap while the point
estimate remains a ratio of pooled token and round counts.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping


def committed_within_limit(
    row: Mapping[str, object],
    sample: Mapping[str, object],
    max_new_tokens: int,
) -> int:
    """Return tokens from a verification that survive max-token clipping.

    ``start_pos`` is the already-generated token that anchors the current
    verification, so only positions strictly after it can be newly committed.
    """

    limit = int(sample["n_input_tokens"]) + int(max_new_tokens)
    remaining = max(0, limit - (int(row["start_pos"]) + 1))
    return min(int(row["committed"]), remaining)


def add_effective_committed(
    rows: Iterable[Mapping[str, object]],
    samples: Mapping[int, Mapping[str, object]],
    max_new_tokens: int,
) -> list[dict]:
    """Copy round rows and add the output-aligned committed-token count."""

    enriched = []
    for source in rows:
        row = dict(source)
        uid = int(row["uid"])
        effective = committed_within_limit(row, samples[uid], max_new_tokens)
        row["committed_within_limit"] = effective
        row["boundary_excess_tokens"] = int(row["committed"]) - effective
        enriched.append(row)
    return enriched


def pooled_value_per_verification(
    rows: Iterable[Mapping[str, object]],
    value_key: str = "committed",
) -> float:
    rows = list(rows)
    if not rows:
        raise ValueError("pooled_value_per_verification requires at least one round")
    return sum(float(row[value_key]) for row in rows) / len(rows)


def cluster_totals(
    rows: Iterable[Mapping[str, object]],
    value_key: str = "committed",
) -> dict[int, tuple[float, int]]:
    totals: dict[int, list[float | int]] = defaultdict(lambda: [0.0, 0])
    for row in rows:
        uid = int(row["uid"])
        totals[uid][0] += float(row[value_key])
        totals[uid][1] += 1
    return {uid: (float(total), int(count)) for uid, (total, count) in totals.items()}


def _ratio_of_sums(
    totals: Mapping[int, tuple[float, int]], sampled_uids: Iterable[int]
) -> float:
    numerator = 0.0
    denominator = 0
    for uid in sampled_uids:
        total, count = totals[uid]
        numerator += total
        denominator += count
    if denominator == 0:
        raise ValueError("bootstrap sample contains no rounds")
    return numerator / denominator


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_cluster_bootstrap_diff(
    rows_a: Iterable[Mapping[str, object]],
    rows_b: Iterable[Mapping[str, object]],
    *,
    value_key: str = "committed",
    iterations: int = 4000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Estimate a pooled value difference with prompts as paired clusters."""

    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    totals_a = cluster_totals(rows_a, value_key)
    totals_b = cluster_totals(rows_b, value_key)
    if set(totals_a) != set(totals_b):
        raise ValueError("paired arms must contain the same prompt uids")
    uids = sorted(totals_a)
    if not uids:
        raise ValueError("paired bootstrap requires at least one prompt")

    point = _ratio_of_sums(totals_a, uids) - _ratio_of_sums(totals_b, uids)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [uids[rng.randrange(len(uids))] for _ in uids]
        draws.append(
            _ratio_of_sums(totals_a, sampled)
            - _ratio_of_sums(totals_b, sampled)
        )
    draws.sort()
    return point, _percentile(draws, 0.025), _percentile(draws, 0.975)


def paired_cluster_bootstrap_ratio(
    reference_rows: Iterable[Mapping[str, object]],
    treatment_rows: Iterable[Mapping[str, object]],
    *,
    value_key: str = "committed",
    iterations: int = 4000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Estimate treatment/reference quality retention with paired prompts."""

    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    reference = cluster_totals(reference_rows, value_key)
    treatment = cluster_totals(treatment_rows, value_key)
    if set(reference) != set(treatment):
        raise ValueError("paired arms must contain the same prompt uids")
    uids = sorted(reference)
    if not uids:
        raise ValueError("paired bootstrap requires at least one prompt")

    def ratio(sampled_uids: Iterable[int]) -> float:
        sampled_uids = list(sampled_uids)
        denominator = _ratio_of_sums(reference, sampled_uids)
        if denominator == 0.0:
            raise ValueError("reference pooled value must be positive")
        return _ratio_of_sums(treatment, sampled_uids) / denominator

    point = ratio(uids)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [uids[rng.randrange(len(uids))] for _ in uids]
        draws.append(ratio(sampled))
    draws.sort()
    return point, _percentile(draws, 0.025), _percentile(draws, 0.975)


def validate_paired_samples(
    samples_a: Mapping[int, Mapping[str, object]],
    samples_b: Mapping[int, Mapping[str, object]],
) -> None:
    """Reject accidental uid pairing across different prompts or seeds."""

    if set(samples_a) != set(samples_b):
        raise ValueError("paired arms must contain the same sample uids")
    identity_fields = ("dataset", "row_idx", "prompt_sha1", "seed")
    for uid in sorted(samples_a):
        for field in identity_fields:
            if samples_a[uid].get(field) != samples_b[uid].get(field):
                raise ValueError(
                    f"paired sample mismatch at uid={uid}, field={field}: "
                    f"{samples_a[uid].get(field)!r} != {samples_b[uid].get(field)!r}"
                )
