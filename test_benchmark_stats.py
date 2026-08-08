from __future__ import annotations

import unittest

from benchmark_stats import (
    add_effective_committed,
    paired_cluster_bootstrap_diff,
    paired_cluster_bootstrap_ratio,
    pooled_value_per_verification,
    validate_paired_samples,
)


class BenchmarkStatsTest(unittest.TestCase):
    def test_effective_commits_match_returned_outputs_after_initial_token(self):
        samples = {
            0: {"n_input_tokens": 10, "n_output_tokens": 5},
            1: {"n_input_tokens": 20, "n_output_tokens": 3},
        }
        rows = [
            {"uid": 0, "start_pos": 10, "committed": 3},
            {"uid": 0, "start_pos": 13, "committed": 4},
            {"uid": 1, "start_pos": 20, "committed": 2},
            {"uid": 1, "start_pos": 24, "committed": 4},
        ]

        enriched = add_effective_committed(rows, samples, max_new_tokens=5)

        self.assertEqual(
            sum(row["committed_within_limit"] for row in enriched),
            sum(sample["n_output_tokens"] - 1 for sample in samples.values()),
        )
        self.assertEqual([3, 1, 2, 0], [r["committed_within_limit"] for r in enriched])

    def test_cluster_bootstrap_point_matches_displayed_pooled_estimand(self):
        # Equal-prompt averaging would report 5.0.  The pooled round-level
        # difference displayed by the benchmark is 10 / 11 instead.
        arm_a = [{"uid": 0, "x": 10.0}] + [
            {"uid": 1, "x": 0.0} for _ in range(10)
        ]
        arm_b = [{"uid": 0, "x": 0.0}] + [
            {"uid": 1, "x": 0.0} for _ in range(10)
        ]

        point, low, high = paired_cluster_bootstrap_diff(
            arm_a, arm_b, value_key="x", iterations=200
        )

        self.assertAlmostEqual(
            point,
            pooled_value_per_verification(arm_a, "x")
            - pooled_value_per_verification(arm_b, "x"),
        )
        self.assertAlmostEqual(point, 10.0 / 11.0)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_paired_bootstrap_rejects_silent_uid_drops(self):
        with self.assertRaisesRegex(ValueError, "same prompt uids"):
            paired_cluster_bootstrap_diff(
                [{"uid": 0, "x": 1}],
                [{"uid": 1, "x": 1}],
                value_key="x",
                iterations=10,
            )

    def test_pairing_rejects_same_uid_with_different_prompt(self):
        common = {"dataset": "gsm8k", "row_idx": 3, "seed": 7}
        with self.assertRaisesRegex(ValueError, "prompt_sha1"):
            validate_paired_samples(
                {0: {**common, "prompt_sha1": "aaa"}},
                {0: {**common, "prompt_sha1": "bbb"}},
            )

    def test_quality_retention_uses_pooled_ratio(self):
        reference = [
            {"uid": 0, "committed": 4},
            {"uid": 1, "committed": 2},
        ]
        treatment = [
            {"uid": 0, "committed": 2},
            {"uid": 1, "committed": 1},
        ]

        point, low, high = paired_cluster_bootstrap_ratio(
            reference, treatment, iterations=200
        )

        self.assertEqual(point, 0.5)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)


if __name__ == "__main__":
    unittest.main()
