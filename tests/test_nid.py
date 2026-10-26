"""CPU-only regression checks against the retained NID implementations."""

import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from nid import compute_nid, extract_labels


class NIDTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(17).normal(size=(80, 5))
        self.y = np.repeat([0, 1], 40)

    def test_swap_matches_retained_source(self):
        # Frozen outputs obtained by evaluating the unmodified retained
        # nid_swap*.py calculation functions on this embedding fixture.
        cases = [
            ("l2", "zscore", 0.6596827374404473),
            ("linf", "zscore", 0.44120846360159305),
            ("cosine", "zscore", 1.121451689855399),
            ("l2", "none", 0.6713915219625612),
            ("l2", "minmax", 0.11597706176411454),
            ("l2", "robust", 1.0798820041002042),
        ]
        for metric, normalization, expected in cases:
            with self.subTest(metric=metric, normalization=normalization):
                result = compute_nid(
                    self.x, self.y, variant="swap", traversals=4, seed=23,
                    metric=metric, normalization=normalization,
                )
                self.assertAlmostEqual(result["nid"], expected, places=12)
                self.assertEqual(result["comparable_classes_per_round"], [2] * 4)
                self.assertFalse(result["threshold_selected"])

    def test_legacy_matches_retained_source(self):
        # Encoder extraction is bypassed with the same fixed embeddings;
        # partitioning and arithmetic are those of the 2023 source.
        cases = [
            ("legacy_random", "mean", 0.6658679659155872),
            ("legacy_random", "var", 0.04585793154962285),
            ("legacy_swap", "mean", 0.9641660035346372),
            ("legacy_swap", "var", 0.018972097823450502),
        ]
        for variant, aggregation, expected in cases:
            with self.subTest(variant=variant, aggregation=aggregation):
                result = compute_nid(
                    self.x, self.y, variant=variant, traversals=4,
                    seed=23, aggregation=aggregation,
                )
                self.assertAlmostEqual(result["nid"], expected, places=12)

    def test_label_selection_is_explicit_and_preserves_metadata(self):
        labels = np.stack([np.repeat([5, 9], 40), np.arange(80)], axis=1)
        before = labels.copy()
        with self.assertRaises(ValueError):
            extract_labels(labels, None)
        np.testing.assert_array_equal(extract_labels(labels, 0), labels[:, 0])
        np.testing.assert_array_equal(labels, before)
        temporal = np.repeat(labels[:, None, :], 3, axis=1)
        np.testing.assert_array_equal(extract_labels(temporal, 0), labels[:, 0])
        temporal[0, 1, 0] += 1
        with self.assertRaises(ValueError):
            extract_labels(temporal, 0)

    def test_does_not_mutate_inputs_or_global_random_generators(self):
        x_before, y_before = self.x.copy(), self.y.copy()
        random.seed(77)
        np.random.seed(77)
        py_state, np_state = random.getstate(), np.random.get_state()
        for variant in ("swap", "legacy_random", "legacy_swap"):
            first = compute_nid(self.x, self.y, variant=variant, traversals=4, seed=23)
            second = compute_nid(self.x, self.y, variant=variant, traversals=4, seed=23)
            self.assertEqual(first, second)
        self.assertEqual(random.getstate(), py_state)
        after_np_state = np.random.get_state()
        self.assertEqual(after_np_state[0], np_state[0])
        np.testing.assert_array_equal(after_np_state[1], np_state[1])
        self.assertEqual(after_np_state[2:], np_state[2:])
        np.testing.assert_array_equal(self.x, x_before)
        np.testing.assert_array_equal(self.y, y_before)

    def test_missing_classes_are_reported(self):
        result = compute_nid(self.x, np.arange(80), variant="swap", traversals=4)
        self.assertEqual(result["nid"], 0.0)
        self.assertEqual(result["comparable_classes_per_round"], [0] * 4)
        self.assertTrue(result["warnings"])

    def test_invalid_inputs_and_unsupported_combinations_fail(self):
        with self.assertRaises(TypeError):
            compute_nid(self.x, self.y)
        with self.assertRaises(ValueError):
            compute_nid(self.x, self.y, variant="swap", traversals=41)
        with self.assertRaises(ValueError):
            compute_nid(self.x, self.y, variant="swap", metric="cosine", normalization="none")
        with self.assertRaises(ValueError):
            compute_nid(self.x, self.y, variant="legacy_random", metric="linf")
        nonfinite = self.x.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_nid(nonfinite, self.y, variant="swap")
        with self.assertRaises(ValueError):
            compute_nid(np.ones((80, 5)), self.y, variant="legacy_random", traversals=2)
        with self.assertRaises(ValueError):
            compute_nid(np.arange(4).reshape(2, 2), np.zeros(2),
                        variant="legacy_random", seed=0, traversals=1)

    def test_cli_requires_variant_and_writes_finite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            np.save(tmp / "embeddings.npy", self.x)
            np.save(tmp / "labels.npy", self.y)
            command = [
                sys.executable, str(REPO_ROOT / "nid.py"),
                "--embeddings", str(tmp / "embeddings.npy"),
                "--labels", str(tmp / "labels.npy"),
            ]
            missing = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("--variant", missing.stderr)
            output = tmp / "result.json"
            done = subprocess.run(
                command + ["--variant", "swap", "--traversals", "4", "--seed", "23",
                           "--output", str(output)],
                capture_output=True, text=True, check=True,
            )
            result = json.loads(done.stdout)
            self.assertEqual(result, json.loads(output.read_text()))
            self.assertAlmostEqual(result["nid"], 0.6596827374404473, places=12)
            self.assertEqual(result["variant"], "swap")


if __name__ == "__main__":
    unittest.main()
