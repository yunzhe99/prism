"""NID calculations on task-specific encoder embeddings.

The caller must select a historical variant. ``swap`` implements the 2026
2k-clip traversal; ``legacy_random`` and ``legacy_swap`` implement the
2023 classwise, prevalence-weighted calculation. They are different metrics.

This module does not train an encoder, choose an NID threshold, or trigger
domain estimation. Use embeddings from the training split of the selected M0
encoder. See scripts/nid/README.md for the input contract and limitations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np


def _inputs(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(embeddings)
    y = np.asarray(labels)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("embeddings must have shape [samples, embedding_dimensions]")
    if y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("labels must be one-dimensional and match at least two embeddings")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("embeddings must contain finite numbers")
    if not np.issubdtype(y.dtype, np.number) or not np.isfinite(y).all():
        raise ValueError("labels must contain finite numeric class identifiers")
    return x, y


def extract_labels(labels: np.ndarray, label_index: int | None) -> np.ndarray:
    """Select task labels explicitly; metadata columns are never overwritten."""
    labels = np.asarray(labels)
    if labels.ndim == 1:
        if label_index not in (None, 0):
            raise ValueError("one-dimensional labels only support label_index=0")
        return labels
    if labels.ndim not in (2, 3) or label_index is None:
        raise ValueError("two/three-dimensional labels require --label-index")
    if not 0 <= label_index < labels.shape[-1]:
        raise ValueError("label_index is outside the final label dimension")
    if labels.ndim == 2:
        return labels[:, label_index]
    if labels.shape[1] == 0:
        raise ValueError("the temporal label dimension is empty")
    selected = labels[:, :, label_index]
    if not np.all(selected == selected[:, :1]):
        raise ValueError("task labels must be constant within each input window")
    return selected[:, 0]


def _distance(a: np.ndarray, b: np.ndarray, metric: str, scale: float, eps: float) -> float:
    if metric == "cosine":
        a = (a / scale).astype(np.float64, copy=False)
        b = (b / scale).astype(np.float64, copy=False)
        denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
        similarity = float(np.dot(a, b) / denominator)
        return 1.0 - min(1.0, max(-1.0, similarity))
    order = 2 if metric == "l2" else np.inf
    return float(np.linalg.norm((a - b) / scale, ord=order))


def _class_scale(values: np.ndarray, normalization: str, eps: float) -> float:
    if normalization == "none":
        return 1.0
    flat = values.reshape(-1)
    if normalization == "zscore":
        scale = np.std(flat) if len(values) >= 2 else eps
    elif normalization == "minmax":
        scale = np.max(flat) - np.min(flat)
    else:
        scale = np.median(np.abs(flat - np.median(flat)))
    return float(max(float(scale), eps))


def _swap(
    x: np.ndarray,
    y: np.ndarray,
    traversals: int,
    seed: int,
    metric: str,
    normalization: str,
    eps: float,
) -> dict[str, Any]:
    if len(x) < 2 * traversals:
        raise ValueError("swap requires at least 2 * traversals input samples")
    rng = np.random.default_rng(seed)
    clips = np.array_split(rng.permutation(len(x)), 2 * traversals)
    alpha, beta = list(clips[:traversals]), list(clips[traversals:])
    classes, counts = np.unique(y, return_counts=True)
    scales = {c: _class_scale(x[y == c], normalization, eps) for c in classes}
    values, coverage = [], []
    for round_index in range(traversals):
        if round_index:
            j = round_index - 1
            alpha[j], beta[j] = beta[j], alpha[j]
        a_indices, b_indices = np.concatenate(alpha), np.concatenate(beta)
        terms = []
        for c in classes:
            a = a_indices[y[a_indices] == c]
            b = b_indices[y[b_indices] == c]
            if len(a) == 0 or len(b) == 0:
                continue
            terms.append(_distance(x[a].mean(axis=0), x[b].mean(axis=0), metric, scales[c], eps))
        # Retained historical behavior: a round with no comparable class is 0.
        # Coverage is returned so that such a value cannot be mistaken for
        # positive evidence of consistency.
        values.append(float(np.mean(terms)) if terms else 0.0)
        coverage.append(len(terms))
    return {
        "nid": float(np.mean(values)),
        "round_values": values,
        "comparable_classes_per_round": coverage,
        "class_aggregation": "equal_weight_over_classes_present_in_both_halves",
        "classes": classes.tolist(),
        "class_counts": counts.tolist(),
        "class_scales": [scales[c] for c in classes],
        "warnings": (
            ["Some rounds omit classes absent from one half; inspect class coverage."]
            if any(n != len(classes) for n in coverage)
            else []
        ),
    }


def _legacy(
    x: np.ndarray,
    y: np.ndarray,
    traversals: int,
    seed: int,
    divide: str,
    aggregation: str,
) -> dict[str, Any]:
    # These local generators reproduce random.seed / np.random.seed without
    # mutating randomness in an importing training process.
    py_rng, np_rng = random.Random(seed), np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    per_class, all_rounds = [], []

    def split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.array([py_rng.random() < 0.5 for _ in values])
        if not mask.any() or mask.all():
            raise ValueError(
                "legacy random split produced an empty class half; "
                "provide more samples or select another seed explicitly"
            )
        return values[mask].copy(), values[~mask].copy()

    for c in classes:
        values = x[y == c]
        if len(values) < 2:
            raise ValueError("legacy variants require at least two samples per class")
        rounds = []
        if divide == "swap":
            a, b = split(values)
            length = min(len(a), len(b))
            a, b = a[:length], b[:length]
        for _ in range(traversals):
            if divide == "random":
                a, b = split(values)
            else:
                indices = np_rng.choice(length, size=length // 10, replace=False)
                temporary = a[indices].copy()
                a[indices], b[indices] = b[indices], temporary
            scale = float(np.std(np.concatenate((a, b), axis=0)))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("legacy NID is undefined for zero class scale")
            rounds.append(float(np.linalg.norm((a.mean(axis=0) - b.mean(axis=0)) / scale)))
        all_rounds.append(rounds)
        per_class.append(float(np.mean(rounds) if aggregation == "mean" else np.var(rounds)))
    return {
        "nid": float(np.average(per_class, weights=counts)),
        "class_values": per_class,
        "class_round_values": all_rounds,
        "class_aggregation": "weighted_by_original_class_sample_count",
        "round_aggregation": aggregation,
        "classes": classes.tolist(),
        "class_counts": counts.tolist(),
        "warnings": [],
    }


def compute_nid(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    variant: str,
    traversals: int = 10,
    seed: int = 42,
    metric: str = "l2",
    normalization: str = "zscore",
    aggregation: str = "mean",
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Return NID plus diagnostics; ``variant`` is deliberately mandatory.

    Scalar normalization is computed over all elements of a class's embedding
    matrix, matching the retained implementations, not independently per
    embedding dimension. ``zscore`` divides the mean difference by that scalar;
    it does not center every input embedding.
    """
    x, y = _inputs(embeddings, labels)
    if variant not in {"swap", "legacy_random", "legacy_swap"}:
        raise ValueError("choose variant: swap, legacy_random, or legacy_swap")
    if isinstance(traversals, bool) or not isinstance(traversals, (int, np.integer)) or traversals < 1:
        raise ValueError("traversals must be a positive integer")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if metric not in {"l2", "linf", "cosine"}:
        raise ValueError("metric must be l2, linf, or cosine")
    if normalization not in {"zscore", "none", "minmax", "robust"}:
        raise ValueError("unknown normalization")
    if aggregation not in {"mean", "var"}:
        raise ValueError("aggregation must be mean or var")
    if variant == "swap":
        if aggregation != "mean":
            raise ValueError("swap uses mean over traversal rounds")
        if metric != "l2" and normalization != "zscore":
            raise ValueError("retained cosine/linf scripts only support zscore normalization")
        result = _swap(x, y, int(traversals), seed, metric, normalization, eps)
    else:
        if metric != "l2" or normalization != "zscore":
            raise ValueError("legacy variants only support l2 with zscore normalization")
        result = _legacy(x, y, int(traversals), seed, variant.split("_", 1)[1], aggregation)
    result.update(
        variant=variant,
        traversals=int(traversals),
        seed=seed,
        metric=metric,
        normalization=normalization,
        samples=len(x),
        embedding_dimensions=x.shape[1],
        threshold_selected=False,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True, help="M0 training-split embeddings, [N,D]")
    parser.add_argument("--labels", type=Path, required=True, help="matching training-split task labels")
    parser.add_argument("--label-index", type=int, help="required for a multi-column label array")
    parser.add_argument("--variant", required=True, choices=("swap", "legacy_random", "legacy_swap"))
    parser.add_argument("--traversals", type=int, default=10, help="k / compute_time; an example default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", choices=("l2", "linf", "cosine"), default="l2")
    parser.add_argument("--normalization", choices=("zscore", "none", "minmax", "robust"), default="zscore")
    parser.add_argument("--aggregation", choices=("mean", "var"), default="mean")
    parser.add_argument("--output", type=Path, help="optional JSON output; stdout always receives the result")
    args = parser.parse_args(argv)
    try:
        embeddings = np.load(args.embeddings, allow_pickle=False)
        labels = extract_labels(np.load(args.labels, allow_pickle=False), args.label_index)
        result = compute_nid(
            embeddings,
            labels,
            variant=args.variant,
            traversals=args.traversals,
            seed=args.seed,
            metric=args.metric,
            normalization=args.normalization,
            aggregation=args.aggregation,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
