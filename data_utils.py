"""Explicit public-dataset inputs; no private paths or implicit split generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Split:
    features: np.ndarray
    labels: np.ndarray
    metadata: Optional[np.ndarray] = None


def select_column(array, index=None, name="labels"):
    """Read (N,), (N, fields), or constant-within-window (N, T, fields)."""
    array = np.asarray(array)
    if array.ndim == 1:
        if index is not None:
            raise ValueError("%s is 1D: omit its column index" % name)
        selected = array
    elif array.ndim in (2, 3):
        if index is None:
            raise ValueError("%s has multiple dimensions: supply an explicit column index" % name)
        if index < 0 or index >= array.shape[-1]:
            raise ValueError("%s column index is outside the available fields" % name)
        selected = array[..., index]
        if array.ndim == 3:
            if array.shape[1] == 0 or not np.all(selected == selected[:, :1]):
                raise ValueError("%s changes within a window; preprocess inconsistent windows first" % name)
            selected = selected[:, 0]
    else:
        raise ValueError("%s must have 1, 2, or 3 dimensions" % name)
    if np.issubdtype(selected.dtype, np.number) and not np.isfinite(selected).all():
        raise ValueError("%s contains non-finite values" % name)
    return selected.copy()


def fit_label_classes(train_labels):
    classes = np.unique(train_labels)
    if not len(classes):
        raise ValueError("Training labels are empty")
    return classes


def encode_labels(labels, classes):
    """Fit classes on training data only; reject unseen validation/test classes."""
    mapping = {value: index for index, value in enumerate(classes.tolist())}
    missing = set(np.unique(labels).tolist()) - set(mapping)
    if missing:
        raise ValueError("Split contains classes absent from training: %r" % sorted(missing))
    return np.asarray([mapping[value] for value in labels.tolist()], dtype=np.int64)


def normalize_features(features, normalization):
    result = np.asarray(features, dtype=np.float32).copy()
    if normalization == "none":
        return result
    if normalization != "raw_imu":
        raise ValueError("normalization must be raw_imu or none")
    if result.shape[-1] not in (6, 9):
        raise ValueError("raw_imu expects accelerometer+gyroscope, optionally magnetometer")
    result[:, :, :3] /= 9.8
    if result.shape[-1] == 9:
        magnitude = np.linalg.norm(result[:, :, 6:9], axis=-1, keepdims=True)
        result[:, :, 6:9] /= magnitude + 1e-5
    return result


def load_split(features_path, labels_path, label_index=None, metadata_path=None,
               metadata_index=None, normalization="none", classes=None):
    features = np.load(Path(features_path), allow_pickle=False)
    raw_labels = np.load(Path(labels_path), allow_pickle=False)
    if features.ndim != 3 or not all(features.shape):
        raise ValueError("Features must be a nonempty (N, time, channels) numeric array")
    if not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all():
        raise ValueError("Features must be finite numeric values")
    labels = select_column(raw_labels, label_index, "labels")
    metadata = None
    if metadata_path is not None:
        raw_metadata = np.load(Path(metadata_path), allow_pickle=False)
        metadata = select_column(raw_metadata, metadata_index, "metadata")
    elif metadata_index is not None:
        # Extract metadata from the ORIGINAL array, not the already selected labels.
        metadata = select_column(raw_labels, metadata_index, "metadata")
    if len(labels) != len(features) or (metadata is not None and len(metadata) != len(features)):
        raise ValueError("Features, labels, and metadata lengths must agree")
    if classes is not None:
        labels = encode_labels(labels, classes)
    return Split(normalize_features(features, normalization), labels, metadata)


def add_data_arguments(parser):
    parser.add_argument("--dataset", choices=["uci", "hhar", "motion", "shl"], required=True,
                        help="Dataset name for the run record; inputs are supplied explicitly")
    for split in ("train", "val", "test"):
        parser.add_argument("--%s-features" % split, required=True)
        parser.add_argument("--%s-labels" % split, required=True)
        parser.add_argument("--%s-metadata" % split)
    parser.add_argument("--label-index", type=int,
                        help="Required for multidimensional labels; omit for 1D labels")
    parser.add_argument("--metadata-index", type=int,
                        help="Column in original labels, or in separately supplied metadata")
    parser.add_argument("--normalization", choices=["raw_imu", "none"], required=True,
                        help="Use none if inputs are already normalized")


def load_named_split(args, name, classes=None):
    return load_split(getattr(args, name + "_features"), getattr(args, name + "_labels"),
                      args.label_index, getattr(args, name + "_metadata"),
                      args.metadata_index, args.normalization, classes)
