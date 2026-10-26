"""Parameterize the retained uci_convert.py without changing its 128-sample windows.

This converts the official UCI HAR pre-windowed release, not the paper's complete
20 Hz / 120-sample preprocessing protocol. It preserves official train/test sets;
users must supply their separately specified validation split for training.
"""

import argparse
from pathlib import Path
import numpy as np


def convert(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    signals = ("total_acc_x_", "total_acc_y_", "total_acc_z_",
               "body_gyro_x_", "body_gyro_y_", "body_gyro_z_")
    for split in ("train", "test"):
        arrays = [np.loadtxt(root / split / "Inertial Signals" / (name + split + ".txt"),
                            dtype=np.float32, ndmin=2) for name in signals]
        features = np.stack(arrays, axis=-1)
        activity = np.atleast_1d(np.loadtxt(root / split / ("y_" + split + ".txt"), dtype=np.int64)) - 1
        subject = np.atleast_1d(np.loadtxt(root / split / ("subject_" + split + ".txt"), dtype=np.int64)) - 1
        if len(activity) != len(features) or len(subject) != len(features):
            raise ValueError("UCI signal and label lengths differ")
        labels = np.column_stack((activity, subject))
        labels = np.repeat(labels[:, None, :], features.shape[1], axis=1)
        np.save(output / ("data_" + split + ".npy"), features)
        np.save(output / ("label_" + split + ".npy"), labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Extracted UCI HAR Dataset directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.root, args.output)
