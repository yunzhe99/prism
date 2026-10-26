"""Run the corrected IMU-to-NID path on small synthetic arrays, using CPU only."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
import prism  # noqa: E402


def smoke(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rng = np.random.RandomState(23)
    for split, count in (("train", 24), ("val", 8), ("test", 8)):
        labels = np.arange(count) % 2
        features = (rng.randn(count, 120, 6) + labels[:, None, None] * 0.15).astype(np.float32)
        np.save(output / (split + "_features.npy"), features)
        np.save(output / (split + "_labels.npy"), labels)
    arguments = ["--dataset", "uci", "--normalization", "none", "--domains", "2",
                 "--epochs", "2", "--pretrain-epochs", "1", "--batch-size", "8",
                 "--device", "cpu", "--export-pretrain-embeddings", str(output / "m0.npy"),
                 "--output", str(output / "run")]
    for split in ("train", "val", "test"):
        arguments += ["--" + split + "-features", str(output / (split + "_features.npy")),
                      "--" + split + "-labels", str(output / (split + "_labels.npy"))]
    result = prism.run(prism.get_args(arguments))
    subprocess.run([sys.executable, str(REPOSITORY / "nid.py"),
                    "--embeddings", str(output / "m0.npy"),
                    "--labels", str(output / "m0_labels.npy"), "--variant", "swap",
                    "--traversals", "3", "--output", str(output / "nid.json")], check=True)
    summary = {"implementation": result["implementation"],
               "contrastive_gradient": result["contrastive_gradient"],
               "test_evaluation_count": result["test_evaluation_count"],
               "embedding_shape": list(np.load(output / "m0.npy").shape),
               "verification_data": "synthetic", "device": "cpu"}
    (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="A new directory for synthetic inputs and results")
    smoke(parser.parse_args().output)
