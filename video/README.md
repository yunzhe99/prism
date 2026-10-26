# Video research code

This directory contains the video utilities and detector entry points of the authors' `comi` research checkout, with portable input paths.

## What is available

| Entry point | Behavior | Notes |
|---|---|---|
| `bdd2yolo.py` | BDD100K tracking boxes to normalized YOLO boxes, original ten-category order | Requires the user's own BDD tracking images and annotations |
| `bdd_gen_txt.py` | Generate detector image-path lists | Writes one manifest per sequence directory |
| `resnet_embedding.py` | Resize 256, center crop 224, tensor conversion, pretrained ResNet-18 **1000-dimensional classification outputs** | Preserves the extractor as it was used |
| `clustering.py` | K-means of those outputs, seed 42 and 10 clusters by default | One-pass grouping |
| `detector_gpl/yolo_train.py` | Retained YOLO detector training entry point | Requires explicit model/data configuration and separately obtained weights |
| `detector_gpl/yolo_val.py` | Retained detector evaluation entry point | Reuses PyTorch-YOLOv3 evaluation code |

The extractor takes an explicit manifest directory and the cluster exporter writes per-cluster manifests. Fit clusters on training data only, and apply a fixed protocol for evaluation.

## Dependencies

Use a separate environment for these historical video dependencies. The versions in `requirements-observed.txt` were read from the original server environment. Python 3.8 was used there; Python 3.9 is also supported by the listed PyTorch packages.

```bash
python -m pip install -r video/requirements-observed.txt
```

The YOLO dependency is [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3), distributed as `pytorchyolo`. Obtain detector configurations and pretrained weights from that project or from your own training. Configure class counts and YOLO output filters for the chosen annotation categories.

## BDD100K conversion

The image root must contain per-sequence directories. The label root must contain one JSON file per sequence in BDD `box_track_20` format. Obtain the data directly from [BDD100K](https://www.bdd100k.com/) under its own terms.

```bash
python video/bdd2yolo.py \
  --images /path/to/bdd100k/images/track/train \
  --labels /path/to/bdd100k/labels/box_track_20/train \
  --output work/bdd-yolo

python video/bdd_gen_txt.py \
  --images work/bdd-yolo/images \
  --output work/bdd-all.txt
```

Conversion creates `images/<sequence>/*.jpg`, `labels/<sequence>/*.txt`, and `bdd.names`. Images are symlinked by default; use `--copy-images` if needed. Source images are not moved or modified. Existing output directories are rejected. The generated all-image list is an inventory, not an evaluation split. Use `--sequences ID1 ID2 ...` to export a chosen sequence list.

## Retained feature grouping

Provide one `.txt` image manifest per source domain. Each line is an absolute image path or a path relative to that manifest. Supply a trusted local ResNet-18 state dictionary; the script never downloads model weights automatically.

```bash
python video/resnet_embedding.py \
  --manifests work/train-manifests \
  --weights /path/to/resnet18-state-dict.pth \
  --output work/train-outputs.joblib \
  --device cpu

python video/clustering.py \
  --embeddings work/train-outputs.joblib \
  --output work/train-clusters \
  --clusters 10 --seed 42 --n-init 10
```

This produces per-cluster image manifests and `centroids.npy`. Use only trusted checkpoint/joblib inputs. Both serialized formats can execute code when loaded by older library versions. No numerical results are bundled or claimed.

## Detector training and evaluation

Prepare a YOLO `.data` file with `train`, `valid`, and `names` entries, and a matching `.cfg` model. The data lists must point to images whose matching labels are in the corresponding `labels` tree. `valid` means validation during training; use a separate `.data` file pointing to a sealed test list for final evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 python video/detector_gpl/yolo_train.py \
  --model /path/to/model.cfg --data /path/to/train.data \
  --pretrained_weights /path/to/initial.weights \
  --epochs 200 --seed 42 --model_name domain0

CUDA_VISIBLE_DEVICES=0 python video/detector_gpl/yolo_val.py \
  --model /path/to/model.cfg --data /path/to/test.data \
  --weights /path/to/domain0_selected.pth
```

These commands use the existing parser and training/evaluation code. Training writes `checkpoints/`, `output/`, and the selected log directory in the current working directory. A validation-based checkpoint selection policy must be specified by the experiment; the trainer saves periodic checkpoints and logs validation metrics.

## Validation performed for this release

All six Python entry points passed syntax compilation and `--help` without loading models or downloading data. BDD conversion and manifest generation passed a synthetic one-frame check with known box coordinates.

## Source and licensing

The original source checkout had an MIT license, copyright 2022 `yunzhe99`, preserved in `LICENSE-comi-MIT`. That notice covers the retained non-YOLO utilities and their portability edits in this directory.

The detector scripts reuse PyTorch-YOLOv3 training/evaluation code. They are isolated under `detector_gpl/` with its **GPL-3.0** license and attribution; the comi MIT notice does not replace their upstream terms. Dataset and pretrained-weight licenses remain separate. See `SOURCE_MAP.md` for the retained source paths and changes.
