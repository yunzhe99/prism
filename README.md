# Prism: Task-aware Domain Partitioning in Non-i.i.d. Data for Flexible Mobile Perception

Research code for the paper above, by Yunzhe Li, Facheng Hu, Hongzi Zhu and colleagues. An earlier conference version of this work appeared at IEEE INFOCOM 2025.

## Citation

```bibtex
@inproceedings{li2025prism,
  title     = {Prism: Mining Task-Aware Domains in Non-i.i.d. IMU Data for
               Flexible User Perception},
  author    = {Li, Yunzhe and Hu, Facheng and Zhu, Hongzi and Liu, Quan and
               Zhao, Xiaoke and Shen, Jiangang and Chang, Shan and Guo, Minyi},
  booktitle = {IEEE INFOCOM},
  pages     = {1--10},
  year      = {2025},
  publisher = {IEEE}
}
```

If you use this code, please cite the paper above.

## Available components

| Component | What this release provides |
|---|---|
| IMU training | Pretraining, alternating clustering and training, FCN/ConvLSTM/Transformer encoders, domain-specific classifiers |
| Inputs | Explicit NumPy train/validation/test arrays for UCI, HHAR, Motion, or SHL |
| Evaluation | Validation macro-F1 checkpoint selection, followed by one test evaluation |
| NID | Explicitly selected calculation variants and training-embedding export |
| UCI conversion | Official 128-sample window converter |
| Video | BDD conversion, retained ResNet feature grouping, YOLO training/evaluation utilities; see [video documentation](video/README.md) |

## Install

Use a separate Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-imu.txt
```

The CPU checks used Python 3.8.13, PyTorch 1.12.1+cu102, NumPy 1.23.4, scikit-learn 1.0, and einops 0.6.0. The dependency file specifies compatibility ranges. Video dependencies are separate. The software has no automatic dataset or weight downloads.

## Run IMU training

Supply numeric `.npy` features with shape `[samples, time, channels]` and task labels with shape `[samples]`. For labels containing multiple fields, set `--label-index`; for domain-metadata reporting, additionally set `--metadata-index` or supply separate metadata arrays. Temporal labels must be constant within each window.

This example assumes already-normalized, one-dimensional task labels:

```bash
python prism.py \
  --dataset uci --backbone FCN \
  --train-features data/train_x.npy --train-labels data/train_y.npy \
  --val-features data/val_x.npy --val-labels data/val_y.npy \
  --test-features data/test_x.npy --test-labels data/test_y.npy \
  --normalization none --domains 4 \
  --pretrain-epochs 100 --epochs 200 --batch-size 128 \
  --lr 0.001 --contrastive-weight 1 --margin 1 \
  --seed 42 --device cpu \
  --export-pretrain-embeddings outputs/m0.npy \
  --output outputs/example-run
```

These numbers are an example. Select the domain count and other hyperparameters using validation data. Keep the test set separate throughout selection. Use `--device cuda:0` only when an appropriate GPU is available.

`--normalization raw_imu` divides acceleration channels by 9.8 and, for nine-channel inputs, normalizes the magnetometer vector. Use `none` for inputs already normalized under your documented protocol.

The output directory must not already exist. It contains `model.pt`, `metrics.json`, and `history.csv`; the metrics record the implementation, arguments, selected epoch, and that historical results have not been reproduced. Exact input requirements are in [input_contract.json](configs/imu/input_contract.json).

## Calculate NID

The training command exports pretrained encoder embeddings and the matching training labels as `outputs/m0.npy` and `outputs/m0_labels.npy`.

```bash
python nid.py \
  --embeddings outputs/m0.npy --labels outputs/m0_labels.npy \
  --variant swap --traversals 10 --seed 42 \
  --output outputs/nid.json
```

NID variants are deliberately explicit because the retained scripts use different aggregation rules. See the [NID documentation](scripts/nid/README.md). This command computes the statistic; it does not select an application threshold or trigger model training automatically.

## Verification

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

For a small, CPU-only run through training, checkpoint selection, embedding export, and NID:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python scripts/smoke_imu.py --output outputs/smoke
```

Checks cover loss arithmetic, gradient flow and encoder updates, routing, input validation, train/validation/test separation in the training interface, and the NID calculations. These are software checks on synthetic data, not measurements of manuscript accuracy.

## Source and licenses

The release is distributed as a source snapshot. Original Prism contributions use the [MIT license](LICENSE). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) describes upstream components and applicable licenses. The YOLO scripts retain GPL-3.0 terms, and the video utilities retain their MIT notice. Dataset and pretrained-weight terms are separate.
