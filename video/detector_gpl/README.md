# Retained detector entry points — GPL-3.0

These scripts were retained from the authors' `comi` checkout and use training and evaluation code from [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3). The server's installed `pytorchyolo` 1.8.0 distribution identifies GPL-3.0 as its license; that distribution's complete license text is preserved here. Its package metadata identifies Florian Vahl as the package author.

The enclosing project's MIT notices do not override the upstream GPL terms for these files. No YOLO library source, weights, or detector configuration files are vendored here.

Portability changes made for this release:

- Model and dataset configuration paths must be explicit command-line inputs.
- `--help` works without importing the optional training dependencies.
- GPU selection preserves `CUDA_VISIBLE_DEVICES` unless the trainer receives an explicit `--gpu_id`.
- Removed a local checkpoint-specific plotting helper from the evaluator.
- The evaluator now returns cleanly when no detections are found.

The retained detector implementations do not provide Prism's complete joint video encoder/domain-estimation training or inference routing. See the parent README for usage and the remaining experiment-mapping gaps.
