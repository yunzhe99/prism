# Third-party notices

This package contains original Prism contributions and adaptations of external research code. A license for the Prism authors' original contributions does not replace the licenses or copyright notices below. In particular, the complete package must not be described as exclusively MIT-licensed.

The source and license links below were checked on 2026-10-20. They identify the upstream material used here.

## TS-TCC — MIT

Copyright (c) 2022 Emadeldeen Eldele.

The residual, pre-normalization, feed-forward, attention, transformer-stack, and sequence-token components in `models/attention.py` correspond to [TS-TCC's attention implementation](https://github.com/emadeldeen24/TS-TCC/blob/main/models/attention.py), with modifications in the retained Prism source. The FCN convolution-block structure in `models/backbones.py` also appears in [TS-TCC's model implementation](https://github.com/emadeldeen24/TS-TCC/blob/main/models/model.py). Prism adds its own domain-specific heads and routing behavior.

The upstream [MIT license](https://github.com/emadeldeen24/TS-TCC/blob/main/LICENSE.txt) is preserved in [licenses/TS-TCC-MIT.txt](licenses/TS-TCC-MIT.txt).

## The Annotated Transformer — MIT

Copyright (c) 2018 Alexander Rush.

`PositionalEncoding` in `models/attention.py` corresponds to the sinusoidal position-encoding implementation in [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding). This attribution is to the matching Harvard implementation, rather than an assumption that all position-encoding implementations have the same license.

The upstream [MIT license](https://github.com/harvardnlp/annotated-transformer/blob/master/LICENSE) is preserved in [licenses/annotated-transformer-MIT.txt](licenses/annotated-transformer-MIT.txt).

## kmeans_pytorch — MIT

Copyright (c) 2020 subhadarshi.

The retained Prism `kmeans.py` implementation corresponds to [kmeans_pytorch](https://github.com/subhadarship/kmeans_pytorch/blob/master/kmeans_pytorch/__init__.py). `utils/clustering.py` is a smaller adaptation of that retained implementation, with explicit random initialization, input checks, a finite iteration budget, and empty-cluster handling. Its attribution is preserved despite those changes.

The upstream [MIT license](https://github.com/subhadarship/kmeans_pytorch/blob/master/LICENSE) is preserved in [licenses/kmeans-pytorch-MIT.txt](licenses/kmeans-pytorch-MIT.txt).

## CL-HAR — historical implementation reference

The historical Prism model sources used [CL-HAR](https://github.com/Tian0426/CL-HAR), including its [DeepConvLSTM implementation](https://github.com/Tian0426/CL-HAR/blob/main/models/backbones.py). No explicit project license was located in that upstream repository during this check, and this package does not assert an MIT grant from CL-HAR.

For this release, the DeepConvLSTM constructor and forward implementation were reimplemented using a convolution stack and a batch-first LSTM. The earlier sequence of CL-HAR-derived constructor and forward statements is not retained. The four-convolution/two-layer-LSTM architecture, dimensions, and feature order remain the reference; checkpoint parameter names have changed. CL-HAR is acknowledged for that historical implementation reference. This acknowledgement does not relicense its repository.

The retained FCN and attention components have the separately identified MIT sources above. No claim is made that TS-TCC supplies a DeepConvLSTM implementation.

## Retained video utilities — comi MIT and separate GPL detector scripts

The retained video research utilities originated in the authors' `comi` repository. Its MIT notice, Copyright (c) 2022 yunzhe99, is preserved in [video/LICENSE-comi-MIT](video/LICENSE-comi-MIT). It covers the comi-origin material to which that license applies; it does not relicense external code. The per-file source correspondence is documented in [video/SOURCE_MAP.md](video/SOURCE_MAP.md).

`video/detector_gpl/yolo_train.py` and `video/detector_gpl/yolo_val.py` retain substantial structure from the `pytorchyolo` / PyTorch-YOLOv3 training and evaluation scripts. These files are distributed under GPL-3.0, not solely under the comi MIT notice. The GPL text from the inspected `pytorchyolo` 1.8.0 package is preserved in [video/detector_gpl/LICENSE](video/detector_gpl/LICENSE). The upstream project is [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3). See [the detector notice](video/detector_gpl/README.md) for scope.

## Dependencies, datasets, and model weights

Imported packages retain their respective licenses. Installing a dependency does not change its license to the project license. Package source, pretrained weights, dataset images, sensor records, and private experiment artifacts are not bundled here.

Dataset names, conversion scripts, and experiment commands do not grant permission to redistribute the datasets. Obtain each dataset and any model weights from its authorized source and follow its own terms.
