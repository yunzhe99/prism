# Source mapping

Source checkout: `comi`, last repository commit observed `0908dd6`. Per-file source SHA-256 hashes are recorded below.

| Released file | Retained source path | Changes |
|---|---|---|
| `bdd2yolo.py` | `src/bdd/bdd2yolo.py` | Preserve box formula/category order; explicit roots; source images kept in place; output isolation; Pillow reads dimensions |
| `bdd_gen_txt.py` | `src/bdd/bdd_gen_txt.py` | Explicit inputs/output/sequence IDs; deterministic enumeration; writes only when invoked |
| `resnet_embedding.py` | `src/cluster_resnet/resnet_embedding.py` | Explicit manifests/weights/device; same image transform and full ResNet output |
| `clustering.py` | `src/cluster_resnet/clustering.py` | Explicit inputs/output/k/seed; same k-means grouping; explicit split and detector configuration |
| `detector_gpl/yolo_train.py` | `src/yolo_train.py` | Require model/data paths; portable help; optional GPU override |
| `detector_gpl/yolo_val.py` | `src/yolo_val.py` | Require model/data/weights paths; portable help; remove fixed GPU and checkpoint plotting helper |

The ResNet utilities preserve the one-pass grouping experiment they came from.

## Source hashes

| Source file | SHA-256 |
|---|---|
| `src/bdd/bdd2yolo.py` | `027fafc9fa6ff9f80b844c752278d2a9fbdc4972f69ff14ba37c9b36d53fe747` |
| `src/bdd/bdd_gen_txt.py` | `7be8d152653fe210f410ad987f94eb141a5ba381939725434f8e5e8ae491f702` |
| `src/cluster_resnet/resnet_embedding.py` | `7ffdf5f88ad1174772e17423670d01c9b1a3abeaf3ed253f18c415d1eb6c40b8` |
| `src/cluster_resnet/clustering.py` | `fe4f7364eb2b4d4c2b10d69d3cf46a9c20fb65c630a1f54d89a72024fd7db3c8` |
| `src/yolo_train.py` | `b829239a9ab71cb4485ada340b2ece1e75b621f020e21df11e9f051be4a6dcde` |
| `src/yolo_val.py` | `c9e1d9de8b4453627063bb7aea309a056970acb7b2feeaf27f9dc770dcffa9e0` |
