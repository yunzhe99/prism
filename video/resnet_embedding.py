"""ResNet-18 output extraction for the video utilities.

Preserves the legacy resize/crop/tensor transforms and the full 1000-dimensional
classification output.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True,
                        help="Directory of .txt image lists, one list per source domain.")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Trusted local ResNet-18 state_dict; no automatic download.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    import joblib
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import models, transforms
    if not args.weights.is_file() or not args.manifests.is_dir():
        parser.error("weights and manifest directory must exist")
    if args.output.exists():
        parser.error("--output already exists")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    manifests = sorted(args.manifests.glob("*.txt"))
    if not manifests:
        parser.error("no .txt manifests found")
    model = models.resnet18(weights=None)
    # Use a trusted state_dict. Never load an untrusted pickle/checkpoint.
    try:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
    except TypeError:  # Legacy torch 1.12 environment.
        state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    model.to(args.device).eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()
    ])
    all_groups = []
    for manifest in manifests:
        paths = [Path(line.strip()) for line in manifest.read_text().splitlines()
                 if line.strip()]
        paths = [path if path.is_absolute() else manifest.parent / path for path in paths]
        group = []
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            with torch.no_grad():
                outputs = model(torch.stack(images).to(args.device)).cpu().numpy()
            group.extend([([str(path.absolute())], output)
                          for path, output in zip(batch_paths, outputs)])
        all_groups.append(group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_groups, args.output)
    print("Saved {} image outputs from {} manifests".format(
        sum(map(len, all_groups)), len(all_groups)))


if __name__ == "__main__":
    main()
