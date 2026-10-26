"""Portable BDD100K tracking-label conversion retained from comi.

The original box conversion and ten-category order are unchanged. Input images
are no longer moved: an output YOLO tree is created using symlinks or copies.
"""
import argparse
import json
import shutil
from pathlib import Path

CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle", "traffic light", "traffic sign"]


def convert_box(width, height, label):
    roi = label["box2d"]
    w, h = roi["x2"] - roi["x1"], roi["y2"] - roi["y1"]
    x, y = roi["x1"] + w / 2, roi["y1"] + h / 2
    return "{} {} {} {} {}\n".format(
        CLASSES.index(label["category"]), x / width, y / height,
        w / width, h / height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="BDD tracking split directory containing sequence folders.")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Directory of per-sequence BDD tracking JSON labels.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory. Existing output is not overwritten.")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of symlinking them.")
    args = parser.parse_args()
    from PIL import Image
    if not args.images.is_dir() or not args.labels.is_dir():
        parser.error("--images and --labels must be existing directories")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    label_files = sorted(args.labels.glob("*.json"))
    if not label_files:
        parser.error("no per-sequence JSON files found")
    args.output.mkdir(parents=True)
    (args.output / "bdd.names").write_text("\n".join(CLASSES) + "\n")
    converted, missing = 0, 0
    for label_file in label_files:
        sequence = label_file.stem
        frames = json.loads(label_file.read_text())
        for frame in frames:
            # The original converter accepted BDD box_track_20 frame records.
            name = Path(frame["name"]).name
            source = (args.images / sequence / name).resolve()
            if not source.is_file():
                missing += 1
                continue
            image_out = args.output / "images" / sequence / name
            label_out = args.output / "labels" / sequence / Path(name).with_suffix(".txt")
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                width, height = image.size
            lines = [convert_box(width, height, label)
                     for label in (frame.get("labels") or [])
                     if label.get("category") in CLASSES and "box2d" in label]
            label_out.write_text("".join(lines))
            if args.copy_images:
                shutil.copy2(source, image_out)
            else:
                image_out.symlink_to(source)
            converted += 1
    print(json.dumps({"converted_images": converted, "missing_images": missing}))


if __name__ == "__main__":
    main()
