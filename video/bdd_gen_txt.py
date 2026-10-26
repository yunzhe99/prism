"""Generate image manifests from the converted BDD tree."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Converted images directory containing sequence folders.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+",
                        help="Optional sequence IDs; all available sequences otherwise.")
    args = parser.parse_args()
    if not args.images.is_dir():
        parser.error("--images must be an existing directory")
    if args.output.exists():
        parser.error("--output already exists")
    directories = ([args.images / name for name in args.sequences]
                   if args.sequences else sorted(args.images.iterdir()))
    missing = [str(path) for path in directories if not path.is_dir()]
    if missing:
        parser.error("missing sequence directories: " + ", ".join(missing))
    images = [str(path.absolute()) for directory in directories
              for path in sorted(directory.glob("*.jpg"))]
    if not images:
        parser.error("no JPEG images found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(images) + "\n")
    print("Wrote {} image paths to {}".format(len(images), args.output))


if __name__ == "__main__":
    main()
