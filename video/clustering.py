"""k-means grouping of ResNet outputs."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Trusted joblib file written by resnet_embedding.py.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for per-cluster image lists and centroids.")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10,
                        help="Explicit legacy sklearn default.")
    args = parser.parse_args()
    import joblib
    import numpy as np
    from sklearn.cluster import KMeans
    if args.output.exists():
        parser.error("--output already exists")
    if args.clusters < 1 or args.n_init < 1:
        parser.error("--clusters and --n-init must be positive")
    groups = joblib.load(args.embeddings)
    names, features = [], []
    for group in groups:
        for names_batch, embedding in group:
            names.append(names_batch[0])
            features.append(embedding)
    matrix = np.asarray(features)
    if len(matrix) < args.clusters:
        parser.error("fewer images than requested clusters")
    clustering = KMeans(n_clusters=args.clusters, random_state=args.seed,
                        n_init=args.n_init).fit(matrix)
    args.output.mkdir(parents=True)
    np.save(args.output / "centroids.npy", clustering.cluster_centers_)
    for index in range(args.clusters):
        selected = [name for name, cluster in zip(names, clustering.labels_)
                    if cluster == index]
        (args.output / "cluster-{}.txt".format(index)).write_text(
            "\n".join(selected) + "\n")
    print("Wrote {} clusters for {} images".format(args.clusters, len(names)))


if __name__ == "__main__":
    main()
