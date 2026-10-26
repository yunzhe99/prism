"""Lloyd k-means adapted from the retained Prism kmeans.py implementation."""

import torch


@torch.no_grad()
def kmeans(features, n_clusters, generator=None, max_iterations=1000, tolerance=1e-4):
    if features.ndim != 2 or len(features) < n_clusters or n_clusters < 1:
        raise ValueError("k-means requires 1 <= domains <= number of training samples")
    if not torch.isfinite(features).all():
        raise ValueError("k-means features contain non-finite values")
    # CPU generator makes seeded initialization independent of the selected device.
    indices = torch.randperm(len(features), generator=generator)[:n_clusters]
    centers = features[indices.to(features.device)].clone()
    for _ in range(max_iterations):
        distance = torch.cdist(features, centers)
        assignments = distance.argmin(dim=1)
        updated = centers.clone()
        for index in range(n_clusters):
            members = features[assignments == index]
            if len(members):
                updated[index] = members.mean(dim=0)
            else:
                # The old mean(empty) produced NaN. Reinitialize from a far sample.
                updated[index] = features[distance.min(dim=1).values.argmax()]
        shift = (updated - centers).norm(dim=1).sum().square()
        centers = updated
        if shift.item() < tolerance:
            break
    # Preserve the original ordering convention for domain-specific heads.
    return centers[centers.mean(dim=1).argsort()]
