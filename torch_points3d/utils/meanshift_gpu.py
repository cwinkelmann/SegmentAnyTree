"""Batched flat-kernel mean shift in torch, a drop-in for
sklearn.cluster.MeanShift(bandwidth=b, bin_seeding=True, cluster_all=True).

Same algorithm as scikit-learn: seeds are the occupied cells of a grid with cell size
= bandwidth, every seed is shifted to the mean of the points within `bandwidth`
until it moves less than 1e-3 * bandwidth (or max_iter), seeds are ranked by the
number of points around them and centres within one bandwidth of a stronger centre
are dropped, and every point gets the label of its nearest centre.

All seeds are iterated together on the device the data lives on, which turns the
per-seed Python loop of scikit-learn into a few large matrix operations. On the
GPU this is one to two orders of magnitude faster for the tens of thousands of
5-dimensional embeddings a training cylinder produces.
"""
import numpy as np
import torch


def _bin_seeds(x, bandwidth):
    bins = torch.unique(torch.round(x / bandwidth), dim=0)
    return bins * bandwidth


def _shift_seeds(x, seeds, bandwidth, max_iter, chunk):
    """Return converged centres and their point counts (0 for seeds that lost all points)."""
    bw2 = bandwidth * bandwidth
    stop = 1e-3 * bandwidth
    centers = seeds.clone()
    intensity = torch.zeros(seeds.shape[0], dtype=torch.long, device=x.device)
    active = torch.ones(seeds.shape[0], dtype=torch.bool, device=x.device)
    for _ in range(max_iter + 1):
        idx = torch.nonzero(active, as_tuple=True)[0]
        if idx.numel() == 0:
            break
        for start in range(0, idx.numel(), chunk):
            sel = idx[start:start + chunk]
            c = centers[sel]
            d2 = torch.cdist(c, x).pow_(2)
            within = d2 <= bw2
            counts = within.sum(1)
            has_points = counts > 0
            new_c = (within.float() @ x) / counts.clamp(min=1).unsqueeze(1).float()
            shift = (new_c - c).norm(dim=1)
            converged = has_points & (shift <= stop)
            # sklearn: no points -> stop with the old centre and intensity 0
            centers[sel[has_points]] = new_c[has_points]
            intensity[sel] = torch.where(has_points, counts, torch.zeros_like(counts))
            active[sel[~has_points]] = False
            active[sel[converged]] = False
        # seeds still active after max_iter keep their last centre and count (sklearn stops at max_iter)
    return centers, intensity


def _dedupe_centers(centers, intensity, bandwidth):
    keep = intensity > 0
    centers, intensity = centers[keep].cpu().numpy(), intensity[keep].cpu().numpy()
    if len(centers) == 0:
        return None
    # sklearn orders by (intensity, centre) descending
    order = np.lexsort([-centers[:, k] for k in range(centers.shape[1] - 1, -1, -1)] + [-intensity])
    centers = centers[order]
    diff = centers[:, None, :] - centers[None, :, :]
    within = (diff * diff).sum(-1) <= bandwidth * bandwidth
    unique = np.ones(len(centers), dtype=bool)
    for i in range(len(centers)):
        if unique[i]:
            unique[within[i]] = False
            unique[i] = True
    return centers[unique]


@torch.no_grad()
def mean_shift(x, bandwidth, max_iter=300, chunk=4096):
    """Cluster the rows of x (tensor [N, D], any device). Returns int64 labels on the CPU."""
    x = torch.as_tensor(x).float()
    n = x.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long)
    seeds = _bin_seeds(x, bandwidth)
    centers, intensity = _shift_seeds(x, seeds, bandwidth, max_iter, chunk)
    cluster_centers = _dedupe_centers(centers, intensity, bandwidth)
    if cluster_centers is None:
        raise ValueError("No point was within bandwidth=%f of any seed." % bandwidth)
    cluster_centers = torch.as_tensor(cluster_centers, device=x.device)
    labels = torch.empty(n, dtype=torch.long, device=x.device)
    for start in range(0, n, 65536):
        labels[start:start + 65536] = torch.cdist(x[start:start + 65536], cluster_centers).argmin(1)
    return labels.cpu()
