import numpy as np
import pytest
import torch
from sklearn.cluster import MeanShift
from sklearn.metrics import adjusted_rand_score

# load the module by path: importing torch_points3d.utils pulls in matplotlib, omegaconf, ...
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "meanshift_gpu", os.path.join(os.path.dirname(__file__), "..", "torch_points3d", "utils", "meanshift_gpu.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
mean_shift = _mod.mean_shift


def blobs(n_clusters=12, points_per_cluster=400, dim=5, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-4, 4, size=(n_clusters, dim))
    x = np.concatenate([c + rng.normal(0, spread, size=(points_per_cluster, dim)) for c in centers])
    truth = np.repeat(np.arange(n_clusters), points_per_cluster)
    return x.astype(np.float32), truth


@pytest.mark.parametrize("bandwidth", [0.6, 1.0])
def test_matches_sklearn_meanshift(bandwidth):
    x, truth = blobs()
    ref = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(x)
    ours = mean_shift(torch.from_numpy(x), bandwidth)

    assert ours.shape == (len(x),)
    assert ours.dtype == torch.long
    assert adjusted_rand_score(ref.labels_, ours.numpy()) > 0.99
    assert abs(len(np.unique(ours.numpy())) - len(ref.cluster_centers_)) <= 1
    assert adjusted_rand_score(truth, ours.numpy()) > 0.99


def test_single_cluster_and_tiny_inputs():
    x = torch.zeros(10, 5) + torch.rand(10, 5) * 0.01
    assert mean_shift(x, 0.6).unique().numel() == 1
    assert mean_shift(torch.empty(0, 5), 0.6).numel() == 0
    assert mean_shift(torch.rand(1, 5), 0.6).tolist() == [0]


def test_matches_sklearn_on_overlapping_clusters():
    """Harder case: clusters closer than the bandwidth, where the merge order matters."""
    x, _ = blobs(n_clusters=30, points_per_cluster=200, spread=0.3, seed=3)
    ref = MeanShift(bandwidth=0.6, bin_seeding=True).fit(x)
    ours = mean_shift(torch.from_numpy(x), 0.6)
    assert adjusted_rand_score(ref.labels_, ours.numpy()) > 0.95
