"""Time the per-batch clustering that runs in training after prepare_epoch.

Loads a checkpoint and a dataset the same way the trainer does, runs a few training
batches through the model forward with clustering on, and reports how long region
growing, mean shift (scikit-learn vs the batched torch implementation) and the score
network take per batch.

  python scripts/profile_clustering.py --checkpoint_dir <dir with PointGroup-PAPER.pt> \
      --dataroot <dataroot> [--batches 5] [--batch_size 6] [--epoch 40]
"""
import argparse
import time

import numpy as np
import torch
from omegaconf import OmegaConf

import torch_points3d.models.panoptic.PointGroup3heads as pg
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint
from torch_points3d.utils import meanshift_cluster
from torch_points3d.utils.meanshift_gpu import mean_shift


class Timer:
    def __init__(self):
        self.totals = {}
        self.samples = []

    def wrap(self, name, fn):
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            t = time.time()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            self.totals[name] = self.totals.get(name, 0.0) + time.time() - t
            return out
        return wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epoch", type=int, default=40, help="epoch passed to forward; > prepare_epoch enables clustering")
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = OmegaConf.create({"model_name": "PointGroup-PAPER", "data": {"fold": [], "dataroot": args.dataroot}})
    checkpoint = ModelCheckpoint(args.checkpoint_dir, "PointGroup-PAPER", "latest", run_config=cfg, resume=False)
    dataset = instantiate_dataset(checkpoint.data_config)
    model = checkpoint.create_model(dataset, weight_name="latest").to(device)
    model.eval()
    dataset.create_dataloaders(model, args.batch_size, True, 0, False)
    print("epochs in checkpoint:", len(checkpoint.get_stats()["train"]) if hasattr(checkpoint, "get_stats") else "?")

    timer = Timer()
    pg.region_grow = timer.wrap("region_grow", pg.region_grow)
    model._compute_score = timer.wrap("score_net", model._compute_score)
    captured = []

    original_meanshift = meanshift_cluster.meanshift_cluster

    def capturing_meanshift(prediction, bandwidth):
        as_numpy = prediction.detach().cpu().numpy() if torch.is_tensor(prediction) else np.asarray(prediction)
        captured.append((as_numpy, bandwidth))
        return original_meanshift(prediction, bandwidth)

    meanshift_cluster.meanshift_cluster = timer.wrap("meanshift", capturing_meanshift)

    loader = dataset.train_dataloader
    n_points = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= args.batches:
                break
            model.set_input(data, device)
            n_points.append(data.pos.shape[0])
            torch.cuda.synchronize()
            t = time.time()
            model.forward(epoch=args.epoch)
            torch.cuda.synchronize()
            timer.totals["forward_total"] = timer.totals.get("forward_total", 0.0) + time.time() - t

    print(f"\n{args.batches} batches of {args.batch_size}, mean {np.mean(n_points):.0f} points per batch")
    for k, v in sorted(timer.totals.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v / args.batches:7.2f} s per batch")

    if captured:
        sizes = [len(x) for x, _ in captured]
        print(f"\nmean shift calls: {len(captured)} (per batch {len(captured) / args.batches:.1f}), "
              f"points per call min/mean/max {min(sizes)}/{np.mean(sizes):.0f}/{max(sizes)}")
        from sklearn.cluster import MeanShift
        from sklearn.metrics import adjusted_rand_score
        t_gpu, t_cpu, aris, n_ref, n_ours = 0.0, 0.0, [], [], []
        for x, bw in captured:
            xt = torch.from_numpy(x).to(device)
            torch.cuda.synchronize(); t = time.time()
            ours = mean_shift(xt, bw)
            torch.cuda.synchronize(); t_gpu += time.time() - t
            t = time.time()
            ref = torch.from_numpy(MeanShift(bandwidth=bw, bin_seeding=True).fit(x).labels_)
            t_cpu += time.time() - t
            aris.append(adjusted_rand_score(ref.numpy(), ours.numpy()))
            n_ref.append(len(np.unique(ref.numpy()))); n_ours.append(len(np.unique(ours.numpy())))
        print(f"  sklearn total {t_cpu:.1f} s, torch GPU total {t_gpu:.1f} s  (x{t_cpu / max(t_gpu, 1e-6):.0f})")
        print(f"  agreement (ARI) min/mean {min(aris):.3f}/{np.mean(aris):.3f}; clusters per call sklearn {np.mean(n_ref):.1f} vs ours {np.mean(n_ours):.1f}")


if __name__ == "__main__":
    main()
