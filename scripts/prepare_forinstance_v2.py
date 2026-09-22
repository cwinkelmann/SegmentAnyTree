"""Convert FOR-instance V2 (ForAINetV2, Zenodo 16742708) PLY files into the layout train.py expects.

FOR-instance V2 ships binary PLYs with x, y, z, semantic_seg (1 ground, 2 wood, 3 leaf)
and treeID (0 = no tree), named <region>_<plot>_{train,val,test}.ply. This model uses two
semantic classes (1 non-tree, 2 tree, 0 ignored) and plot-local coordinates with the
ground near z = 0 (the same shift run_inference.sh applies at prediction time).

For every input file this script
  - maps semantic_seg 2 and 3 to 2 (tree), keeps 1 (non-tree), everything else becomes 0,
  - subtracts the per-file minimum from x, y, z,
  - keeps treeID as is,
  - writes <output_root>/<region>/<same name>.ply (binary).
Files without semantic_seg/treeID fields (e.g. unlabeled tiles) are skipped.

Usage:
  python scripts/prepare_forinstance_v2.py -i <ForAINetV2>/train_val_data <ForAINetV2>/test_data \
      -o <dataroot>/treeinsfused/raw [-j 4]
then train with data.dataroot=<dataroot>.
"""
import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from plyfile import PlyData, PlyElement

SEMANTIC_MAP = {1: 1, 2: 2, 3: 2}  # FOR-instance V2 class -> SegmentAnyTree class
_SPLIT_SUFFIX = re.compile(r"_(train|val|test)\.ply$")


def region_of(filename):
    """Dataset name from the file name: the prefix repeated before the plot name, or the first token."""
    stem = _SPLIT_SUFFIX.sub("", os.path.basename(filename))
    tokens = stem.split("_")
    # names are <dataset>_<original file name> and the original name starts with the
    # dataset's last token: NIBIO2_NIBIO2_plot12_annotated, NIBIO_MLS_MLS_Plot_56_panoptic
    for k in range(1, len(tokens)):
        if tokens[k - 1] == tokens[k]:
            return "_".join(tokens[:k])
    return tokens[0]


def convert_file(input_path, output_root):
    vertex = PlyData.read(input_path)["vertex"].data
    names = vertex.dtype.names
    if "semantic_seg" not in names or "treeID" not in names:
        print(f"skip (no labels): {input_path}")
        return None

    n = len(vertex)
    out = np.empty(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("semantic_seg", "i4"), ("treeID", "i4")])
    for axis in "xyz":
        values = vertex[axis].astype(np.float64)
        out[axis] = values - values.min()

    semantic = vertex["semantic_seg"].astype(np.int64)
    mapped = np.zeros(n, dtype=np.int32)
    for src, dst in SEMANTIC_MAP.items():
        mapped[semantic == src] = dst
    out["semantic_seg"] = mapped
    out["treeID"] = vertex["treeID"].astype(np.int32)

    output_dir = os.path.join(output_root, region_of(input_path))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(output_path)

    n_tree = int((mapped == 2).sum())
    n_ids = len(np.unique(out["treeID"][out["treeID"] > 0]))
    print(f"{os.path.basename(input_path)}: {n} pts, {n_tree} tree pts, {n_ids} trees -> {output_path}")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input_dirs", nargs="+", required=True, help="Folders with FOR-instance V2 .ply files")
    parser.add_argument("-o", "--output_root", required=True, help="<dataroot>/treeinsfused/raw")
    parser.add_argument("-j", "--n_jobs", type=int, default=4, help="Files converted in parallel (each is held in memory)")
    args = parser.parse_args(argv)

    files = sorted(
        os.path.join(d, f) for d in args.input_dirs for f in os.listdir(d) if f.endswith(".ply")
    )
    if not files:
        sys.exit(f"No .ply files in {args.input_dirs}")
    # largest first so the big file does not end up alone at the end
    files.sort(key=os.path.getsize, reverse=True)

    with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
        written = [p for p in pool.map(convert_file, files, [args.output_root] * len(files)) if p]
    print(f"{len(written)} of {len(files)} files written under {args.output_root}")


if __name__ == "__main__":
    main()
