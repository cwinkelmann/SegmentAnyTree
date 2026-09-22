"""Merge the model's semantic and instance predictions back onto the input point cloud.

Inputs are the three PLY files that exist after eval.py + rename_result_files_*:
  point_cloud            <name>_out.ply from utm2local (all input attributes, local coords)
  semantic_segmentation  semantic_segmentation_<name>_out.ply (preds, orig_index) for every point
  instance_segmentation  instance_segmentation_<name>_out.ply (preds, orig_index) for points with an instance

The prediction files carry `orig_index`, the row number of each point in the
input cloud, written by the tracker. Merging is therefore a plain array
assignment, not a join on coordinates: duplicate points are safe, and every
input point appears exactly once in the output, in input order.

Output columns: all input columns, x/y/z shifted back to the original (UTM)
frame, PredSemantic (0 = non-tree, 1 = tree) and PredInstance (1..N, 0 = none).
"""
import argparse
import json
import os
import sys

import numpy as np

from nibio_inference.pandas_to_las import pandas_to_las
from nibio_inference.ply_to_pandas import ply_to_pandas


def min_values_path_for(point_cloud_path):
    return os.path.splitext(point_cloud_path)[0] + "_min_values.json"


class MergePtSsIs(object):
    def __init__(self,
                 point_cloud,
                 semantic_segmentation,
                 instance_segmentation,
                 output_file_path,
                 verbose=False,
                 source_las_path=None,
                 ):
        self.point_cloud = point_cloud
        self.semantic_segmentation = semantic_segmentation
        self.instance_segmentation = instance_segmentation
        self.output_file_path = output_file_path
        self.verbose = verbose
        self.source_las_path = source_las_path

    @staticmethod
    def _read_prediction(path, n_points):
        df = ply_to_pandas(path)
        for column in ("orig_index", "preds"):
            if column not in df.columns:
                raise ValueError(
                    f"{path} has no '{column}' column (found {list(df.columns)}). "
                    "It was probably produced by an older version of the tracker; re-run inference."
                )
        index = df["orig_index"].to_numpy()
        if len(index) and (index.min() < 0 or index.max() >= n_points):
            raise ValueError(f"{path}: orig_index out of range for a cloud of {n_points} points")
        if len(np.unique(index)) != len(index):
            raise ValueError(f"{path}: orig_index contains duplicates")
        return index, df["preds"].to_numpy()

    def merge(self):
        if self.verbose:
            print("Merging point cloud, semantic segmentation and instance segmentation.")

        point_cloud_df = ply_to_pandas(self.point_cloud)
        point_cloud_df = point_cloud_df.rename(columns={"X": "x", "Y": "y", "Z": "z"})
        n_points = len(point_cloud_df)

        sem_index, sem_preds = self._read_prediction(self.semantic_segmentation, n_points)
        ins_index, ins_preds = self._read_prediction(self.instance_segmentation, n_points)

        pred_semantic = np.zeros(n_points, dtype=np.uint8)
        pred_semantic[sem_index] = sem_preds.astype(np.uint8)

        # tracker instance ids start at 0; output ids start at 1 and 0 means "no instance"
        pred_instance = np.zeros(n_points, dtype=np.uint32)
        pred_instance[ins_index] = ins_preds.astype(np.int64) + 1

        with open(min_values_path_for(self.point_cloud), "r") as f:
            min_x, min_y, min_z = json.load(f)

        merged_df = point_cloud_df
        merged_df["x"] = merged_df["x"].to_numpy(dtype=np.float64) + min_x
        merged_df["y"] = merged_df["y"].to_numpy(dtype=np.float64) + min_y
        merged_df["z"] = merged_df["z"].to_numpy(dtype=np.float64) + min_z
        merged_df["PredSemantic"] = pred_semantic
        merged_df["PredInstance"] = pred_instance

        if self.verbose:
            n_trees = len(np.unique(pred_instance[pred_instance > 0]))
            print(f"{n_points} points, {int((pred_semantic == 1).sum())} tree points, {n_trees} tree instances")

        return merged_df

    def save(self, merged_df):
        return pandas_to_las(
            merged_df,
            csv_file_provided=False,
            output_file_path=self.output_file_path,
            do_compress=True,
            verbose=self.verbose,
            source_las_path=self.source_las_path,
        )

    def run(self):
        if self.verbose:
            print("point_cloud: {}".format(self.point_cloud))
            print("semantic_segmentation: {}".format(self.semantic_segmentation))
            print("instance_segmentation: {}".format(self.instance_segmentation))

        merged_df = self.merge()
        if self.output_file_path is not None:
            self.save(merged_df)

        if self.verbose:
            print("Done for:")
            print("output_file_path: {}".format(self.output_file_path))

        return merged_df

    def __call__(self):
        return self.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge point cloud, semantic segmentation and instance segmentation.")
    parser.add_argument("-pc", "--point_cloud", help="Path to the point cloud file.")
    parser.add_argument("-ss", "--semantic_segmentation", help="Path to the semantic segmentation file.")
    parser.add_argument("-is", "--instance_segmentation", help="Path to the instance segmentation file.")
    parser.add_argument("-o", "--output_file_path", help="Path to the output file.")
    parser.add_argument("--source_las", default=None, help="Original LAS/LAZ whose CRS should be copied.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    MergePtSsIs(
        point_cloud=args.point_cloud,
        semantic_segmentation=args.semantic_segmentation,
        instance_segmentation=args.instance_segmentation,
        output_file_path=args.output_file_path,
        verbose=args.verbose,
        source_las_path=args.source_las,
    )()
