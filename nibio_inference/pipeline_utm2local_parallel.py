"""Shift each input point cloud so its minimum corner is at the origin.

The model is trained on plot-local coordinates with the ground near z = 0 and
stores coordinates as float32, so UTM eastings/northings must be removed before
inference. The per-file shift is stored next to the output as
<name>_out_min_values.json and added back by merge_pt_ss_is.py.
"""
import argparse
import json
import os

from joblib import Parallel, delayed

from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply
from nibio_inference.ply_to_pandas import ply_to_pandas

POINT_CLOUD_EXTENSIONS = (".ply", ".las", ".laz")


def output_paths(filename, output_folder):
    base_filename = os.path.splitext(filename)[0]
    return (
        os.path.join(output_folder, f"{base_filename}_out.ply"),
        os.path.join(output_folder, f"{base_filename}_out_min_values.json"),
    )


def process_file(filename, input_folder, output_folder):
    input_file_path = os.path.join(input_folder, filename)
    ext = os.path.splitext(filename)[1].lower()
    if not os.path.isfile(input_file_path) or ext not in POINT_CLOUD_EXTENSIONS:
        print(f"Skipping (not a .ply/.las/.laz file): {input_file_path}")
        return None
    output_file_path, json_file_path = output_paths(filename, output_folder)
    modification_pipeline(input_file_path, output_file_path, json_file_path, ext == ".ply")
    return output_file_path


def modification_pipeline(input_file_path, output_file_path, json_file_path, is_ply):
    coord_names = ["x", "y", "z"]
    print(f"Processing in utm2local: {input_file_path}")

    points_df = ply_to_pandas(input_file_path) if is_ply else las_to_pandas(input_file_path)
    # LAS gives X/Y/Z, PLY usually x/y/z; the output always uses lowercase
    points_df = points_df.rename(columns={"X": "x", "Y": "y", "Z": "z"})
    missing = [c for c in coord_names if c not in points_df.columns]
    if missing:
        raise KeyError(f"{input_file_path} has no {missing} column(s); found {list(points_df.columns)}")

    min_values = [float(points_df[name].min()) for name in coord_names]
    for name, min_value in zip(coord_names, min_values):
        points_df[name] = points_df[name] - min_value

    with open(json_file_path, "w") as f:
        print(f"Saving min values to: {json_file_path}")
        json.dump(min_values, f)

    pandas_to_ply(points_df, csv_file_provided=False, output_file_path=output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shift las/laz/ply files to local coordinates and save them as ply.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Folder with .las/.laz/.ply files.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Folder to write the _out.ply files to.")
    parser.add_argument("-j", "--n_jobs", type=int, default=int(os.environ.get("SAT_N_JOBS", "4")),
                        help="Files processed in parallel; each one is held in memory (default 4, env SAT_N_JOBS).")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    filenames = sorted(os.listdir(args.input_folder))

    print(f"Processing {len(filenames)} files...")
    written = Parallel(n_jobs=args.n_jobs)(
        delayed(process_file)(filename, args.input_folder, args.output_folder) for filename in filenames
    )
    written = [w for w in written if w]
    if not written:
        raise SystemExit(f"No .ply/.las/.laz files found in {args.input_folder}")
    print(f"{len(written)} output files are saved in: {args.output_folder}")
