"""Rename the tracker's result_<i>.ply to instance_segmentation_<input name>.ply.

<i> is the position of the input file in data.fold of the eval config.
"""
import os
import sys

import yaml


def rename_files(yaml_file, directory, old_pattern="result_{}.ply", new_prefix="instance_segmentation_"):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    fold_section = data.get("data", {}).get("fold", [])
    if not fold_section:
        raise ValueError(f"{yaml_file} has an empty data.fold; nothing to rename")

    for index, file_path in enumerate(fold_section):
        old_file_path = os.path.join(directory, old_pattern.format(index))
        new_file_path = os.path.join(directory, new_prefix + os.path.basename(file_path))

        if not os.path.isfile(old_file_path):
            raise FileNotFoundError(
                f"{old_file_path} not found: inference did not write output for {file_path}"
            )
        os.replace(old_file_path, new_file_path)
        print(f"Renamed {old_file_path} to {new_file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_result_files_instance.py <eval.yaml> <directory with result_i.ply>")
        sys.exit(1)

    rename_files(sys.argv[1], sys.argv[2])
