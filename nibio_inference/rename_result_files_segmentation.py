"""Rename the tracker's semantic_result_<i>.ply to semantic_segmentation_<input name>.ply."""
import sys

from nibio_inference.rename_result_files_instance import rename_files as _rename_files


def rename_files(yaml_file, directory):
    _rename_files(yaml_file, directory, old_pattern="semantic_result_{}.ply", new_prefix="semantic_segmentation_")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_result_files_segmentation.py <eval.yaml> <directory with semantic_result_i.ply>")
        sys.exit(1)

    rename_files(sys.argv[1], sys.argv[2])
