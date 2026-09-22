import argparse
import glob
import os
import sys
from typing import Any

from tqdm import tqdm

from nibio_inference.merge_pt_ss_is import MergePtSsIs

SEMANTIC_PREFIX = "semantic_segmentation_"
INSTANCE_PREFIX = "instance_segmentation_"
LAS_EXTENSIONS = (".las", ".laz", ".LAS", ".LAZ")


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


class MergePtSsIsInFolders(object):
    """For every <name>.ply in input_data_folder_path, merge it with
    semantic_segmentation_<name>.ply and instance_segmentation_<name>.ply from
    segmented_data_folder_path and write <name>.laz to output_data_folder_path.

    original_data_folder_path (optional) is the folder with the untouched input
    LAS/LAZ files; when a matching file is found its CRS is copied to the output.
    utm2local names files <orig>_out.ply, so <orig>.las/.laz is looked up there.
    """

    def __init__(self,
                 input_data_folder_path,
                 segmented_data_folder_path,
                 output_data_folder_path,
                 verbose=False,
                 original_data_folder_path=None,
                 ):
        self.input_data_folder_path = input_data_folder_path
        self.segmented_data_folder_path = segmented_data_folder_path
        self.output_data_folder_path = output_data_folder_path
        self.original_data_folder_path = original_data_folder_path
        self.verbose = verbose

        os.makedirs(self.output_data_folder_path, exist_ok=True)

    def _find_original(self, name):
        if self.original_data_folder_path is None:
            return None
        orig_stem = name[:-len("_out")] if name.endswith("_out") else name
        for ext in LAS_EXTENSIONS:
            candidate = os.path.join(self.original_data_folder_path, orig_stem + ext)
            if os.path.isfile(candidate):
                return candidate
        return None

    def matched_files(self):
        input_data = {stem(f): f for f in glob.glob(os.path.join(self.input_data_folder_path, "*.ply"))}
        if not input_data:
            raise FileNotFoundError(f"No .ply files in {self.input_data_folder_path}")

        segmented = glob.glob(os.path.join(self.segmented_data_folder_path, "*.ply"))
        semantic = {stem(f)[len(SEMANTIC_PREFIX):]: f for f in segmented if stem(f).startswith(SEMANTIC_PREFIX)}
        instance = {stem(f)[len(INSTANCE_PREFIX):]: f for f in segmented if stem(f).startswith(INSTANCE_PREFIX)}

        missing = [name for name in input_data if name not in semantic or name not in instance]
        if missing:
            raise FileNotFoundError(
                f"No {SEMANTIC_PREFIX}<name>.ply / {INSTANCE_PREFIX}<name>.ply in {self.segmented_data_folder_path} "
                f"for: {', '.join(sorted(missing))}. Inference did not produce output for these files."
            )

        return [(input_data[name], semantic[name], instance[name], self._find_original(name))
                for name in sorted(input_data)]

    def merge(self):
        written = []
        for point_cloud, semantic, instance, original in tqdm(self.matched_files()):
            output_path = os.path.join(self.output_data_folder_path, stem(point_cloud) + ".laz")
            if original is None and self.verbose:
                print(f"No original LAS/LAZ found for {point_cloud}; output will have no CRS")
            MergePtSsIs(
                point_cloud=point_cloud,
                semantic_segmentation=semantic,
                instance_segmentation=instance,
                output_file_path=output_path,
                verbose=self.verbose,
                source_las_path=original,
            )()
            written.append(output_path)
        return written

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.verbose:
            print("Merging point cloud, semantic segmentation and instance segmentation.")
        return self.merge()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge point cloud, semantic segmentation and instance segmentation.")
    parser.add_argument("-i", "--input_data_folder_path", type=str, required=True,
                        help="Folder with the utm2local <name>_out.ply files.")
    parser.add_argument("-s", "--segmented_data_folder_path", type=str, required=True,
                        help="Folder with the renamed semantic_segmentation_*/instance_segmentation_* files.")
    parser.add_argument("-o", "--output_data_folder_path", type=str, required=True, help="Output folder.")
    parser.add_argument("-d", "--original_data_folder_path", type=str, default=None,
                        help="Folder with the original LAS/LAZ input files (CRS is copied from them).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    MergePtSsIsInFolders(
        input_data_folder_path=args.input_data_folder_path,
        segmented_data_folder_path=args.segmented_data_folder_path,
        output_data_folder_path=args.output_data_folder_path,
        verbose=args.verbose,
        original_data_folder_path=args.original_data_folder_path,
    )()
