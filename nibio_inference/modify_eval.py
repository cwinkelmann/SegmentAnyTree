"""Fill the run-specific paths into a copy of conf/eval.yaml.

Sets data.fold (the .ply files to predict), data.dataroot (where the dataset
cache goes), checkpoint_dir (where <model_name>.pt lives) and hydra.run.dir
(where eval.py writes its outputs).
"""
import argparse
import os

import yaml


def get_all_ply_paths(directory):
    """Get paths of all .ply files in the given directory (sorted, recursive)."""
    ply_paths = []
    for root, _directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".ply"):
                ply_paths.append(os.path.join(root, filename))
    return sorted(ply_paths)


def modify_yaml(file_path, new_fold, output_dir_path=None, checkpoint_dir=None, dataroot=None):
    """Modify the YAML file in place."""
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    data.setdefault("data", {})
    data["data"]["fold"] = list(new_fold)
    if dataroot:
        data["data"]["dataroot"] = dataroot
    if checkpoint_dir:
        data["checkpoint_dir"] = checkpoint_dir
    if output_dir_path:
        data.setdefault("hydra", {}).setdefault("run", {})["dir"] = output_dir_path

    with open(file_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Fill run-specific paths into an eval.yaml.")
    parser.add_argument("yaml_file_path", help="Path to the eval.yaml to be modified in place.")
    parser.add_argument("folder_path", help="Folder containing the .ply files to predict (utm2local output).")
    parser.add_argument("output_dir_path", help="Directory eval.py should write its outputs to.")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory containing <model_name>.pt.")
    parser.add_argument("--dataroot", default=None,
                        help="Dataset root for the processed cache (default: folder_path).")

    args = parser.parse_args()

    ply_paths = get_all_ply_paths(args.folder_path)
    if not ply_paths:
        raise SystemExit(f"No .ply files found in the directory: {args.folder_path}")

    modify_yaml(
        args.yaml_file_path,
        ply_paths,
        args.output_dir_path,
        checkpoint_dir=args.checkpoint_dir,
        dataroot=args.dataroot or args.folder_path,
    )


if __name__ == "__main__":
    main()
