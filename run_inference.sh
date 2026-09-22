#!/bin/bash
# Run SegmentAnyTree inference on a folder of .las / .laz / .ply point clouds.
#
#   bash run_inference.sh <input_dir> <output_dir> [clean_output_dir=true]
#
# Environment variables:
#   SAT_MODEL_DIR   directory containing PointGroup-PAPER.pt   (default: <repo>/model_file)
#   SAT_N_JOBS      files converted in parallel in the utm2local step (default: 4)
#
# Results: <output_dir>/final_results/<name>_out.laz with PredSemantic and PredInstance
# dimensions and the CRS of the input file.
set -euo pipefail

SOURCE_DIR="${1:-}"
DEST_DIR="${2:-}"
CLEAN_OUTPUT_DIR="${3:-true}"

# The directory this script lives in; everything else is relative to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SAT_MODEL_DIR:-$SCRIPT_DIR/model_file}"
MODEL_FILE="$MODEL_DIR/PointGroup-PAPER.pt"

: "${SOURCE_DIR:=$SCRIPT_DIR/data_for_test}"
: "${DEST_DIR:=$SCRIPT_DIR/data_for_test_results}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Usage: run_inference.sh <path_to_input_dir> <path_to_output_dir> [clean_output_dir]"
    echo "Input directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Fail early on the most common setup problem: the checkpoint is a Git LFS pointer, not the model
if [ ! -f "$MODEL_FILE" ]; then
    echo "Model checkpoint not found: $MODEL_FILE (set SAT_MODEL_DIR)"
    exit 1
fi
if [ "$(wc -c < "$MODEL_FILE")" -lt 1000000 ]; then
    echo "$MODEL_FILE is only $(wc -c < "$MODEL_FILE") bytes: it is a Git LFS pointer, not the model."
    echo "Run 'git lfs install --local && git lfs pull' in $SCRIPT_DIR first."
    exit 1
fi

# Make relative paths absolute (hydra changes the working directory)
[[ "$SOURCE_DIR" != /* ]] && SOURCE_DIR="$(pwd)/$SOURCE_DIR"
[[ "$DEST_DIR" != /* ]] && DEST_DIR="$(pwd)/$DEST_DIR"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Input directory:  $SOURCE_DIR"
echo "Output directory: $DEST_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Model:            $MODEL_FILE"

mkdir -p "$DEST_DIR"
if [ "$CLEAN_OUTPUT_DIR" = "true" ]; then
    find "$DEST_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

# Copy the input files so the originals are never touched
mkdir -p "$DEST_DIR/input_data"
cp -r "$SOURCE_DIR/"* "$DEST_DIR/input_data/"

# '-' and spaces in file names become '_' (the inference scripts split names on these)
python3 "$SCRIPT_DIR/nibio_inference/fix_naming_of_input_files.py" "$DEST_DIR/input_data"

# UTM -> local coordinates, written as binary PLY, with the shift stored in *_min_values.json
python3 "$SCRIPT_DIR/nibio_inference/pipeline_utm2local_parallel.py" -i "$DEST_DIR/input_data" -o "$DEST_DIR/utm2local"

# The dataset cache lives under the dataroot; never reuse one from an earlier run
rm -rf "$DEST_DIR/utm2local/treeinsfused"

# Fill the run-specific paths into a copy of conf/eval.yaml
cp "$SCRIPT_DIR/conf/eval.yaml" "$DEST_DIR/eval.yaml"
python3 "$SCRIPT_DIR/nibio_inference/modify_eval.py" "$DEST_DIR/eval.yaml" "$DEST_DIR/utm2local" "$DEST_DIR" \
    --checkpoint_dir "$MODEL_DIR" --dataroot "$DEST_DIR/utm2local"

# Run the model. hydra.run.dir is $DEST_DIR, so result_<i>.ply etc. land there.
python3 "$SCRIPT_DIR/eval.py" --config-name "$DEST_DIR/eval.yaml"

echo "Done with inference using the config file: $DEST_DIR/eval.yaml"

# result_<i>.ply -> instance_segmentation_<name>.ply, semantic_result_<i>.ply -> semantic_segmentation_<name>.ply
python3 "$SCRIPT_DIR/nibio_inference/rename_result_files_instance.py" "$DEST_DIR/eval.yaml" "$DEST_DIR"
python3 "$SCRIPT_DIR/nibio_inference/rename_result_files_segmentation.py" "$DEST_DIR/eval.yaml" "$DEST_DIR"

FINAL_DEST_DIR="$DEST_DIR/final_results"

# Merge predictions onto the input points, shift back to UTM, copy the CRS, write LAZ
python3 "$SCRIPT_DIR/nibio_inference/merge_pt_ss_is_in_folders.py" \
    -i "$DEST_DIR/utm2local" -s "$DEST_DIR" -o "$FINAL_DEST_DIR" -d "$DEST_DIR/input_data" -v

num_files=$(find "$FINAL_DEST_DIR" -maxdepth 1 -type f -name '*.laz' | wc -l)
echo "Number of files in the final results directory: $num_files"
