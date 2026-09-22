 
![SegmentAnyTree_logo](https://github.com/user-attachments/assets/8849a4b2-3bb3-4c6d-b1f1-13f91efc0936)

## Description
This repo includes the code for training and inference for the method developed by [Wielgosz et al. (2024)SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data. Remote Sensing of Environment](https://www.sciencedirect.com/science/article/pii/S0034425724003936). 

Under the hood, SegmentAnyTree relies on the [torch-points3d framework](https://github.com/torch-points3d/torch-points3d) as the code base. So please take a look there for more information regarding the training and parametrization of the code.

## Usage
The code has been tested on a Linux machine and it relies on a docker image. The method has not been tested in a Windows environment and parts of the code (e.g. Minkowski Engine) might not be available for Windows.

### Using the docker image
There is a quick start for all who want to quickly process the data.

### Quick start
1. Create the necessary folders
2. Upload your files to the folders
3. Pull docker image
4. Run the SaT model
5. Check the results in the output folder

```
mkdir -p $HOME/segmentanytree/input
mkdir -p $HOME/segmentanytree/output

docker pull maciekwielgosz/segment-any-tree:latest

docker run -it --rm --gpus all \
  --mount type=bind,source=$HOME/segmentanytree/input,target=/home/nibio/mutable-outside-world/bucket_in_folder \
  --mount type=bind,source=$HOME/segmentanytree/output,target=/home/nibio/mutable-outside-world/bucket_out_folder \
  maciekwielgosz/segment-any-tree:latest

```

### Additional information

A pre-built docker image can be pulled from [here](https://hub.docker.com/repository/docker/donaldmaen/segment-any-tree/general)

You can also download the docker image : `docker pull maciekwielgosz/segment-any-tree:latest`

There is also image for cuda 11.8.0 available : `docker pull maciekwielgosz/segment-any-tree-cuda11.8.0` . This one is not well tested.

In order to run code using docker container you should edit the content of `run_docker_locally.sh` file.  You should change the following lines:
```
docker run -it --gpus all \
    --name $CONTAINER_NAME \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_in_folder,target=/home/nibio/mutable-outside-world/bucket_in_folder \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_out_folder,target=/home/nibio/mutable-outside-world/bucket_out_folder \
    $IMAGE_NAME
```

You should change the mounting folders : 
```
 --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_in_folder
```
and 
```
 --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_out_folder
```
to match your folders where you keep your local point clound files (las, ply, laz or zip ) to be processed. 

Once you introduce changes to `run_docker_locally.sh` file, you should run it: `bash run_docker_locally.sh` and expect the results in your output folder e.g. `/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/bucket_out_folder`.


## Inference
`run_inference.sh` runs the whole pipeline on a folder of point clouds. This is what the docker image
runs internally; you can also run it directly on a Linux machine with an NVIDIA GPU and the
dependencies from the `Dockerfile` installed.

```bash
git lfs install --local && git lfs pull      # once: fetch the 665 MB checkpoint model_file/PointGroup-PAPER.pt
bash run_inference.sh <input_dir> <output_dir> [clean_output_dir=true]
```

- **Input**: a flat folder of `.las`, `.laz` or `.ply` files, one plot per file. Only x/y/z are needed;
  other attributes are passed through. Georeferenced (UTM) coordinates are fine: each file is shifted to
  local coordinates before inference and shifted back afterwards.
- **Output**: `<output_dir>/final_results/<name>_out.laz` (LAS 1.4, point format 6) with the original
  attributes plus `PredSemantic` (0 = non-tree, 1 = tree) and `PredInstance` (tree id from 1, 0 = no tree).
  The CRS of the input LAS/LAZ is copied to the output.
- `SAT_MODEL_DIR` points to a directory with a different `PointGroup-PAPER.pt` (default `model_file/`).
- `tracker_options.save_viz: True` in `conf/eval.yaml` writes per-proposal debug PLYs (`viz*/`); off by default.

Steps the script performs: copy inputs to `<output_dir>/input_data`, normalise file names
(`nibio_inference/fix_naming_of_input_files.py`), shift to local coordinates
(`pipeline_utm2local_parallel.py`), fill the run paths into a copy of `conf/eval.yaml` (`modify_eval.py`),
run `eval.py`, rename `result_<i>.ply` / `semantic_result_<i>.ply` to the input names, and merge the
predictions back onto the input points (`merge_pt_ss_is_in_folders.py`).

Tests for the pre/post-processing scripts (no GPU needed): `pip install pytest laspy[lazrs] pyproj && pytest tests/`

## Training
```bash
python train.py task=panoptic data=panoptic/treeins_rad8 models=panoptic/area4_ablation_3heads_5 \
    model_name=PointGroup-PAPER training=treeins job_name=treeins_my_first_run \
    data.dataroot=/path/to/data training.epochs=150 training.batch_size=4 training.wandb.log=False
```
(These are also the defaults in `conf/config.yaml`, so `python train.py data.dataroot=... training.wandb.log=False` works.)

Data layout: `<dataroot>/treeinsfused/raw/<region>/*.ply`, binary PLY with `x, y, z, semantic_seg, treeID`.
`semantic_seg`: 0 = unclassified (ignored), 1 = non-tree, 2 = tree. `treeID`: instance id, 0 for non-tree points.
Files whose names end in `val.ply` go to validation, `test.ply` to test, everything else to training.
Coordinates should be plot-local with the ground near z = 0 (the same convention as inference).
`sample_data_conversion.py` shows the LAS to PLY conversion used for the paper.

Instance clustering only runs after `prepare_epoch` (30) epochs, so train for more than that.
Set `training.wandb.entity` to your own account if you keep `training.wandb.log=True`.

## Issues
If you encounter any issues with the code please provide your feedback by raising an issue in this repo rather than contacting the paper authors!

## Citation
If you use the code or data in this repository for your research or project, please make sure to cite the associated article:

```
@article{WIELGOSZ2024114367,
title = {SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data},
journal = {Remote Sensing of Environment},
volume = {313},
pages = {114367},
year = {2024},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2024.114367},
url = {https://www.sciencedirect.com/science/article/pii/S0034425724003936},
author = {Maciej Wielgosz and Stefano Puliti and Binbin Xiang and Konrad Schindler and Rasmus Astrup},
keywords = {3D deep learning, Instance segmentation, ITC, ALS, TLS, Drones}
}
```
