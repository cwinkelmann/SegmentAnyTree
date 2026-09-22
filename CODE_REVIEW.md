# SegmentAnyTree codebase review

Date: 2026-09-22. Scope: the whole repo at commit a3561ed plus the colleague-supplied
`segment_any_tree_gpu.docker`. Goal: run inference on our own georeferenced point clouds,
later train on our own labelled data.

Line references are to files in this repo. Items marked (verified) were checked directly;
the rest come from reading the code.

---

## 0. Executive summary

The repo is a research fork of torch-points3d (PointGroup with three heads on a
MinkowskiEngine sparse-conv backbone) wrapped in an Oracle-cloud batch pipeline. It works
for its authors' environment but is not a reusable tool yet. Five things block us today:

1. **The shipped checkpoint is a Git LFS pointer, not a model.** `model_file/PointGroup-PAPER.pt`
   is 134 bytes; the real file is 665 MB and has not been fetched (verified). Every Dockerfile
   `COPY`s the working tree, so a locally built image also has no model. Fix: `git lfs pull`
   before building, or use the pre-built Docker Hub image.
2. **The colleague's docker file is not a build recipe for this repo.** It is the
   `setup/segment_any_tree/` Dockerfile from a separate, non-public "tree_seg_benchmark"
   project: it `COPY`s from `../setup/…` and installs `./tree_seg_benchmark`, has no
   ENTRYPOINT, and clones a fork (`josafatburmeister/SegmentAnyTree@fix/pipeline`) instead of
   this checkout. It will not build from this directory as-is.
3. **No NVIDIA GPU on this machine (Apple Silicon, no `nvidia-smi`).** MinkowskiEngine, the
   losses and eval NMS all call `.cuda()` unconditionally. Inference and training need a
   Linux box with an NVIDIA GPU. Compiled kernels target sm 6.0–8.6, so Ampere/Ada work,
   Hopper and newer do not with the CUDA 11.1 image.
4. **Paths are hard-coded to the authors' machines.** `/home/nibio/mutable-outside-world`
   (`run_inference.sh:35`, `conf/eval.yaml:13`) and `/home/datascience` (`run_oracle_pipeline.sh`,
   `conf/data/panoptic/treeins_rad8.yaml:8`, and the dataroot baked into the checkpoint).
5. **The README training command references config files that do not exist**
   (`data=panoptic/treeins`, `models=panoptic/area4_ablation_3heads`; verified). The bare
   `conf/config.yaml` defaults also point at missing files. The only working set of overrides
   is `model_file/.hydra/overrides.yaml`.

Recommended path: use the pre-built image `maciekwielgosz/segment-any-tree:latest` on a GPU
host for a first inference run, then fork this repo and fix the items in section 6 before
training.

---

## 1. Architecture at a glance

| Layer | What it is | Key files |
|---|---|---|
| Orchestration | Bash scripts moving files between "bucket" folders in 10-file chunks | `run_oracle_pipeline.sh`, `run_inference.sh`, `run_batch_inference.sh` |
| Pre/post-processing | LAS/LAZ/PLY conversion, per-file UTM to local shift, merging predictions back | `nibio_inference/*.py` |
| Model runtime | Hydra + torch-points3d Trainer, MinkowskiEngine backbone | `eval.py`, `train.py`, `torch_points3d/trainer.py` |
| Dataset | `treeinsfused` dataset, 0.2 m grid subsampling, 8 m radius cylinders | `torch_points3d/datasets/panoptic/treeins.py`, `…/segmentation/treeins.py` |
| Model | PointGroup with semantic, offset and embedding heads, mean-shift clustering, score net | `torch_points3d/models/panoptic/PointGroup3heads.py` |
| Block merging + output | Merges per-cylinder instances into whole-plot ids, writes result PLYs | `torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py` |
| Environment | Three Dockerfiles, one from a different project | `Dockerfile`, `Dockerfile_cuda:11.8.0`, `segment_any_tree_gpu.docker` |

Inference data flow (`run_inference.sh`):

```
input_data/*.{las,laz,ply}
  -> fix_naming_of_input_files.py         ('-' and ' ' to '_')
  -> pipeline_utm2local_parallel.py       (subtract per-file min XYZ, save *_min_values.json, write binary PLY)
  -> modify_eval.py + clear_cache.py       (inject data.fold list, delete stale processed cache)
  -> eval.py                               (cylinders, model, block merge; writes result_N.ply etc.)
  -> rename_result_files_*.py              (result_N.ply -> per-input names)
  -> merge_pt_ss_is_in_folders.py          (outer join on "x_y_z" string key, add min XYZ back, write LAZ)
final_results/<name>_out.laz
```

---

## 2. Inference: what goes in, what comes out

**Input**
- A flat directory of `.las`, `.laz` or `.ply`; extension check is case-sensitive and
  subfolders are skipped (`nibio_inference/pipeline_utm2local_parallel.py:16`).
- Only XYZ is required. `intensity`, `classification`, RGB etc. are passed through untouched
  but ignored by the model. PLY inputs need lowercase `x,y,z` at the utm2local step (`:21`).
- Ground truth (`semantic_seg`, `treeID`) is optional at test time; metrics are then meaningless
  but predictions are produced.
- Ground should be at z ≈ 0 relative to the plot. Absolute z is fed as a feature
  (`conf/data/panoptic/treeins_rad8.yaml:58-72`, `XYZFeature add_z` before `Center`), so the
  per-file min shift is doing real work. Very steep terrain or tall plots will behave differently
  from training data.
- Filenames: avoid extra dots (merge matching uses `split('.')[0]`,
  `nibio_inference/merge_pt_ss_is_in_folders.py:36-48`) and avoid a leading `NNN_` prefix
  (`run_inference.sh:80-93` strips it, and `mv -n` silently drops collisions).

**Output** (`final_results/<name>_out.laz`, `nibio_inference/pandas_to_las.py:68-126`)
- LAS 1.4, point format 6, scale 0.001, offset = per-file min. Coordinates are numerically
  back in the input UTM frame.
- **CRS is lost.** The header is built from scratch; no VLRs or GeoKeys are copied. Re-attach
  the CRS afterwards with PDAL or las2las.
- Extra dimensions: `PredSemantic` uint8 (0 = non-tree, 1 = tree), `PredInstance` uint16
  (tree ids from 1, 0 = no instance; caps at 65535 trees), a filler `gt_semantic_segmentation`.
- Points without an instance (non-tree, more than 1 m from any voxel, or in clusters of fewer
  than 10 points) come out with `PredInstance = 0`
  (`torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py:641-664`).
- Precision: everything is cast to float32 before the join
  (`nibio_inference/pandas_to_ply.py:23`). Local coordinates survive at ~mm level for plot-sized
  files; `gps_time` and other float64 attributes are damaged.

**Runtime facts that matter**
- The dataset and transform config used at inference comes from the `run_config` embedded in the
  checkpoint, not from `conf/eval.yaml` (`torch_points3d/trainer.py:89-92`,
  `torch_points3d/metrics/model_checkpoint.py:173-174`, verified). Only `data.fold` is injected.
  So `dataroot` at inference is `/home/datascience/tmp_out_folder/utm2local` from
  `model_file/.hydra/overrides.yaml:7`, and the processed cache goes under that path.
  `nibio_inference/clear_cache.py` exists purely to work around this. `forward_scripts/forward.py:73`
  shows the clean way to override it.
- With `resume=True` the 665 MB checkpoint is copied into a new run dir under
  `model_file/eval/<timestamp>/` on every eval (`model_checkpoint.py:76-80`).
- `weight_name: latest` loads the last epoch's weights, not the best validation epoch
  (`model_checkpoint.py:141-155`).
- Clustering only runs when the checkpoint's epoch count exceeds `prepare_epoch` (30)
  (`PointGroup3heads.py:138-139`, verified). A model trained for fewer epochs returns zero
  instances with no warning. This matters for our own training later.
- The tracker writes unconditional debug dumps of every proposal in every cylinder into
  `viz_for_test_all_proposals/`, `viz_for_test_valid_proposals/` and `viz/`
  (tracker `:300-415`). This is a large disk and time cost with no off switch.
- Cache files are keyed by basename only (`torch_points3d/datasets/segmentation/treeins.py:287,302,432`).
  Editing a file without renaming it reuses old data.

---

## 3. Training: what the code expects

- Layout: `<dataroot>/treeinsfused/raw/<region>/*.ply` (recursive scan,
  `torch_points3d/datasets/segmentation/treeins.py:249-255`). Split is decided purely by filename
  suffix: `*val.ply` → val, `*test.ply` → test, everything else → train (`:342-347`).
  `fold: []` means all test files.
- Labels in the PLY: `semantic_seg` with 0 = unclassified (ignored), 1 = non-tree, 2 = tree;
  `treeID` with 0 for all non-tree points. Internally tree → class 1, instance = treeID + 1.
- `sample_data_conversion.py` shows the LAS → PLY conversion the authors used: classification
  1/2 → non-tree, 4/5/6 → tree, 3 → unclassified. It also documents that LAS offsets are
  dropped (`sample_data_conversion.py:44-46`), the same "local coordinates" convention as inference.
- PLYs must be binary; ASCII is rejected (`torch_points3d/modules/KPConv/plyutils.py:155-156`).
- Hard limit of 80 instances per 8 m cylinder (`torch_points3d/datasets/panoptic/treeins.py:519`)
  raises `ValueError` in dense stands. Training cylinders must contain at least one tree point.
- Cache: `processed_<grid>[_<regions>]/{train,val,test,trainval}.pt` plus `preprocessed.pt`.
  Adding files after `preprocessed.pt` exists does nothing until you delete the cache
  (`segmentation/treeins.py:327,375`).
- `conf/training/treeins.yaml`: `epochs: 2` (needs more than 30 for clustering to ever engage),
  `batch_size: 4`, `num_workers: 0`, and **W&B logging on with a hard-coded entity**
  (`maciej-wielgosz-nibio`, `public: True`, `:33-35`, verified). Training will fail at startup
  without a W&B login and access to that entity. Set `training.wandb.log=False` or change entity.
- `epochs=100 batch_size=6` in `model_file/.hydra/overrides.yaml` are top-level keys; the Trainer
  reads `training.epochs` and `training.batch_size`, so pass them with that prefix.

---

## 4. Environment and Docker

Three files, three different stacks:

| File | Python | torch | CUDA | Notes |
|---|---|---|---|---|
| `Dockerfile` | 3.8 (apt) | 1.9.0+cu111 | 11.1.1 | pip pinned 21.2.4 so `--install-option` still works. ENTRYPOINT is the Oracle pipeline. |
| `Dockerfile_cuda:11.8.0` | 3.8 | 2.3.1+cu118 | 11.8 | README says "not well tested". Only file that can target sm_89/9.0. |
| `segment_any_tree_gpu.docker` | 3.9.21 venv + 3.12.9 venv | 1.10.1+cu111 (venv) and 2.5.1+cu124 (utils venv) | 11.1.1 | From the tree_seg_benchmark project. Not buildable here. |

Findings (fork/Docker agent, partly verified live):
- `pip --install-option` was removed in pip 23.1. The repo Dockerfiles pin pip 21.2.4 and are fine.
  The colleague's 3.9.21 venv gets pip 23.0.1 from ensurepip and still works, but breaks on any
  `pip install -U pip`. Replacement is `python setup.py install --blas=openblas --force_cuda` in a
  clone of MinkowskiEngine.
- MinkowskiEngine is installed from unpinned `master` (last commit 2023-03, effectively
  unmaintained). Builds against torch 1.9/1.10 + CUDA 11.1 are known good; CUDA 12 builds fail.
- `TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6"` without `+PTX`. Ada (8.9) runs sm_86 cubins, Hopper
  and Blackwell do not. CUDA 11.1's nvcc cannot target 8.9/9.0 at all; use the 11.8 file for
  newer cards and add `8.9;9.0+PTX`.
- No `git lfs pull` anywhere, so every image built from a fresh clone ships the LFS pointer.
- `torch-points-kernels==0.7.0` builds against `numpy<1.20` but the final image installs
  numpy 1.24.4; possible ABI mismatch, not verified.
- `nvidia/cuda:11.1.1-*` base tags still exist on Docker Hub (verified) but are EOL; pin by digest.
- The colleague's `apt-get upgrade -y` and unpinned `git+https://…pointtree` installs make that
  image non-reproducible.

**The fork the colleague uses** (`josafatburmeister/SegmentAnyTree@fix/pipeline`, verified with
`git fetch`, no remote added): 2 commits ahead, 10 behind upstream `main`, no conflicts.
- `24f4445` "fix mean shift": replaces a `multiprocessing.Pool` per batch element with a list
  comprehension and `MeanShift(n_jobs=-1)` in `torch_points3d/utils/meanshift_cluster.py:96-109`.
  A real robustness fix (spawning a pool inside a torch process hangs or OOMs). Worth cherry-picking.
  The sibling `cluster()` at `:51-56` still uses the Pool.
- `5edf965` "remove file renaming": comments out the prefix-stripping loop in `run_inference.sh`.
  Naming preference, not a fix.
- The fork lacks upstream's `a845831` (block-merge id collision fix that caused spurious tree
  fusing on sparse plots). Base on upstream `main` and cherry-pick `24f4445`.

---

## 5. Bugs and traps in the pipeline scripts

Most important first.

- **Merge join can blow up.** `merge_pt_ss_is.py:62-63` joins prediction and input frames on a
  string key `"x_y_z"` built from float32 coordinates using dask outer joins. Duplicate XYZ points
  (multi-return TLS, densified MLS) produce a cartesian product, inflating point count and memory.
  Memory is roughly three full copies plus one Python string per point.
- **Silent failures.** `rename_result_files_instance.py:38-39` and
  `rename_result_files_segmentation.py:35-36` catch all exceptions and exit 0;
  `merge_pt_ss_is_in_folders.py:51-53` skips unmatched files without error. You can end up with an
  empty `final_results` and a green exit code.
- `clear_cache.py:14-23`: if `overrides.yaml` has no `data.dataroot=` line, `path` is undefined
  (NameError). Stale `processed_0.2_test/*.pt` are reused by fold index
  (`segmentation/treeins.py:431-436`), so a failed cache clear returns results for the wrong file.
- `pandas_to_ply.py:24` builds one Python tuple per point; slow and memory-hungry above ~50 M points.
  The tracker also runs a full-cloud CPU knn with `num_workers=48` (`tracker:625`).
- `pandas_to_las.py:126,136` do `.replace('.las', '.laz')` on the whole path. `scan_angle_rank`
  (int8) is cast to uint16 (`:46,83-84`), corrupting negative angles. `PredSemantic.astype('uint8')`
  (`:122`) throws on any NaN.
- `pipeline_utm2local_parallel.py:51` runs 4 workers via joblib; four full clouds are in memory at
  once.
- Off-path or dead: `bring_back_to_utm_coordinates.py`, `pipeline_local2utm.py` (uses lowercase
  `x` on LAS frames, KeyError), serial `pipeline_utm2local.py` (writes a differently named JSON the
  merge will not find), `main_parallel_join` in `merge_pt_ss_is.py`, laspy-1 column clipping at
  `merge_pt_ss_is.py:133-148`.

Bugs and dead code in the model side:
- 12 `panoptic_tracker_*` files; only `panoptic_tracker_pointgroup_treeins.py` is used.
  `*_backup.py`, `*_mine.py`, `*_italy.py`, `*_old.py`, `transforms_backup.py`,
  `treeins_set1.py` (both) are dead.
- `segmentation/treeins.py:117-149` `to_eval_ply`/`to_ins_ply` are broken (static `PlyData.write`,
  invalid `"u16"` dtype) but shadowed by the panoptic subclass. `:56` `OBJECT_LABEL["unclassified"]`
  would KeyError if reached.
- Full `Data` lists are printed during preprocessing (`segmentation/treeins.py:354-363,400-407`).
- No tests anywhere in the repo (verified: zero `test_*.py`).
- Two licence files: `LICENSE` (MIT, NIBIO 2025) and `LICENSE.md` (torch-points3d BSD-style,
  Principia Labs 2020). Fine, but worth knowing which applies to which code.

---

## 6. Recommended plan

**Status 2026-09-22:** items A.1 (LFS pull) and all of B are done on branch `fix/pipeline-usability`
(uncommitted apart from the cherry-picked mean-shift fix). Tests for the pre/post-processing scripts
live in `tests/`. The torch-side changes (tracker `orig_index`, `save_viz`, dataroot override) could not be
executed on this machine (no CUDA) and need one real run on a GPU host.

**A. First inference run, no code changes (needs a Linux host with an NVIDIA GPU up to Ampere/Ada):**
1. `docker pull maciekwielgosz/segment-any-tree:latest` (weights included).
2. Mount input/output dirs as in README "Quick start". Inputs: LAS/LAZ, one plot per file,
   simple names, no leading digits, no extra dots.
3. Re-attach the CRS to `final_results/*_out.laz` with PDAL or las2las.
4. Expect `viz*/` debug dumps and a `model_file/eval/<ts>/` copy of the checkpoint per run.

**B. Before relying on it (fork this repo, small fixes):**
1. `git lfs pull`; add `git lfs install && git lfs pull` (or a download step) to the Dockerfile.
2. Cherry-pick `24f4445` from the fork (mean-shift Pool removal).
3. Make `SCRIPT_DIR` and `checkpoint_dir` configurable (env var or argument) instead of
   `/home/nibio/mutable-outside-world`; override `dataroot` on the loaded checkpoint config the way
   `forward_scripts/forward.py:73` does, and drop `clear_cache.py`.
4. Gate the `viz*` dumps behind a config flag.
5. Make the rename and merge steps fail loudly; replace the string-key join with an index-based
   join (the utm2local step controls point order, so an `original_index` column would do).
6. Copy the input LAS header/VLRs into the output so the CRS survives.
7. Fix the README training command and the `conf/config.yaml` defaults.

**C. Training on our own data:**
1. Convert to binary PLY with `x,y,z,semantic_seg,treeID` (0/1/2 semantic, treeID 0 for non-tree),
   local coordinates with ground near z = 0, files named `*_train.ply`, `*_val.ply`, `*_test.ply`
   under `<dataroot>/treeinsfused/raw/<region>/`.
2. Run with the overrides from `model_file/.hydra/overrides.yaml`, plus
   `training.wandb.log=False training.epochs=<>30> training.batch_size=<fits GPU>`.
3. Watch the 80-instances-per-cylinder limit in dense stands; either lower the cylinder radius or
   raise `NUM_MAX_OBJECTS`.
4. Decide whether to fine-tune from `PointGroup-PAPER.pt` (set `training.checkpoint_dir`) or train
   from scratch; the paper model saw 0.2 m voxels and 8 m cylinders, so keep those unless the data
   is very different.

---

## 7. Repo state notes

- `segment_any_tree_gpu.docker` is currently **staged** (`git add`ed) in this checkout. Decide
  whether it belongs in this repo; as it stands it cannot build here.
- `.idea/` and `CLAUDE.md` are untracked. `CLAUDE.md` contains only tool-routing rules for the
  assistant, nothing about the project.
