"""Tests for the pre/post-processing scripts in nibio_inference.

Run from the repo root:  pytest tests/
They only need laspy, pandas, numpy, plyfile and pyyaml (no torch, no GPU).
"""
import json
import os

import laspy
import numpy as np
import pandas as pd
import pytest
import yaml
from plyfile import PlyData

from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.merge_pt_ss_is import MergePtSsIs
from nibio_inference.merge_pt_ss_is_in_folders import MergePtSsIsInFolders
from nibio_inference.modify_eval import modify_yaml
from nibio_inference.pandas_to_las import pandas_to_las
from nibio_inference.pandas_to_ply import pandas_to_ply
from nibio_inference.pipeline_utm2local_parallel import process_file
from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.rename_result_files_instance import rename_files as rename_instance
from nibio_inference.rename_result_files_segmentation import rename_files as rename_semantic

UTM_ORIGIN = (599_000.0, 6_640_000.0, 120.0)  # a plausible UTM 33N location, far from 0


def make_utm_las(path, n=200, point_format=3, with_crs=True, seed=0):
    """Write a small georeferenced LAS/LAZ with a duplicate point and a WKT CRS."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 30, size=(n, 3)) + np.array(UTM_ORIGIN)
    xyz[1] = xyz[0]  # one exact duplicate point
    header = laspy.LasHeader(point_format=point_format, version="1.4" if point_format >= 6 else "1.2")
    header.scale = [0.001, 0.001, 0.001]
    header.offset = list(UTM_ORIGIN)
    if with_crs:
        wkt = ('PROJCS["ETRS89 / UTM zone 33N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",'
               'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
               'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],'
               'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],'
               'UNIT["metre",1],AUTHORITY["EPSG","25833"]]')
        header.vlrs.append(laspy.vlrs.known.WktCoordinateSystemVlr(wkt))
        if point_format >= 6:
            header.global_encoding.wkt = True
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.intensity = rng.integers(0, 4000, n).astype(np.uint16)
    las.classification = rng.integers(1, 6, n).astype(np.uint8)
    las.gps_time = 300_000_000.0 + np.arange(n) * 1e-4  # needs float64 to survive
    if point_format < 6:
        las.scan_angle_rank = rng.integers(-30, 30, n).astype(np.int8)
    las.write(path)
    return xyz


def write_prediction_plys(out_dir, index, n_points, local_xyz, seed=1):
    """Mimic what the tracker writes: semantic_result_<i>.ply for all points,
    result_<i>.ply only for points that got an instance, both with orig_index."""
    from plyfile import PlyElement

    rng = np.random.default_rng(seed)
    sem = rng.integers(0, 2, n_points).astype(np.int16)
    arr = np.empty(n_points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("preds", "i2"), ("gt", "i2"),
                                    ("orig_index", "u4")])
    arr["x"], arr["y"], arr["z"] = local_xyz[:, 0], local_xyz[:, 1], local_xyz[:, 2]
    arr["preds"], arr["gt"], arr["orig_index"] = sem, -1, np.arange(n_points)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(os.path.join(out_dir, f"semantic_result_{index}.ply"))

    things = np.flatnonzero(sem == 1)
    ins = rng.integers(0, 3, len(things)).astype(np.int16)  # 0-based instance ids like the tracker
    arr = np.empty(len(things), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("preds", "i2"), ("orig_index", "u4")])
    arr["x"], arr["y"], arr["z"] = local_xyz[things, 0], local_xyz[things, 1], local_xyz[things, 2]
    arr["preds"], arr["orig_index"] = ins, things
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(os.path.join(out_dir, f"result_{index}.ply"))
    return sem, things, ins


# --------------------------------------------------------------------------- PLY round trip

def test_ply_roundtrip_preserves_dtypes_and_order(tmp_path):
    df = pd.DataFrame({
        "x": np.array([1.5, 2.5, 0.5], dtype=np.float64),
        "gps_time": np.array([300_000_000.0001, 300_000_000.0002, 300_000_000.0003]),
        "classification": np.array([2, 5, 1], dtype=np.uint8),
        "flag": np.array([True, False, True]),
    })
    out = tmp_path / "rt.ply"
    pandas_to_ply(df, output_file_path=str(out))
    back = ply_to_pandas(str(out))

    assert list(back.columns) == list(df.columns)
    assert back["x"].dtype == np.float64
    assert back["gps_time"].dtype == np.float64
    assert back["classification"].dtype == np.uint8
    np.testing.assert_array_equal(back["x"].to_numpy(), df["x"].to_numpy())  # no sorting
    np.testing.assert_array_equal(back["gps_time"].to_numpy(), df["gps_time"].to_numpy())  # no float32 loss
    assert PlyData.read(str(out)).elements[0].name == "vertex"
    assert not PlyData.read(str(out)).text  # binary, the dataset reader rejects ASCII


# --------------------------------------------------------------------------- utm2local

@pytest.mark.parametrize("ext", ["las", "laz"])
def test_utm2local_writes_local_ply_and_min_values(tmp_path, ext):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir(); out_dir.mkdir()
    xyz = make_utm_las(str(in_dir / f"plot.{ext}"))

    process_file(f"plot.{ext}", str(in_dir), str(out_dir))

    ply = ply_to_pandas(str(out_dir / "plot_out.ply"))
    mins = json.load(open(out_dir / "plot_out_min_values.json"))
    assert mins == pytest.approx(list(xyz.min(axis=0)), abs=1e-3)
    assert ply["x"].min() == pytest.approx(0.0, abs=1e-6)
    assert len(ply) == len(xyz)
    np.testing.assert_allclose(ply["x"].to_numpy() + mins[0], xyz[:, 0], atol=1e-3)  # order kept


def test_utm2local_ignores_non_point_cloud_files(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir(); out_dir.mkdir()
    (in_dir / "notes.txt").write_text("hello")
    process_file("notes.txt", str(in_dir), str(out_dir))
    assert list(out_dir.iterdir()) == []


# --------------------------------------------------------------------------- merge

def _run_pipeline_to_merge(tmp_path, point_format=3, with_crs=True):
    in_dir, work = tmp_path / "input_data", tmp_path / "work"
    in_dir.mkdir(); work.mkdir()
    utm2local = work / "utm2local"; utm2local.mkdir()
    xyz = make_utm_las(str(in_dir / "plot_a.las"), point_format=point_format, with_crs=with_crs)
    process_file("plot_a.las", str(in_dir), str(utm2local))
    local = ply_to_pandas(str(utm2local / "plot_a_out.ply"))[["x", "y", "z"]].to_numpy()
    sem, things, ins = write_prediction_plys(str(work), 0, len(xyz), local)
    return in_dir, work, utm2local, xyz, sem, things, ins


def test_merge_uses_orig_index_and_restores_utm(tmp_path):
    in_dir, work, utm2local, xyz, sem, things, ins = _run_pipeline_to_merge(tmp_path)

    merged = MergePtSsIs(
        point_cloud=str(utm2local / "plot_a_out.ply"),
        semantic_segmentation=str(work / "semantic_result_0.ply"),
        instance_segmentation=str(work / "result_0.ply"),
        output_file_path=None,
    ).run()

    assert len(merged) == len(xyz)  # the duplicate point did not multiply
    np.testing.assert_allclose(merged[["x", "y", "z"]].to_numpy(), xyz, atol=1e-3)
    np.testing.assert_array_equal(merged["PredSemantic"].to_numpy(), sem)
    expected_ins = np.zeros(len(xyz), dtype=np.int64)
    expected_ins[things] = ins + 1  # ids start at 1, 0 = no instance
    np.testing.assert_array_equal(merged["PredInstance"].to_numpy(), expected_ins)
    assert "intensity" in merged.columns  # pass-through attributes survive


def test_merge_refuses_prediction_files_without_orig_index(tmp_path):
    in_dir, work, utm2local, *_ = _run_pipeline_to_merge(tmp_path)
    # strip orig_index from the instance file
    from numpy.lib import recfunctions as rfn
    from plyfile import PlyElement
    ply = PlyData.read(str(work / "result_0.ply"))
    data = rfn.repack_fields(ply["vertex"].data[["x", "y", "z", "preds"]])
    PlyData([PlyElement.describe(data, "vertex")]).write(str(work / "result_0.ply"))

    with pytest.raises(ValueError, match="orig_index"):
        MergePtSsIs(str(utm2local / "plot_a_out.ply"), str(work / "semantic_result_0.ply"),
                    str(work / "result_0.ply"), None).run()


@pytest.mark.parametrize("point_format", [3, 6])
def test_end_to_end_laz_keeps_crs_and_attributes(tmp_path, point_format):
    in_dir, work, utm2local, xyz, sem, things, ins = _run_pipeline_to_merge(tmp_path, point_format=point_format)
    final = work / "final_results"
    # the tracker output was renamed by rename_result_files_* in the real pipeline
    os.rename(work / "semantic_result_0.ply", work / "semantic_segmentation_plot_a_out.ply")
    os.rename(work / "result_0.ply", work / "instance_segmentation_plot_a_out.ply")

    MergePtSsIsInFolders(str(utm2local), str(work), str(final), original_data_folder_path=str(in_dir))()

    out_path = final / "plot_a_out.laz"
    assert out_path.exists()
    las = laspy.read(str(out_path))
    assert len(las.points) == len(xyz)
    np.testing.assert_allclose(np.c_[las.x, las.y, las.z], xyz, atol=1e-3)
    assert las.header.parse_crs() is not None
    assert "25833" in las.header.parse_crs().to_wkt()
    np.testing.assert_array_equal(las.PredSemantic, sem)
    assert las.PredInstance.max() == ins.max() + 1
    src = laspy.read(str(in_dir / "plot_a.las"))
    np.testing.assert_array_equal(las.gps_time, src.gps_time)  # float64 survived
    np.testing.assert_array_equal(las.intensity, src.intensity)
    np.testing.assert_array_equal(las.classification, src.classification)
    if point_format < 6:
        # scan_angle_rank (degrees, int8) becomes scan_angle (0.006 degree units, int16)
        np.testing.assert_allclose(las.scan_angle * 0.006, src.scan_angle_rank, atol=0.006)


def test_falls_back_to_las_when_no_laz_backend(tmp_path, monkeypatch):
    in_dir, work, utm2local, xyz, sem, things, ins = _run_pipeline_to_merge(tmp_path)
    os.rename(work / "semantic_result_0.ply", work / "semantic_segmentation_plot_a_out.ply")
    os.rename(work / "result_0.ply", work / "instance_segmentation_plot_a_out.ply")
    monkeypatch.setattr(laspy.LazBackend, "detect_available", staticmethod(lambda: ()))

    written = MergePtSsIsInFolders(str(utm2local), str(work), str(tmp_path / "final"))()

    assert written == [str(tmp_path / "final" / "plot_a_out.las")]
    assert len(laspy.read(written[0]).points) == len(xyz)


def test_merge_in_folders_fails_loudly_when_predictions_are_missing(tmp_path):
    in_dir, work, utm2local, *_ = _run_pipeline_to_merge(tmp_path)
    # no renamed prediction files at all
    with pytest.raises(FileNotFoundError, match="plot_a_out"):
        MergePtSsIsInFolders(str(utm2local), str(work), str(tmp_path / "final"))()


# --------------------------------------------------------------------------- rename + eval config

def _eval_yaml(tmp_path, fold):
    cfg = {"checkpoint_dir": "/old", "data": {"fold": [], "dataroot": ""},
           "hydra": {"run": {"dir": "/old"}}, "tracker_options": {"save_viz": False}}
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.safe_dump(cfg))
    modify_yaml(str(p), fold, str(tmp_path), checkpoint_dir="/models", dataroot=str(tmp_path / "utm2local"))
    return p


def test_modify_eval_sets_all_paths(tmp_path):
    p = _eval_yaml(tmp_path, ["/x/a_out.ply", "/x/b_out.ply"])
    cfg = yaml.safe_load(p.read_text())
    assert cfg["data"]["fold"] == ["/x/a_out.ply", "/x/b_out.ply"]
    assert cfg["data"]["dataroot"] == str(tmp_path / "utm2local")
    assert cfg["checkpoint_dir"] == "/models"
    assert cfg["hydra"]["run"]["dir"] == str(tmp_path)
    assert cfg["tracker_options"]["save_viz"] is False


def test_rename_result_files_by_fold_order(tmp_path):
    p = _eval_yaml(tmp_path, ["/x/a_out.ply", "/x/b_out.ply"])
    for i in range(2):
        (tmp_path / f"result_{i}.ply").write_bytes(b"")
        (tmp_path / f"semantic_result_{i}.ply").write_bytes(b"")
    rename_instance(str(p), str(tmp_path))
    rename_semantic(str(p), str(tmp_path))
    assert (tmp_path / "instance_segmentation_a_out.ply").exists()
    assert (tmp_path / "instance_segmentation_b_out.ply").exists()
    assert (tmp_path / "semantic_segmentation_a_out.ply").exists()
    assert (tmp_path / "semantic_segmentation_b_out.ply").exists()


def test_rename_result_files_raises_when_inference_output_is_missing(tmp_path):
    p = _eval_yaml(tmp_path, ["/x/a_out.ply"])
    with pytest.raises(FileNotFoundError, match="result_0.ply"):
        rename_instance(str(p), str(tmp_path))
