import numpy as np
from plyfile import PlyData, PlyElement

from scripts.prepare_forinstance_v2 import convert_file, region_of


def write_ply(path, n=50, semantic=None, tree_id=None, offset=(100.0, 200.0, 335.0), with_labels=True):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 10, size=(n, 3)) + np.array(offset)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if with_labels:
        fields += [("semantic_seg", "i4"), ("treeID", "i4")]
    arr = np.empty(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if with_labels:
        arr["semantic_seg"] = semantic
        arr["treeID"] = tree_id
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    return xyz


def test_convert_remaps_labels_and_shifts_to_local(tmp_path):
    semantic = np.array([1] * 20 + [2] * 10 + [3] * 15 + [0] * 5)  # ground, wood, leaf, unlabeled
    tree_id = np.array([0] * 20 + [7] * 10 + [7] * 10 + [8] * 5 + [0] * 5)
    xyz = write_ply(tmp_path / "CULS_CULS_plot_1_annotated_train.ply", semantic=semantic, tree_id=tree_id)
    out_dir = tmp_path / "raw"

    out = convert_file(str(tmp_path / "CULS_CULS_plot_1_annotated_train.ply"), str(out_dir))

    assert out == str(out_dir / "CULS" / "CULS_CULS_plot_1_annotated_train.ply")
    v = PlyData.read(out)["vertex"].data
    assert not PlyData.read(out).text
    assert v.dtype.names == ("x", "y", "z", "semantic_seg", "treeID")
    np.testing.assert_allclose(v["x"], xyz[:, 0] - xyz[:, 0].min(), atol=1e-4)
    np.testing.assert_allclose(v["z"], xyz[:, 2] - xyz[:, 2].min(), atol=1e-4)
    expected_sem = np.array([1] * 20 + [2] * 25 + [0] * 5)
    np.testing.assert_array_equal(v["semantic_seg"], expected_sem)
    np.testing.assert_array_equal(v["treeID"], tree_id)


def test_convert_skips_unlabeled_files(tmp_path):
    write_ply(tmp_path / "r12_tegel_E381300_N5828300_100m.ply", with_labels=False)
    assert convert_file(str(tmp_path / "r12_tegel_E381300_N5828300_100m.ply"), str(tmp_path / "raw")) is None
    assert not (tmp_path / "raw").exists()


def test_region_of():
    assert region_of("NIBIO_MLS_MLS_Plot_56_panoptic_train.ply") == "NIBIO_MLS"
    assert region_of("NIBIO2_NIBIO2_plot12_annotated_train.ply") == "NIBIO2"
    assert region_of("BlueCat_RN_merged_trees_val.ply") == "BlueCat"
    assert region_of("Yuchen_2023_dls_merged_230209_panoptic_test.ply") == "Yuchen"
