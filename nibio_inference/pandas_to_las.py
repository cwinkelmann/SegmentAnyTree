import os

import laspy
import numpy as np
import pandas as pd
from laspy.vlrs.known import (
    GeoAsciiParamsVlr,
    GeoDoubleParamsVlr,
    GeoKeyDirectoryVlr,
    WktCoordinateSystemVlr,
)

# works with laspy 2.x

# LAS 1.4 point format 6 dimensions and the dtype we cast DataFrame columns to
STANDARD_COLUMN_DTYPES = {
    "intensity": "uint16",
    "return_number": "uint8",
    "number_of_returns": "uint8",
    "synthetic": "uint8",
    "key_point": "uint8",
    "withheld": "uint8",
    "overlap": "uint8",
    "scanner_channel": "uint8",
    "scan_direction_flag": "uint8",
    "edge_of_flight_line": "uint8",
    "classification": "uint8",
    "user_data": "uint8",
    "scan_angle": "int16",  # units of 0.006 degrees
    "point_source_id": "uint16",
    "gps_time": "float64",
    "red": "uint16",
    "green": "uint16",
    "blue": "uint16",
}

EXTENDED_COLUMN_DTYPES = {
    "Amplitude": "float64",
    "Pulse_width": "float64",
    "Reflectance": "float64",
    "Deviation": "int32",
    "PredSemantic": "uint8",
    "PredInstance": "uint32",
}

CRS_VLR_TYPES = (GeoKeyDirectoryVlr, GeoAsciiParamsVlr, GeoDoubleParamsVlr, WktCoordinateSystemVlr)


def laz_backend_available():
    try:
        return len(laspy.LazBackend.detect_available()) > 0
    except Exception:
        return False


def copy_crs_vlrs(source_las_path, header):
    """Copy the coordinate reference system VLRs of an existing LAS/LAZ into header."""
    with laspy.open(source_las_path) as src:
        crs_vlrs = [vlr for vlr in src.header.vlrs if isinstance(vlr, CRS_VLR_TYPES)]
    for vlr in crs_vlrs:
        header.vlrs.append(vlr)
        if isinstance(vlr, WktCoordinateSystemVlr):
            header.global_encoding.wkt = True
    return len(crs_vlrs)


def pandas_to_las(csv, csv_file_provided=False, output_file_path=None, do_compress=False, verbose=False,
                  source_las_path=None):
    """Write a DataFrame as a LAS 1.4 / point format 6 file (LAZ if do_compress).

    Columns x/y/z (or X/Y/Z) are the real-world coordinates. Columns whose name
    matches a point-format-6 dimension are written as that dimension; every other
    column becomes an extra-bytes dimension with its DataFrame dtype (or the dtype
    from EXTENDED_COLUMN_DTYPES). If source_las_path is given, its CRS VLRs are
    copied so the output stays georeferenced.

    Returns the path that was written.
    """
    if output_file_path is None:
        raise ValueError("output_file_path is required")

    df = pd.read_csv(csv, sep=",") if csv_file_provided else csv

    df = df.rename(columns={"x": "X", "y": "Y", "z": "Z"})

    # scan_angle_rank (point formats 0-5, whole degrees, int8) -> scan_angle (0.006 degree units, int16)
    if "scan_angle_rank" in df.columns:
        df = df.assign(scan_angle=np.round(df["scan_angle_rank"].astype(np.float64) / 0.006).astype(np.int16))
        df = df.drop(columns=["scan_angle_rank"])

    scale = [0.001, 0.001, 0.001]
    offset = [float(df["X"].min()), float(df["Y"].min()), float(df["Z"].min())]

    las_header = laspy.LasHeader(point_format=6, version="1.4")
    las_header.scale = scale
    las_header.offset = offset
    if source_las_path is not None:
        copied = copy_crs_vlrs(source_las_path, las_header)
        if verbose:
            print(f"Copied {copied} CRS VLR(s) from {source_las_path}")

    standard_columns = list(las_header.point_format.dimension_names)
    columns_which_match = [c for c in standard_columns if c in df.columns and c not in ("X", "Y", "Z")]
    extra_columns = [c for c in df.columns if c not in standard_columns]

    extra_dtypes = {}
    for column in extra_columns:
        extra_dtypes[column] = np.dtype(EXTENDED_COLUMN_DTYPES.get(column, df[column].dtype))
        las_header.add_extra_dim(laspy.ExtraBytesParams(name=column, type=extra_dtypes[column]))

    las_file = laspy.LasData(las_header)
    las_file.x = df["X"].to_numpy(dtype=np.float64)
    las_file.y = df["Y"].to_numpy(dtype=np.float64)
    las_file.z = df["Z"].to_numpy(dtype=np.float64)

    for column in columns_which_match:
        las_file[column] = df[column].to_numpy().astype(STANDARD_COLUMN_DTYPES[column])

    for column in extra_columns:
        las_file[column] = df[column].to_numpy().astype(extra_dtypes[column])

    root, ext = os.path.splitext(output_file_path)
    if do_compress and not laz_backend_available():
        print(f"No LAZ backend (lazrs/laszip) installed; writing uncompressed {root}.las instead")
        do_compress = False
    output_file_path = root + (".laz" if do_compress else ".las")
    las_file.write(output_file_path, do_compress=do_compress)

    if verbose:
        print("File saved as: {}".format(output_file_path))
    return output_file_path
