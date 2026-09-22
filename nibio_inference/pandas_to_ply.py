import numpy as np
import pandas as pd
from plyfile import PlyElement, PlyData

# PLY only knows these scalar types. Everything else is widened to the nearest one.
_PLY_DTYPES = {"i1", "u1", "i2", "u2", "i4", "u4", "f4", "f8"}


def _ply_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return np.dtype("u1")
    if dtype.kind in "iu" and dtype.itemsize == 8:
        return np.dtype(dtype.kind + "4")
    if dtype.kind == "f" and dtype.itemsize > 8:
        return np.dtype("f8")
    if dtype.str[1:] not in _PLY_DTYPES:
        raise TypeError(f"Column dtype {dtype} cannot be stored in a PLY file")
    return dtype.newbyteorder("=")


def pandas_to_ply(csv, csv_file_provided=False, output_file_path=None):
    """Write a DataFrame as a binary PLY 'vertex' element, one property per column.

    Column dtypes are preserved (float64 stays float64, uint8 stays uint8), row
    order is preserved, and the file is binary because the dataset reader
    (torch_points3d/modules/KPConv/plyutils.py) rejects ASCII PLY.
    """
    if output_file_path is None:
        raise ValueError("output_file_path is required")

    df = pd.read_csv(csv) if csv_file_provided else csv

    # remove duplicated columns and spaces in column names
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = [col.replace(" ", "_") for col in df.columns]

    dtypes = [(col, _ply_dtype(df[col].dtype)) for col in df.columns]
    data = np.empty(len(df), dtype=dtypes)
    for col in df.columns:
        data[col] = df[col].to_numpy()

    vertex = PlyElement.describe(data, "vertex")
    PlyData([vertex], text=False).write(output_file_path)
    return output_file_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a CSV file to a binary PLY file.")
    parser.add_argument("csv_path", help="Path to the CSV file to be read.")
    parser.add_argument("ply_path", help="Path to the PLY file to be written.")
    args = parser.parse_args()
    pandas_to_ply(args.csv_path, csv_file_provided=True, output_file_path=args.ply_path)
