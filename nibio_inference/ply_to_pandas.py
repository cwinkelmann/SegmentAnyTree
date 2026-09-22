import numpy as np
import pandas as pd
from plyfile import PlyData


def ply_to_pandas(ply_file_path, csv_file_path=None):
    """Read the point element of a PLY file into a DataFrame, one column per property.

    Column dtypes and row order are preserved. Works for ASCII and binary PLY.

    Args:
        ply_file_path (str): The path to the PLY file to be read.
        csv_file_path (str, optional): If given, the data is also written to this CSV file.

    Returns:
        pandas.DataFrame
    """
    ply_content = PlyData.read(ply_file_path)

    available_elements = [elem.name for elem in ply_content.elements]
    if "vertex" in available_elements:
        point_element_name = "vertex"
    elif "point" in available_elements:
        point_element_name = "point"
    else:
        raise ValueError(f"No 'vertex' or 'point' element in {ply_file_path}, found {available_elements}")

    point_data = ply_content[point_element_name].data
    # PLY may be big-endian; pandas wants native byte order
    columns = {name: np.ascontiguousarray(point_data[name]).astype(point_data[name].dtype.newbyteorder("="), copy=False)
               for name in point_data.dtype.names}
    points_df = pd.DataFrame(columns)

    if csv_file_path is not None:
        points_df.to_csv(csv_file_path, index=False)

    return points_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a PLY file to a CSV file.")
    parser.add_argument("ply_path", type=str, help="Path to the PLY file to be read.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file to be saved.")
    args = parser.parse_args()
    ply_to_pandas(args.ply_path, args.csv_path)
