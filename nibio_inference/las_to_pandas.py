import numpy as np
import pandas as pd
import laspy

# works with laspy 2.x


def las_to_pandas(las_file_path, csv_file_path=None):
    """Read a LAS/LAZ file into a DataFrame, one column per dimension.

    X, Y, Z are the scaled (real-world) coordinates as float64. All other
    standard and extra dimensions keep their LAS dtype. Row order is the file order.
    """
    file_content = laspy.read(las_file_path)

    columns = {}
    for dim in file_content.point_format.dimension_names:
        if dim in ("X", "Y", "Z"):
            # laspy: lowercase gives scaled float64 coordinates, uppercase the raw int32
            columns[dim] = np.asarray(getattr(file_content, dim.lower()), dtype=np.float64)
        else:
            # laspy returns array views (ScaledArrayView, SubFieldView); materialise them
            columns[dim] = np.asarray(file_content[dim])

    points_df = pd.DataFrame(columns)

    if csv_file_path is not None:
        points_df.to_csv(csv_file_path, index=False, header=True, sep=",")

    return points_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert las or laz files to pandas dataframes.")
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input file.")
    parser.add_argument("-o", "--output_file", type=str, help="Path to the output file.")

    args = parser.parse_args()

    las_to_pandas(args.input_file, args.output_file)
