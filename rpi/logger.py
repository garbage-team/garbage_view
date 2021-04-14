import struct
import numpy as np


def log_depth_image(depth):
    """
    Logs depth image into a file

    Writes binary values of the depth matrix into a raw file and saves
    the shape information of the matrix into a .csv file with the format
    "[height],[width]"

    :param depth: a numpy depth matrix of int16 values of shape [h, w]
    :return: True if logged correctly
    """
    # Save raw data into a depth .raw file
    path = "./log/last_depth.raw"
    values = np.reshape(depth, np.sum(depth.shape))
    num_values = np.sum(depth.shape)
    values_bytes = struct.pack("H" * num_values, values)
    open(path, "wb").write(values_bytes)

    # Save shape data into a .csv data file
    shape_data = str(depth.shape[0]) + "," + str(depth.shape[1]) + "\n"
    open("./log/shape.csv", "w").write(shape_data)
    return True
