import struct
import numpy as np
import cv2


def log_depth_image(depth, rgb):
    """
    Logs depth image into a file

    Writes binary values of the depth matrix into a raw file and saves
    the shape information of the matrix into a .csv file with the format
    "[height (int)],[width (int)][newline]"

    Also saves the rgb image in a png file

    There needs to be a folder named log in the folder where the script
    is run from

    :param depth: a numpy depth matrix of int16 values of shape [h, w]
    :return: True
    """
    # Save raw data into a depth .raw file
    path = "./log/last_depth.raw"
    values = np.reshape(depth, np.prod(depth.shape)) * 1000
    num_values = np.prod(depth.shape)
    values_bytes = struct.pack("H" * num_values, *values.astype('uint16').tolist())
    open(path, "wb").write(values_bytes)

    # Save shape data into a .csv data file
    shape_data = str(depth.shape[0]) + "," + str(depth.shape[1]) + "\n"
    open("./log/shape.csv", "w").write(shape_data)

    # Save the rgb image in a png file
    rgb = cv2.cvtColor((rgb*255).astype('uint8'), cv2.COLOR_BGR2RGB)
    cv2.imwrite("./log/rgb.png", rgb)
    return True
