import tensorflow as tf
import pathlib
import numpy as np
import struct
import cv2
import image_utils


def create_paths(base):
    # Create the paths for all images in the base folder
    # Differentiates between the file types .png and .raw
    # Each rgb and depth image file-pair must be named with a unique identifier followed
    # by an underscore
    # base :: string with path to base folder
    # returns :: list[(rgb_path, d_path)] rgb paths and depth paths respectively
    base = pathlib.Path(base)
    rgb_paths = list(base.glob("*.png"))
    depth_paths = list(base.glob("*.raw"))
    path_doubles = []
    for path in rgb_paths:
        path_id = path.parts[-1].split("_")[0]
        d_path = None
        for d in depth_paths:
            if path_id == d.parts[-1].split("_")[0]:
                d_path = d
                break
        path_doubles.append((path, d_path))
    return path_doubles


def load_data(path_doubles):
    # Loads data from the list of path doubles generated from the create_paths function,
    # the function converts bgr images to rgb images, therefore returns rgb images
    # path_doubles :: list[(rgb_path, d_path)] list of path doubles
    # returns :: list[(rgb_img, d_img)] list of data doubles
    data_doubles = []
    for paths in path_doubles:
        rgb = paths[0]
        depth = paths[1]
        # read the depth image:
        with open(str(depth), "rb") as file:
            d_img = file.read()
        d_img = np.array(struct.unpack("H" * 230400, d_img)).reshape((360, 640))
        # read the rgb image:
        rgb_img = cv2.cvtColor(cv2.imread(str(rgb)), cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (640, 360))
        # rgb_img = np.array(rgb_img)
        # save the images in list:
        data_doubles.append((rgb_img, d_img))
    return data_doubles


def create_dataset(path="../data/", shape=(224, 224)):
    # Creates the tensorflow dataset object that can be used for training a model
    # The dataset contains pairs of images shuffled and batched as 10 image pairs
    # path :: the string to the folder that contains the training data
    # shape :: a tuple of (height, width) for the output of images in the dataset
    # returns :: a tensorflow.data.Dataset object containing pairs of rgb and depth images
    paths = create_paths(path)
    ds = tf.data.Dataset. \
        from_generator(lambda: ds_generator(paths, shape),
                       output_types=(tf.float32, tf.float32),
                       output_shapes=([shape[0], shape[1], 3], [shape[0], shape[1], 1]))
    ds = ds.shuffle(10).batch(10)
    return ds


def ds_generator(data, shape):
    # Generates the rgb and depth images for the dataset
    # data :: a list[(rgb_path, d_path)]
    # shape :: a tuple of (height, width) for image output size
    # yields :: rgb_img, d_img
    for rgb_path, d_path in data:
        rgb, d = load_data([(rgb_path, d_path)])[0]
        rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        d = np.expand_dims(d, axis=-1)
        rgb = image_utils.normalize_rgb(rgb)
        d = image_utils.normalize_d(d, max_depth=4000)
        yield rgb, d


if __name__ == "__main__":
    dataset = create_dataset("../data/", (224, 224))
    for img_rgb, img_d in dataset.take(1):
        print(img_rgb.shape)
        print(img_d.shape)
        image_utils.display_overlayed(img_rgb[0], img_d[0])
