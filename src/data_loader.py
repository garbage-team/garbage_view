import tensorflow as tf
import pathlib
import numpy as np
import struct
import cv2
import random
import src.image_utils
import tensorflow_datasets as tfds
from src.config import cfg


def create_paths(base):
    """
    Create the paths for all images in the base folder, differentiates between the file types .png and .raw.
    Each rgb and depth image file-pair must be named with a unique identifier followed by an underscore
    @param base: String with path to base folder
    @return: list[(rgb_path, d_path)] rgb paths and depth paths respectively
    """
    base = pathlib.Path(base)
    rgb_paths = list(base.glob("*/*.png"))
    depth_paths = list(base.glob("*/*.raw"))
    path_doubles = []
    for path in rgb_paths:
        path_id = str(path.parts[-2]) + "/" + str(path.parts[-1])
        path_id = path_id.split(".")[0]
        d_path = None
        for d in depth_paths:
            d_p = str(d.parts[-2]) + "/" + str(d.parts[-1])
            if path_id == d_p.split(".")[0]:
                d_path = d
                break
        path_doubles.append((path, d_path))
    return path_doubles


def load_data(path_doubles):
    """
    Loads data from the list of path doubles generated from the create_paths function,the function converts
    bgr images to rgb images, therefore returns rgb images. Intended to be used after generating path doubles
    with create_paths.
    @param path_doubles: list[(rgb_path, d_path)] list of path doubles
    @return: list[(rgb_img, d_img)] list of data doubles
    """
    data_doubles = []
    for paths in path_doubles:
        rgb = paths[0]
        depth = paths[1]
        # Read the depth image:
        with open(str(depth), "rb") as file:
            d_img = file.read()
        d_img = np.array(struct.unpack("H" * 480*640, d_img), dtype='uint16').reshape((480, 640, 1))
        # Read the rgb image:
        rgb_img = cv2.cvtColor(cv2.imread(str(rgb)), cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (640, 480))
        # Save the image pair in list:
        data_doubles.append((rgb_img, d_img))
    return data_doubles


def create_dataset(path="../data/train", shape=(224, 224), shuffle=1000, batch=4):
    """
    NOTE: We recommend using a tfrecord dataset instead, especially for larger datasets.
    Creates the tensorflow dataset object that can be used for training a model
    The dataset contains pairs of images shuffled and batched.
    @param path: String path to the data for the dataset
    @param shape: Tuple of (height, width) for the output of images in the dataset
    @param shuffle: Buffer size for shuffling the the dataset
    @param batch: Batch size for the dataset
    @return: A tensorflow.data.Dataset object containing pairs of rgb and depth images
    """
    paths = create_paths(path)
    ds = tf.data.Dataset. \
        from_generator(lambda: ds_generator(paths, shape),
                       output_types=(tf.float32, tf.float32),
                       output_shapes=([shape[0], shape[1], 3], [shape[0], shape[1], 1]))
    ds = ds.shuffle(shuffle).batch(batch)
    return ds


def ds_generator(data, shape):
    """
    Generates the rgb and depth images for the dataset
    @param data: A list[(rgb_path, d_path)] , list of pairs of the data paths
    @param shape: A tuple of (height, width) for image output size
    @return: rgb_img, d_img, RGB image and depth map pair for the dataset
    """
    for rgb_path, d_path in data:
        rgb, d = load_data([(rgb_path, d_path)])[0]
        rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb, d = src.image_utils.img_augmentation(rgb, d)
        yield rgb, d


def load_nyudv2(batch=4, shuffle=True, ds_path='D:/wsl/tensorflow_datasets', split='train'):
    """
    Loads the nyudv2 dataset from tensorflow datasets. Note that the dataset will download if not existing
    in the path, and is ~80GB.
    @param batch: Batch size for the dataset
    @param shuffle: Bool, determines if the entries should be shuffled or not
    @param ds_path: String path to the tensorflow_datasets folder
    @param split: String, either 'train' or 'validation'
    @return: A tensorflow.data.Dataset object containing pairs of rgb and depth images
    """
    nyudv2, info = tfds.load('nyu_depth_v2', split=split, with_info=True, shuffle_files=shuffle, as_supervised=True,
                             data_dir=ds_path)
    nyudv2 = nyudv2.map(src.image_utils.img_augmentation,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    nyudv2 = nyudv2.map(lambda x, y: src.image_utils.resize_normalize(x, y, max_depth=cfg["max_depth"]),
                        num_parallel_calls=tf.data.AUTOTUNE)
    # Batches the dataset if the param is given
    if batch:
        nyudv2 = nyudv2.batch(batch)
    nyudv2 = nyudv2.prefetch(tf.data.experimental.AUTOTUNE)
    return nyudv2


def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte. Used when generating a tfrecord. See tensorflow tfrecord generation
    documentation for further information.
    @param value: Bytes or string input to be converted
    @return: Bytes list
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_example(image, label):
    """
    Links a rgb and depth image when saving a tfrecord dataset. See tensorflow tfrecord generation
    documentation for further information.
    @param image: RGB image (h, w, c)
    @param label: Depth map image (h, w, c)
    @return: tf.train.Example used when saving as tfrecord
    """
    feature = {
        'depth': _bytes_feature(label.tobytes()),
        'rgb': _bytes_feature(image.tobytes())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(file, data_path):
    """
    Converts data into a tfrecord dataset for easier and quicker handling of dataset. See tensorflow tfrecord
    documentation for further information. Automatically splits the data randomly into training and validation.
    @param file: Output file name for the TFRecord files without _train and _validation
    @param data_path: Path to folder with input data, with .png (rgb) and .raw (depth) images
    @return: True, if successfully writen to given path, with _train and _validation splits
    """
    data = load_data(create_paths(data_path))
    val_split = int(0.1 * len(data))
    with tf.io.TFRecordWriter(file + "_validation") as writer:
        for _ in range(val_split):
            val_data = data.pop(random.randrange(len(data)))
            image = val_data[0]
            label = val_data[1]
            tf_example = image_example(image, label)
            writer.write(tf_example.SerializeToString())
    with tf.io.TFRecordWriter(file+"_train") as writer:
        for i in range(len(data)):
            image = data[i][0]
            label = data[i][1]
            tf_example = image_example(image, label)
            writer.write(tf_example.SerializeToString())
    return True


def load_tfrecord_dataset(tf_record_files, shuffle=2000, batch=4, augment=True):
    """
    Loads a dataset saved as one or more tfrecord files

    :param tf_record_files: input file name for the TFRecord file(s)
    :param shuffle: The size of the shuffle buffer in number of elements (int)
    :param batch: The batch size of the returned dataset (int)
    :param augment: Whether or not to augment the dataset by cropping and flipping etc. (bool)
    :return: dataset with image and label pairs (rgb, depth)
    """
    ds = tf.data.TFRecordDataset(tf_record_files)

    ds = ds.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(src.image_utils.img_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: src.image_utils.resize_normalize(x, y, max_depth=cfg["max_depth"]*1000.),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(shuffle)
    ds = ds.batch(batch)
    return ds


def read_tfrecord(example):
    """
    Decodes and reads a saved tfrecord and returns the image and depth map pairs
    @param example: tf.train.Example from a tfrecord
    @return: RGB image and depth map pair
    """
    feature = {
        'depth': tf.io.FixedLenFeature([], tf.string),
        'rgb': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature)
    image = tf.io.decode_raw(example['rgb'], out_type="uint8")
    label = tf.io.decode_raw(example['depth'], out_type=tf.uint16)
    image = tf.reshape(image, (480, 640, 3))
    label = tf.reshape(label, (480, 640, 1))
    return image, label
