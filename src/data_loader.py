import tensorflow as tf
import pathlib
import numpy as np
import struct
import cv2
import src.image_utils
import tensorflow_datasets as tfds


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
        path_id = path.parts[-1].split(".")[0]
        d_path = None
        for d in depth_paths:
            if path_id == d.parts[-1].split(".")[0]:
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
        d_img = np.array(struct.unpack("H" * 480*640, d_img)).reshape((480, 640, 1))
        d_img = d_img / 1000
        # read the rgb image:
        rgb_img = cv2.cvtColor(cv2.imread(str(rgb)), cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (640, 480))
        # rgb_img = np.array(rgb_img)
        # save the images in list:
        data_doubles.append((rgb_img, d_img))
    return data_doubles


def create_dataset(path="../data/train", shape=(224, 224), shuffle=1000, batch=4):
    # Creates the tensorflow dataset object that can be used for training a model
    # The dataset contains pairs of images shuffled and batched as 10 image pairs
    # path :: the string to the folder that contains the training data
    # shape :: a tuple of (height, width) for the output of images in the dataset
    # shuffle :: buffer size for shuffling the the dataset
    # batch :: batch size for the dataset
    # returns :: a tensorflow.data.Dataset object containing pairs of rgb and depth images
    paths = create_paths(path)
    ds = tf.data.Dataset. \
        from_generator(lambda: ds_generator(paths, shape),
                       output_types=(tf.float32, tf.float32),
                       output_shapes=([shape[0], shape[1], 3], [shape[0], shape[1], 1]))
    ds = ds.shuffle(shuffle).batch(batch)
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
        rgb, d = src.image_utils.img_augmentation(rgb, d)
        yield rgb, d


def load_nyudv2(batch=4, shuffle=True, ds_path='D:/wsl/tensorflow_datasets', split='train'):
    nyudv2, info = tfds.load('nyu_depth_v2', split=split, with_info=True, shuffle_files=shuffle, as_supervised=True,
                             data_dir=ds_path)
    nyudv2 = nyudv2.map(
        src.image_utils.img_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    nyudv2 = nyudv2.cache()

    if batch:
        nyudv2 = nyudv2.batch(batch)

    nyudv2 = nyudv2.prefetch(tf.data.experimental.AUTOTUNE)

    return nyudv2


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image, label, dims):
    feature = {
        'rgb_shape': _bytes_feature(dims[0]),
        'd_shape': _bytes_feature(dims[1]),
        'label': _bytes_feature(label.tobytes()),
        'image_raw': _bytes_feature(image.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(file, data_path):
    """

    :param file: output file name for the TFRecord file
    :param data_path: path to folder with input data, with .png (rgb) and .raw (depth) images
    :return: None, only writes to given path
    """
    data = load_data(create_paths(data_path))
    n_samples = len(data)
    rgb_shape = ','.join(str(i) for i in data[0][0].shape)
    rgb_shape = bytes(rgb_shape, 'utf-8')
    d_shape = ','.join(str(i) for i in data[0][1].shape)
    d_shape = bytes(d_shape, 'utf-8')
    dims = (rgb_shape, d_shape)
    with tf.io.TFRecordWriter(file) as writer:
        for i in range(n_samples):
            image = data[i][0]
            label = data[i][1]
            tf_example = image_example(image, label, dims)
            writer.write(tf_example.SerializeToString())


def read_tfrecord(record_file):
    dataset = tf.data.TFRecordDataset(record_file)
    parsed_record = dataset.map(_parse_record)
    decoded_record = parsed_record.map(_decode_record)
    return decoded_record


def _parse_record(record):
    feature = {
        'rgb_shape': tf.io.FixedLenFeature([], tf.string),
        'd_shape': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, feature)


def _decode_record(record):
    image = tf.io.decode_raw(
        record['image_raw'], out_type="uint8", little_endian=True, fixed_length=None, name=None
    )
    label = tf.io.decode_raw(
        record['label'], out_type=tf.float64, little_endian=True, fixed_length=None, name=None
    )
    rgb_shape = record['rgb_shape'].numpy().decode('utf-8')
    rgb_shape = tuple(int(x) for x in rgb_shape.split(','))
    d_shape = record['d_shape'].numpy().decode('utf-8')
    d_shape = tuple(int(x) for x in d_shape.split(','))
    image = tf.reshape(image, rgb_shape)
    label = tf.reshape(label, d_shape)
    return image, label
# TODO Figure out what to return here to load the data into a functional dataset, https://keras.io/examples/keras_recipes/tfrecord/

if __name__ == "__main__":
    dataset = create_dataset("../data/", (224, 224))
    for img_rgb, img_d in dataset.take(1):
        print(img_rgb.shape)
        print(img_d.shape)
        image_utils.display_overlayed(img_rgb[0], img_d[0])
