# Utils for images
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.config import cfg


def resize_normalize(rgb, d, max_depth=80., model_max_output=80.):
    # Normalizes and resize the tf tensors of rgb and d
    d = tf.expand_dims(d, -1)
    img_size = 224
    resize = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size)
    ])
    rgb = resize(rgb)
    d = resize(d)
    d_scale = model_max_output / max_depth
    return tf.cast(rgb, tf.float32) / 255., tf.cast(d, tf.float32) / d_scale


def normalize_rgb(rgb):
    # Normalizes rgb images from values of 0-255 to values of 0.0-1.0
    # rgb :: numpy.array of shape (height, width, 3) of uint8 type
    # returns :: numpy.array of shape (height, width, 3) of float32 type
    rgb = rgb.astype('float32')
    rgb = rgb / 255.0
    return rgb


def normalize_d(d, max_depth=None):
    # Normalizes depth images from integer values to floating point values
    # If max_depth is not specified or None then the max_depth value is specified to
    # the maximum value of the depth image
    # d :: numpy.array of shape (height, width, 1) of integer type
    # max_depth :: value of the maximum depth to normalize to
    # returns :: numpy.array normalized depth image of values between 0.0-1.0 dtype=float32
    d = d.astype('float32')
    if max_depth is None:
        max_depth = np.max(d)
    d = d / max_depth
    return d


def un_normalize_rgb(normalized_rgb):
    # Un-normalizes a normalized rgb image
    # normalized_rgb :: numpy array of a normalized rgb image with values between 0.0-1.0
    # returns :: numpy array of an rgb image of values 0-255 of dtype uint8
    rgb = normalized_rgb * 255.0
    rgb = np.round(rgb)
    rgb = rgb.astype('uint8')
    return rgb


def un_normalize_d(normalized_d, max_depth):
    # Un-normalizes a normalized depth image
    # normalized_d :: numpy array of normalized depth image
    # max_depth :: the maximum possible value of the un-normalized depth image
    # returns :: un-normalized depth image as numpy array of type uint16
    d = normalized_d * max_depth
    d = np.round(d).astype('uint16')
    return d


def display_images(img_list):
    # Displays a list of images in one plot
    # img_list :: a list of images numpy arrays
    # returns :: None
    plt.figure(0)
    if len(img_list) > 5:
        raise ValueError("Too many images in plot function")
    for i, img in enumerate(img_list):
        plt.subplot(1, len(img_list), i+1)
        plt.imshow(img, cmap='hsv')
    plt.show()
    return None


def display_overlayed(rgb, d):
    plt.figure(0)
    if rgb.shape[:2] != d.shape[:2]:
        raise ValueError("Shape mismatch")
    plt.imshow(rgb)
    plt.imshow(d, alpha=0.5, cmap='hsv')
    plt.show()
    return None


def bins_to_depth(depth_bins):
    """
    Converts a bin tensor into a depth image

    :param depth_bins: the depth bins in one_hot encoding, shape (b, h, w, c)
    the depth bins can also be passed as softmax bins of shape (b, h, w, c)
    :return: a depth image of shape (b, h, w) with type tf.float32
    """
    bin_interval = cfg["bin_interval"]
    # the borders variable here holds the depth for each specific value of the one hot encoded bins
    borders = tf.constant([np.log10(cfg["min_depth"]) + (bin_interval * (i + 0.5)) for i in range(cfg["depth_bins"])],
                          dtype=tf.float32)  # [c]
    depth = tf.reduce_sum(tf.multiply(depth_bins, borders),
                          axis=-1, keepdims=False)  # [b, h, w, c] * [c] -> [b, h, w, c] -> [b, h, w]
    depth = tf.math.pow(10, depth)
    return depth


def depth_to_bins(depth):
    """
    Converts a depth channel of an image into a bins tensor

    :param depth: a depth image, possibly containing a channel dimension
    shape either (b, h, w) or (b, h, w, 1)
    :return: a bin index tensor of shape (b, h, w) of type tf.int32
    """
    if len(depth.shape) >= 4:
        depth = depth[:, :, :, 0]
    bin_interval = cfg["bin_interval"]
    valid_mask = tf.math.greater_equal(depth, tf.constant(cfg["min_depth"]))
    depth = tf.math.add(depth, -cfg["min_depth"])
    depth = tf.keras.activations.relu(depth, max_value=cfg["max_depth"]-cfg["min_depth"])
    depth = tf.math.add(depth, cfg["min_depth"])
    x = tf.math.log(cfg["min_depth"]) / tf.math.log(tf.constant(10, dtype=depth.dtype))
    bins = (tf.math.log(depth) / tf.math.log(tf.constant(10, dtype=depth.dtype)) - x) / bin_interval
    bins = tf.cast(tf.round(bins), tf.int32)
    bins = tf.math.multiply(bins, tf.cast(valid_mask, bins.dtype))
    bins = tf.math.add(bins,
                       tf.math.multiply(
                           tf.cast(
                               tf.math.logical_not(valid_mask),
                               bins.dtype),
                           cfg["depth_bins"] + 1))
    max_mask = tf.math.equal(bins, cfg["depth_bins"])
    bins = tf.math.subtract(bins, tf.cast(max_mask, bins.dtype))
    return bins
