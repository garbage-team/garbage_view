# Utilities for images
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import struct
import cv2
import open3d as o3d
from math import pi
from src.config import cfg


def img_augmentation(rgb, d):
    """
    Image augmentation for altering the training images to combat overfitting. Randomly flips the image pair and adds
    noise to the RGB image.
    @param rgb: RGB image (h, w, 3)
    @param d: Depth map (h, w) or (h, w, 1)
    @return: Augmented RGB and depth map
    """

    if len(d.shape) == 2:
        d = tf.expand_dims(d, -1)
    flip_case = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    # if case == 0, does not flip the image
    # Flip horizontally
    if flip_case == 1:
        rgb = tf.image.flip_left_right(rgb)
        d = tf.image.flip_left_right(d)
    # Flip vertically
    if flip_case == 2:
        rgb = tf.image.flip_up_down(rgb)
        d = tf.image.flip_up_down(d)
    # Flip horizontally and vertically
    if flip_case == 3:
        rgb = tf.image.flip_up_down(rgb)
        rgb = tf.image.flip_left_right(rgb)
        d = tf.image.flip_up_down(d)
        d = tf.image.flip_left_right(d)

    # Add noise augmentation
    max_noise = 2
    noise = tf.random.uniform(shape=rgb.shape, minval=-max_noise, maxval=max_noise, dtype=tf.int32)
    noise_mask = tf.logical_or(tf.greater(rgb, 255-max_noise), tf.less(rgb, max_noise))
    noise = noise * tf.cast(tf.logical_not(noise_mask), tf.int32)
    rgb = tf.cast(tf.cast(rgb, tf.int32) + noise, tf.uint8)
    return rgb, d


def resize_normalize(rgb, d, max_depth=cfg["max_depth"], model_max_output=cfg["max_depth"], img_size=224):
    """
    Resizes the image pair to the input size for the model, and normalizes the image values to values between 0-1.
    @param rgb: RGB image (h, w, 3)
    @param d: Depth map (h, w, 1) or (h, w)
    @param max_depth: Maximum depth in the image dataset, defined in config.py
    @param model_max_output: Maximum depth output from the model, defined in config.py
    @param img_size: Image size for input to the model
    @return: Resized and normalized RGB and depth map
    """
    resize = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size)
    ])
    rgb = resize(rgb)

    if d.shape == 2:
        d = tf.expand_dims(d, -1)
    d = resize(d)
    d_scale = model_max_output / max_depth

    return tf.cast(rgb, tf.float32) / 255., tf.cast(d, tf.float32) * d_scale


def normalize_rgb(rgb):
    """
    NOTE: does not use tensorflow functions and not suitable for use in dataset mapping
    Normalizes rgb images from values of 0-255 to values of 0.0-1.0

    @param rgb: numpy.array of shape (height, width, 3) of uint8 type
    @type rgb: numpy.array of shape (height, width, 3) of float32 type
    @return:
    @rtype:
    """
    rgb = rgb.astype('float32')
    rgb = rgb / 255.0
    return rgb


def normalize_d(d, max_depth=None):
    """
    NOTE: does not use tensorflow functions and not suitable for use in dataset mapping
    Normalizes depth images from integer values to floating point values
    If max_depth is not specified or None then the max_depth value is specified to
    the maximum value of the depth image

    @param d: numpy.array of shape (height, width, 1) of integer type
    @param max_depth: value of the maximum depth to normalize to
    @return: numpy.array normalized depth image of values between 0.0-1.0 dtype=float32
    """
    d = d.astype('float32')
    if max_depth is None:
        max_depth = np.max(d)
    d = d / max_depth
    return d


def un_normalize_rgb(normalized_rgb):
    """
    NOTE: does not use tensorflow functions and not suitable for use in dataset mapping
    Un-normalizes a normalized rgb image
    @param normalized_rgb:numpy array of a normalized rgb image with values between 0.0-1.0
    @return: numpy array of an rgb image of values 0-255 of dtype uint8
    """
    rgb = normalized_rgb * 255.0
    rgb = np.round(rgb)
    rgb = rgb.astype('uint8')
    return rgb


def un_normalize_d(normalized_d, max_depth):
    """
    NOTE: does not use tensorflow functions and not suitable for use in dataset mapping
    Un-normalizes a normalized depth image
    @param normalized_d: numpy array of normalized depth image
    @param max_depth: the maximum possible value of the un-normalized depth image
    @return:  un-normalized depth image as numpy array of type uint16
    """
    d = normalized_d * max_depth
    d = np.round(d).astype('uint16')
    return d


def display_images(img_list):
    """
    Displays a list of maximum 5 images in one plot
    @param img_list: a list of images as numpy arrays
    @return: None
    """
    plt.figure(0)
    if len(img_list) > 5:
        raise ValueError("Too many images in plot function")
    for i, img in enumerate(img_list):
        plt.subplot(1, len(img_list), i+1)
        plt.imshow(img, cmap='hsv')
        plt.clim(0, 5)
    plt.show()
    return None


def display_overlayed(rgb, d):
    """
    Tool for checking alignment of rgb and depth images by displaying them overlayed
    @param rgb: RGB image (h, w, 3)
    @param d: Depth image (h, w, 1) or (h, w)
    @return: None, displays overlayed images
    """
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
    depth = tf.math.pow(10., depth)
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


def depth_to_pcd(depth, rgb, output_path, fov, size=224*224, view=True):
    """
    Transforms a depth map to a point cloud and exports the point cloud as a .pcd file.
    @param depth: Depth map as numpy array (h, w, 1)
    @param rgb: RGB image as numpy array (h, w, 3)
    @param output_path: String to the output path of the .pcd file
    @param fov: String, denoting which camera was used to generate the images
    @param size: h*w of the output point cloud
    @param view: Bool, displays viewer if True
    @return: None
    """
    xyz = numpy_depth_to_xyz(depth, fov)
    p = np.reshape(xyz, (size, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud(output_path, pcd)
    if view:
        o3d.visualization.draw_geometries([pcd])
    return None


def images_to_pcd(rgb_path, d_path, pcd_path, fov="webcam"):
    """
    Converts images from given paths to a .pcd point cloud file
    @param rgb_path: String, path to RGB image
    @param d_path: String, path to depth image
    @param pcd_path: String, outpur file path
    @param fov: String, denoting which camera was used to generate the images
    @return: None
    """
    # TODO perhaps remove image loading here, and allow passing arrays for rgb and d instead
    rgb_img = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    shape = rgb_img.shape[0:2]
    img_size = rgb_img.shape[0] * rgb_img.shape[1]
    rgb_img = np.reshape(rgb_img, (img_size, 3)) / 255
    with open(str(d_path), "rb") as file:
        d_img = file.read()
    d_img = np.array(struct.unpack("H" * img_size, d_img)).reshape(shape)
    depth_to_pcd(d_img, rgb_img, pcd_path, fov=fov, size=img_size)
    return None


def plot_history(history):
    """
    Plots a model training history for an evaluation of over/underfitting
    :param history: history object generated by model.fit()
    :return: None, displays plots in figure
    """
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = list(range(1, len(loss)+1))
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    return None


def numpy_depth_to_xyz(depth, fov):
    """
    Converts a depth map to a point cloud using numpy functions instead of tensorflow functions. Based on
    src/loss_functions.py:depth_to_xyz()
    @param depth: Depth map (h, w, 1)
    @param fov: String, denoting which camera was used to generate the images
    @return: Point cloud as (h, w, 3), where channels are x, y, z
    """
    x_size = depth.shape[1]
    y_size = depth.shape[0]

    x = np.asarray([i - (x_size // 2) for i in range(x_size)])  # [w,]
    x = np.tile(np.expand_dims(x, axis=0), (y_size, 1))         # [h, w]
    x = np.tan(cfg[fov+"_h_fov"] * pi / 360) / (x_size / 2) * np.multiply(x, depth)

    y = np.asarray([i - (y_size // 2) for i in range(y_size)])  # [h,]
    y = np.tile(np.expand_dims(y, axis=-1), (1, x_size))        # [h, w]
    y = np.tan(cfg[fov+"_v_fov"] * pi / 360) / (y_size / 2) * np.multiply(y, depth)
    # The z-axis can be moved here by subtracting a desired height
    z = depth

    x = np.expand_dims(x, -1)  # [h, w, 1]
    y = np.expand_dims(y, -1)  # [h, w, 1]
    z = np.expand_dims(z, -1)  # [h, w, 1]
    p = np.concatenate((x, y, z), axis=-1)  # [h, w, 3]

    return p