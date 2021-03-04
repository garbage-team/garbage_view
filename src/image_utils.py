# Utils for images
import numpy as np
import matplotlib.pyplot as plt


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
    for i, img in iter(img_list):
        plt.subplot(1, len(img_list), i)
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
    # A function that takes the predicted depth bins from the encoder-decoder
    # and returns a numeric depth value for all pixels
    # Depth bins should be in the order of [b, w, h, c]
    # TODO refactor to global variables
    bin_interval = (np.log10(80) - np.log10(0.25))/150
    borders = np.array([np.log10(0.25) + (bin_interval * (i + 0.5)) for i in range(150)])
    depth = depth_bins * borders
    depth = np.sum(depth, axis=3)
    depth = 10 ** depth
    return depth
