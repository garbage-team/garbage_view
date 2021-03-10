import tensorflow as tf
import numpy as np
from src.image_utils import depth_to_bins
from src.config import cfg


def wcel_loss(gt, pred):
    # Weighted cross entropy loss function for determining the
    # loss of the predicted logits and the ground truth depth logits
    # param pred: [[b, h, w, 150], [b, h, w, 150]]
    depth_bins = cfg["depth_bins"]
    [pred_logits, pred_softmax] = pred
    gt_bins = depth_to_bins(gt)
    pred_logsoft = tf.math.log(pred_softmax)
    pred_logsoft = tf.transpose(pred_logsoft, perm=[0, 2, 1, 3])  # [b, h, w, 150] -> [b, w, h, 150]
    valid_mask = tf.logical_not(tf.equal(gt_bins, depth_bins + 1))
    one_hot = tf.one_hot(gt_bins, depth_bins)  # [b, h, w, 1] -> [b, h, w, 150]

    weights = cfg["wcel_weights"]
    weights = tf.linalg.matmul(one_hot, weights)  # [b, h, w, 150] x [150, 150] -> [b, h, w, 150]
    pred_losses = tf.math.multiply(weights, pred_logsoft)

    valid_pixels = tf.reduce_sum(tf.cast(valid_mask, tf.dtypes.float32))
    loss = -1. * tf.reduce_sum(pred_losses) / valid_pixels
    return loss


def virtual_normal_loss(gt, pred):
    """

    :param gt:
    :param pred:
    :return:
    """

    return None


def generate_random_p_groups(xyz_1, xyz_2, shape=(224, 224, 3), sample_ratio=0.15):
    """
    Generates a random set of points groups from two point maps

    The first point map could be ground truth while the other could
    be the predicted point map from a neural network

    The shape should be (h, w, 3) as there should be points in the
    point map in 3-space, ordered (x, y, z)

    :param xyz_1: the first point map with shape=[b, shape] (b, h, w, c)
    :param xyz_2: the second point map
    :param shape: the shape of the point map (h, w, c) where c should be 3
    :param sample_ratio: the ratio of sampled points, defaults to 0.15
    :return: two points groups of respective points from the point maps, (groups_1, groups_2)
    """
    (height, width, channels) = shape
    total_indices = height * width
    sampled_indices = tf.cast(tf.math.round(total_indices * sample_ratio), tf.dtypes.int32)
    p1_i = tf.random.uniform((sampled_indices, 1), minval=0, maxval=total_indices,
                           dtype=tf.dtypes.int32)
    p2_i = tf.random.uniform((sampled_indices, 1), minval=0, maxval=total_indices,
                           dtype=tf.dtypes.int32)
    p3_i = tf.random.uniform((sampled_indices, 1), minval=0, maxval=total_indices,
                           dtype=tf.dtypes.int32)
    indices = tf.concat((p1_i, p2_i, p3_i), axis=-1)
    groups_1 = generate_p_groups(xyz_1, indices, shape=shape)
    groups_2 = generate_p_groups(xyz_2, indices, shape=shape)
    return groups_1, groups_2


def generate_p_groups(xyz, indices, shape=(224, 224, 3), groups_dimension=2):
    """
    Generates points from indices and a point map, xyz

    :param xyz: a points map of shape (B, h, w, c) where c is (x, y, z)
    :param indices: a list of indices, indexed in one dimension [0 -> h*w) with shape (n, 3)
    :param shape: a tuple with the shape of the point map (h, w, c)
    :param groups_dimension: the dimension in which the groups are indexed
    :return: a tensor with the points groups for the point map of shape (B, n, 3, 3)
    """
    (height, width, channels) = shape
    xyz = tf.reshape(xyz, (-1, height*width, channels))
    p1 = tf.expand_dims(tf.gather(xyz, indices[:, 0], axis=1), groups_dimension)
    p2 = tf.expand_dims(tf.gather(xyz, indices[:, 1], axis=1), groups_dimension)
    p3 = tf.expand_dims(tf.gather(xyz, indices[:, 2], axis=1), groups_dimension)
    return tf.concat((p1, p2, p3), groups_dimension)


def generate_normals(groups):
    valid_mask = tf.logical_not(generate_invalid_mask(groups))
    norms = normalize_vectors(groups)
    return None


def generate_invalid_mask(groups, near_margin=0.05, angle_margin=0.876, z_margin=0.01):
    """
    Generates a mask for removing groups that does not satisfy all three conditions

    Condition 1: no two points in the group can be closer than near_margin

    Condition 2: no point in the group can have a z-value less than z_margin

    Condition 3: no angle in the triangle defined by the three points in the group
    can be smaller than angle_margin

    :param groups:
    :param near_margin:
    :param angle_margin:
    :param z_margin:
    :return:
    """
    diffs = create_diff_vectors(groups)  # [b, n, 3points, 3xyz] float32

    # Mask points with small cosine angle
    # TODO

    # Mask near points
    # calculate length of diff-vectors
    # len(x, y, z) = sqrt(x^2 + y^2 + z^2)
    diff_sqrd_lengths = tf.reduce_sum(tf.math.square(diffs), axis=-1)    # [b, n, 3points] float32
    near_mask = tf.less(diff_sqrd_lengths, tf.math.square(near_margin))  # [b, n, 3points] boolean
    near_mask = tf.reduce_any(near_mask, axis=-1)                        # [b, n]          boolean

    # Mask invalid points
    # if any z-component in the group is too small
    z_mask = tf.less(groups[:, :, :, 2], z_margin)  # [b, n, 3points] boolean
    z_mask = tf.reduce_any(z_mask, axis=-1)         # [b, n]          boolean
    return tf.logical_or(near_mask, z_mask)


def create_diff_vectors(groups):
    """
    Creates the difference vectors for the points groups,
    making it easier to compute normals

    :param groups: the points groups with shape (b, n, 3points, 3xyz)
    :return: the difference vectors tensor of shape (b, n, 3points, 3xyz)
    """
    diff_01 = tf.expand_dims(tf.subtract(groups[:, :, 0, :], groups[:, :, 1, :]), 2)
    diff_12 = tf.expand_dims(tf.subtract(groups[:, :, 1, :], groups[:, :, 2, :]), 2)
    diff_20 = tf.expand_dims(tf.subtract(groups[:, :, 2, :], groups[:, :, 0, :]), 2)
    return tf.concat((diff_01, diff_12, diff_20), axis=2)


def normalize_vectors(groups):
    """
    Normalizes the length of the vectors in groups so that
    all vectors in groups have the cartesian length of 1

    :param groups: a tensor with shape (b, n, 3points, 3xyz)
    :return: a tensor with normalized vectors (b, n, 3points, 3xyz)
    """
    lengths = tf.math.sqrt(tf.reduce_sum(tf.math.square(groups), axis=-1))
    return tf.divide(groups, lengths)


def depth_to_xyz(depth, focal_lengths, input_shape=(224, 224)):
    """
    Convert depth map to a cartesian point cloud map

    :param depth: a depth map tensor of shape [b, h, w]
    :param focal_lengths: a tuple of focal lengths as (fx, fy)
    :param input_shape: a tuple of input shape as (h, w)
    :return: a tensor of points in cartesian 3-space [b, h, w, 3]
    """
    x = tf.constant([i - (input_shape[0] // 2) for i in range(input_shape[0])])
    x = tf.tile(tf.expand_dims(x, axis=0), (input_shape[1], 1))
    x = tf.multiply(x, depth)
    x = tf.divide(x, focal_lengths[0])

    y = tf.constant([i - (input_shape[1] // 2) for i in range(input_shape[1])])
    y = tf.tile(tf.expand_dims(y, axis=0), (input_shape[0], 1))   # [b, w, h]
    y = tf.transpose(y, perm=(1, 0))  # [b, h, w]
    y = tf.multiply(y, depth)    # [b, h, w]
    y = tf.divide(y, focal_lengths[1])

    z = depth

    x = tf.expand_dims(x, -1)     # [b, h, w, 1]
    y = tf.expand_dims(y, -1)     # [b, h, w, 1]
    z = tf.expand_dims(z, -1)     # [b, h, w, 1]
    p = tf.concat((x, y, z), -1)  # [b, h, w, 3]
    return p


def custom_loss(label, pred):
    print(label.shape)
    print(pred.shape)
    return 1.
