import tensorflow as tf
from src.loss_functions import depth_to_xyz
from src.image_utils import bins_to_depth
from src.config import cfg


def fill_rate_loss(gt, pred, fov="kinect"):
    """
    Calculates the fill rate loss between two depth maps through squared error

    :param gt: ground truth depth image (tf.float32 [b, h, w])
    :param pred: Predicted depth bins (tf.float32 [b, h, w, c])
    :param fov: a string literal that determines the type of camera used
    :return: the loss
    """
    gt_xyz = depth_to_xyz(gt, fov=fov)
    pred_depth = bins_to_depth(pred)
    pred_xyz = depth_to_xyz(pred_depth, fov=fov)

    lim = tf.constant(cfg["fill_rate_loss_lim"], dtype=tf.float32)
    gt_mask = clip_by_border(gt_xyz, lim=lim)
    # pred_mask = clip_by_border(pred_xyz, lim=lim)

    gt_xyz = tf.multiply(gt_xyz, tf.cast(tf.logical_not(gt_mask), tf.float32))
    pred_xyz = tf.multiply(pred_xyz, tf.cast(tf.logical_not(gt_mask), tf.float32))

    sq_diff = tf.square(tf.subtract(gt_xyz, pred_xyz))
    mse_fr_loss = tf.reduce_mean(sq_diff)

    return mse_fr_loss


def clip_by_border(xyz, lim=tf.constant([[-1, 1], [-1, 1], [-1, 1]], dtype=tf.float32)):
    """
    Creates a mask that has True in all positions where the point lies outside of lim and False
    if the point lies within lim

    :param xyz: a point map (tf.float32 [..., 3])
    :param lim: a border limit (tf.float32 [3, 2])
    :return: an invalid mask where all True values are invalid points
    """
    mask = tf.logical_or(tf.greater(xyz, lim[:, 0]), tf.less(xyz, lim[:, 1]))
    mask = tf.reduce_any(mask, axis=-1)
    return mask
