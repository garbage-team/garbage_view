import tensorflow as tf
from src.loss_functions import depth_to_xyz
from src.image_utils import bins_to_depth
from src.config import cfg


def actual_fill_rate_loss(gt, pred, fov="kinect", z_zero=1.3):
    """
    Calculates the fill rate loss between two depth maps through fill rate error

    :param gt: ground truth depth image (tf.float32 [b, h, w])
    :param pred: Predicted depth bins (tf.float32 [b, h, w, c])
    :param fov: a string literal that determines the type of camera used (string)
    :param z_zero: distance to top of container (float)
    :return: the loss (tf.float32 [,])
    """
    gt_xyz = depth_to_xyz(gt, fov=fov)
    pred_depth = bins_to_depth(pred)
    pred_xyz = depth_to_xyz(pred_depth, fov=fov)

    lim = tf.constant(cfg["fill_rate_loss_lim"], dtype=tf.float32)
    gt_mask = clip_by_border(gt_xyz, lim=lim)
    print(tf.reduce_sum(tf.cast(gt_mask, tf.int32)))
    print(tf.reduce_sum(tf.ones_like(tf.cast(gt_mask, tf.int32))))
    # pred_mask = clip_by_border(pred_xyz, lim=lim)

    gt_xyz = z_zero - gt_xyz
    pred_xyz = z_zero - pred_xyz

    gt_xyz = tf.multiply(gt_xyz, tf.expand_dims(tf.cast(tf.logical_not(gt_mask), tf.float32), axis=-1))
    pred_xyz = tf.multiply(pred_xyz, tf.expand_dims(tf.cast(tf.logical_not(gt_mask), tf.float32), axis=-1))

    indices = tf.constant([[[0, 0], [1, 0], [0, 1]],
                           [[0, 1], [1, 0], [1, 1]]],
                          dtype=tf.int32)  # [2, 3, 3] int32
    # [2, 3, 3] -> [(224-1)*(224-1)*2, 3, 3]
    indices = tf.tile(indices, (223, 1, 1))
    x = tf.constant([i // 2 for i in range(223 * 2)], dtype=tf.int32)  # [446]
    a = indices[:, :, 0] + tf.tile(tf.expand_dims(x, axis=-1), (1, 3))  # [446, 3] + [446, 3]
    a = tf.tile(a, (223, 1))

    indices = tf.tile(indices, (223, 1, 1))
    x = tf.constant([i // (223 * 2) for i in range(223 * 223 * 2)], dtype=tf.int32)
    b = indices[:, :, 1] + tf.tile(tf.expand_dims(x, axis=-1), (1, 3))

    indices = tf.stack([a, b], axis=-1)
    indices = tf.tile(tf.expand_dims(indices, axis=0), (gt.shape[0], 1, 1, 1))
    gt_triangles = tf.gather_nd(gt_xyz, indices, batch_dims=1)      # [b, (223*223*2), 3(points), 3(xyz)]
    pred_triangles = tf.gather_nd(pred_xyz, indices, batch_dims=1)  # [b, (223*223*2), 3(points), 3(xyz)]

    gt_heights = tf.reduce_mean(gt_triangles[:, :, :, 2], axis=-1)      # [b, (223*223*2),]
    pred_heights = tf.reduce_mean(pred_triangles[:, :, :, 2], axis=-1)  # [b, (223*223*2),]

    gt_areas = (((gt_triangles[:, :, 1, 0] - gt_triangles[:, :, 0, 0]) *
                 (gt_triangles[:, :, 2, 1] - gt_triangles[:, :, 0, 1])) -
                ((gt_triangles[:, :, 2, 0] - gt_triangles[:, :, 0, 0]) *
                 (gt_triangles[:, :, 1, 1] - gt_triangles[:, :, 0, 1])))
    gt_areas = tf.abs(0.5 * gt_areas)

    pred_areas = (((pred_triangles[:, :, 1, 0] - pred_triangles[:, :, 0, 0]) *
                   (pred_triangles[:, :, 2, 1] - pred_triangles[:, :, 0, 1])) -
                  ((pred_triangles[:, :, 2, 0] - pred_triangles[:, :, 0, 0]) *
                   (pred_triangles[:, :, 1, 1] - pred_triangles[:, :, 0, 1])))
    pred_areas = tf.abs(0.5 * pred_areas)

    gt_volumes = tf.multiply(gt_heights, gt_areas)
    pred_volumes = tf.multiply(pred_heights, pred_areas)

    return tf.abs(tf.reduce_sum(gt_volumes) - tf.reduce_sum(pred_volumes))


def approximate_fill_rate_loss(gt, pred, fov="kinect"):
    """
    Calculates the approximate fill rate loss between two depth
    maps through squared error

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
    mask = tf.logical_or(tf.less(xyz, lim[:, 0]), tf.greater(xyz, lim[:, 1]))
    mask = tf.reduce_any(mask, axis=-1)
    return mask
