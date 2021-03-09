import tensorflow as tf
from src.image_utils import depth_to_bins
from src.config import cfg


def wcel_loss(gt, pred):
    # Weighted cross entropy loss function for determining the
    # loss of the predicted logits and the ground truth depth logits
    # param pred: [[b, h, w, 150], [b, h, w, 150]]
    depth_bins = cfg["depth_bins"]
    pred_softmax = pred
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


def custom_loss(label, pred):
    print(label.shape)
    print(pred.shape)
    return 1.
