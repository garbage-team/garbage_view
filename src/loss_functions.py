import tensorflow as tf
from src.image_utils import depth_to_bins


def wcel_loss(gt, pred):
    # Weighted cross entropy loss function for determining the
    # loss of the predicted logits and the ground truth depth logits
    # param pred: [[b, h, w, 1], [b, h, w, 1]]
    depth_bins = 150
    pred_logits, pred_softmax = pred
    gt_bins = depth_to_bins(gt)
    pred_logsoft = tf.math.log(pred_softmax)
    pred_logsoft = tf.transpose(pred_logsoft, perm=[0, 2, 1, 3])  # [b, h, w, 150] -> [b, w, h, 150]
    valid_mask = tf.logical_not(tf.equal(gt_bins, 151))
    # gt_bins = tf.math.multiply(gt_bins, tf.cast(valid_mask, gt_bins.dtype))
    one_hot = tf.one_hot(gt_bins, 150)  # [b, h, w, 1] -> [b, h, w, 150]
    weights = tf.ones_like(one_hot)  # Here, custom code for weights can be added
    weights = tf.math.multiply(one_hot, weights)
    pred_losses = tf.math.multiply(weights, pred_logsoft)
    pred_losses = tf.math.multiply(pred_losses, tf.cast(valid_mask, pred_losses.dtype))
    return None


def custom_loss(label, pred):
    print(label.shape)
    print(pred.shape)
    return 1.
