import unittest
import numpy as np
import tensorflow as tf
from src.loss_functions import wcel_loss, virtual_normal_loss
from src.image_utils import depth_to_bins


class WCELTest(unittest.TestCase):
    def test_run(self):
        gt_depth = tf.random.uniform((8, 224, 224, 1), 0.25, 80)
        pred_depth_bins = tf.random.uniform((8, 224, 224, 150))
        pred_depth_bins_sm = tf.keras.activations.softmax(pred_depth_bins)
        loss = wcel_loss(gt_depth, [pred_depth_bins, pred_depth_bins_sm])
        print(loss)
        self.assertTrue(loss != 0)
        self.assertTrue(loss.dtype == tf.dtypes.float32)


class VNLTest(unittest.TestCase):
    def test_run(self):
        gt_depth = tf.random.uniform(shape=(8, 224, 224), minval=0.25, maxval=80.)
        no_loss = virtual_normal_loss(gt_depth, depth_to_bins(gt_depth))
        self.assertTrue(no_loss == 0.0)
        pred_softmax = tf.keras.activations.softmax(tf.random.uniform((8, 224, 224, 150)))
        some_loss = virtual_normal_loss(gt_depth, pred_softmax)
        self.assertTrue(some_loss != 0.0)


if __name__ == '__main__':
    unittest.main()
