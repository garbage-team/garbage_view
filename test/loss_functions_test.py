import unittest
import numpy as np
import tensorflow as tf
from src.loss_functions import wcel_loss


class WCELTest(unittest.TestCase):
    def test_run(self):
        gt_depth = tf.random.uniform((8, 224, 224, 1), 0.25, 80)
        pred_depth_bins = tf.random.uniform((8, 224, 224, 150))
        pred_depth_bins_sm = tf.keras.activations.softmax(pred_depth_bins)
        loss = wcel_loss(gt_depth, [pred_depth_bins, pred_depth_bins_sm])
        print(loss)
        self.assertTrue(loss != 0)
        self.assertTrue(loss.dtype == tf.dtypes.float32)


if __name__ == '__main__':
    unittest.main()
