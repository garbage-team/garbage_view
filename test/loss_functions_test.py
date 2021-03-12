import unittest
import numpy as np
import tensorflow as tf
from src.loss_functions import wcel_loss, virtual_normal_loss
from src.image_utils import depth_to_bins
from src.model import sm_model
from src.data_loader import load_nyudv2


class WCELTest(unittest.TestCase):
    def test_run(self):
        gt_depth = tf.random.uniform((8, 224, 224), 0.25, 80)
        pred_depth_bins = tf.random.uniform((8, 224, 224, 150))
        pred_depth_bins_sm = tf.keras.activations.softmax(pred_depth_bins)
        loss = wcel_loss(gt_depth, pred_depth_bins_sm)
        print(loss)
        self.assertTrue(loss != 0)
        self.assertTrue(loss.dtype == tf.dtypes.float32)


class VNLTest(unittest.TestCase):
    def test_run(self):
        gt_depth = tf.random.uniform(shape=(8, 224, 224), minval=0.25, maxval=80.)
        gt_bins = depth_to_bins(gt_depth)
        one_hot = tf.one_hot(gt_bins, 150)
        no_loss = virtual_normal_loss(gt_depth, one_hot)
        self.assertTrue(no_loss < 0.01)
        pred_softmax = tf.keras.activations.softmax(tf.random.uniform((8, 224, 224, 150)))
        some_loss = virtual_normal_loss(gt_depth, pred_softmax)
        self.assertTrue(some_loss != 0.0)
        tf.debugging.enable_check_numerics()
        model= sm_model()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        model.compile(optimizer=optimizer,
                      loss=virtual_normal_loss,
                      metrics=['accuracy'])
        nyu = load_nyudv2(batch=4)
        for rgb, d in nyu.take(1):
            rgb = rgb
            gt = d

        pred = model.predict(rgb)
        vnl_loss = virtual_normal_loss(gt, pred)
        print(vnl_loss)


if __name__ == '__main__':
    unittest.main()
