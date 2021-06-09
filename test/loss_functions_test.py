import unittest
import numpy as np
import struct
import tensorflow as tf
from src.loss_functions import wcel_loss, virtual_normal_loss
from src.fill_rate_loss import actual_fill_rate_loss
from src.image_utils import depth_to_bins, bins_to_depth, resize_normalize
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


class FRLTest(unittest.TestCase):
    def test_run(self):
        d_file = "C:/Users/Victor/Documents/Github/garbage_view/data/train/data_0/000345.raw"
        with open(d_file, "rb") as file:
            d_img = file.read()
        d_img = np.array(struct.unpack("H" * 480 * 640, d_img), dtype='uint16').reshape((480, 640, 1))
        d_img = tf.expand_dims(d_img, axis=0)
        _, gt_depth = resize_normalize(d_img, d_img)
        gt_depth = gt_depth[:, :, :, 0] / 1000
        #gt_depth = tf.random.uniform(shape=(8, 224, 224), minval=0.25, maxval=3.)
        gt_bins = depth_to_bins(gt_depth)
        one_hot = tf.one_hot(gt_bins, 150)
        gt_depth = bins_to_depth(one_hot)
        no_loss = actual_fill_rate_loss(gt_depth, one_hot)
        print(no_loss)
        self.assertTrue(no_loss == 0.)
        pred_depth = gt_depth + 0.01
        pred_bins = depth_to_bins(pred_depth)
        one_hot = tf.one_hot(pred_bins, 150)
        some_loss = actual_fill_rate_loss(gt_depth, one_hot)
        print(some_loss)
        self.assertTrue(some_loss != 0.0)
        


if __name__ == '__main__':
    unittest.main()
