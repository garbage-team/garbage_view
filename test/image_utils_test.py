import unittest
from src import image_utils
import numpy as np
import tensorflow as tf


class NormalizeTestCase(unittest.TestCase):
    def setUp(self):
        super(NormalizeTestCase, self).setUp()
        self.rgb_norm = np.random.random((256, 256, 3)) * 255
        self.rgb = self.rgb_norm * 255
        self.rgb = np.round(self.rgb).astype('uint8')
        self.d_norm = np.random.random((256, 256, 1))
        self.d = self.d_norm * 4500
        self.d = np.round(self.d).astype('uint16')

    def test_normalize_rgb(self):
        rgb_normalize = image_utils.normalize_rgb(self.rgb)
        self.assertTrue(np.max(rgb_normalize) <= 1.0)
        self.assertTrue(np.min(rgb_normalize) >= 0.0)

    def test_normalize_d(self):
        d_norm = image_utils.normalize_d(self.d)
        d_norm_5000 = image_utils.normalize_d(self.d, max_depth=5000)
        self.assertTrue(np.max(d_norm) <= 1.0)
        self.assertTrue(np.min(d_norm) >= 0.0)
        self.assertTrue(np.max(d_norm_5000) <= 1.0)
        self.assertTrue(np.min(d_norm_5000) >= 0.0)

    def test_un_normalize_rgb(self):
        rgb = image_utils.un_normalize_rgb(self.rgb_norm)
        self.assertTrue(np.max(rgb) <= 255)
        self.assertTrue(np.min(rgb) >= 0)

    def test_un_normalize_d(self):
        d = image_utils.un_normalize_d(self.d_norm, 4500)
        self.assertTrue(np.max(d) <= 4500)
        self.assertTrue(np.min(d) >= 0)


class BinsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.some_bins = tf.random.uniform([224, 224, 1], minval=0, maxval=149, dtype=tf.int32)
        self.some_depth = tf.random.uniform([224, 224, 1], minval=-1., maxval=82., dtype=tf.float32)

    def test_bins_to_depth(self):
        self.assertTrue(True)

    def test_depth_to_bins(self):
        bins = image_utils.depth_to_bins(self.some_depth)
        self.assertTrue(bins.dtype == tf.dtypes.int32)
        self.assertTrue(tf.reduce_max(bins) <= 151)
        self.assertTrue(tf.reduce_min(bins) >= 0)
        # Test that bins is 151 at all depths < 0.25
        min_mask = tf.math.less(self.some_depth, 0.25)
        min_tensor = tf.boolean_mask(bins, min_mask)
        self.assertTrue(tf.reduce_all(tf.equal(min_tensor, 151)))
        # Test that bins is 149 at all depths > 80
        max_mask = tf.math.greater(self.some_depth, 80.)
        max_tensor = tf.boolean_mask(bins, max_mask)
        self.assertTrue(tf.reduce_all(tf.equal(max_tensor, 149)))


if __name__ == '__main__':
    unittest.main()
