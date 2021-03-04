import unittest
from src import image_utils
import numpy as np


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


if __name__ == '__main__':
    unittest.main()
