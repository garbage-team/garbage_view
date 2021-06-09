import unittest
import tensorflow as tf
from src.evaluation import evaluate_error


class MyTestCase(unittest.TestCase):
    def test_error(self):
        gt1 = tf.random.uniform((224, 224), 0.25, 10)
        gt2 = tf.random.uniform((224, 224), 0.25, 10)
        pred1 = gt1 + 0.01
        pred2 = gt2 + 0.01
        criteria = {'err_absRel': 0, 'err_squaRel': 0, 'err_rms': 0,
                    'err_silog': 0, 'err_logRms': 0, 'err_silog2': 0,
                    'err_delta1': 0, 'err_delta2': 0, 'err_delta3': 0,
                    'err_log10': 0, 'err_whdr': 0, 'n_pixels': 0}
        criteria = evaluate_error(gt1, pred1, criteria)
        criteria = evaluate_error(gt2, pred2, criteria)
        for key in criteria:
            if not key == 'n_pixels':
                criteria[key] = criteria[key] / criteria['n_pixels']

        self.assertNotEqual(criteria['err_rms'], 0)


if __name__ == '__main__':
    unittest.main()
