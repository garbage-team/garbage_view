import numpy as np
import csv
import matplotlib.pyplot as plt
from src.main import load_model
from src.data_loader import load_nyudv2
from src.image_utils import bins_to_depth

# Code for running ablation study
# NOTE: has not been fully tested or used


def evaluate_model(model_path):
    """
    Evaluates a model using common metrics for comparison
    @param model_path: String path to the model
    @return: dictionary with the criteria as the key, and their values
    """
    model = load_model(model_path)
    ds = load_nyudv2(batch=4, shuffle=False, split='validation')
    criteria = {'err_absRel': 0, 'err_squaRel': 0, 'err_rms': 0,
                         'err_silog': 0, 'err_logRms': 0, 'err_silog2': 0,
                         'err_delta1': 0, 'err_delta2': 0, 'err_delta3': 0,
                         'err_log10': 0, 'err_whdr': 0, 'n_pixels': 0}

    for rgb, d in ds:
        pred_bins = model.predict(rgb)
        pred = bins_to_depth(pred_bins)
        criteria = evaluate_error(d, pred, criteria)
    return criteria


def evaluate_error(gt, pred, criteria):
    """
    Calculates various types of error for between ground truth and prediction. The various errors are intended to
    showcase different strengths and weaknesses in the predictions.
    @param gt: Ground truth depth
    @param pred: Predicted depth
    @param criteria: dict with criteria as the keys
    @return: updated criteria dict
    """

    zero_mask = gt > 0
    gt = gt[zero_mask]
    criteria['n_pixels'] += gt.shape[0]
    pred = pred[zero_mask]
    gt_rescaled = gt * 80.
    pred_rescaled = pred * 80.

    # Mean Absolute Relative Error
    rel = np.abs(gt - pred) / gt  # compute errors
    abs_rel_sum = np.sum(rel)
    criteria['err_absRel'] += abs_rel_sum

    # Square Mean Relative Error
    s_rel = ((gt_rescaled - pred_rescaled) * (gt_rescaled - pred_rescaled)) / (gt_rescaled * gt_rescaled)  # compute errors
    squa_rel_sum = np.sum(s_rel)
    criteria['err_squaRel'] += squa_rel_sum

    # Root Mean Square error
    square = (gt_rescaled - pred_rescaled) ** 2
    rms_squa_sum = np.sum(square)
    criteria['err_rms'] += rms_squa_sum

    # Log Root Mean Square error
    log_square = (np.log(gt_rescaled) - np.log(pred_rescaled)) ** 2
    log_rms_sum = np.sum(log_square)
    criteria['err_logRms'] += log_rms_sum

    # Scale invariant error
    diff_log = np.log(pred_rescaled) - np.log(gt_rescaled)
    diff_log_sum = np.sum(diff_log)
    criteria['err_silog'] += diff_log_sum
    diff_log_2 = diff_log ** 2
    diff_log_2_sum = np.sum(diff_log_2)
    criteria['err_silog2'] += diff_log_2_sum

    # Mean log10 error
    log10_sum = np.sum(np.abs(np.log10(gt) - np.log10(pred)))
    criteria['err_log10'] += log10_sum

    # Deltas
    gt_pred = gt_rescaled / pred_rescaled
    pred_gt = pred_rescaled / gt_rescaled
    gt_pred = np.reshape(gt_pred, (1, -1))
    pred_gt = np.reshape(pred_gt, (1, -1))
    gt_pred_gt = np.concatenate((gt_pred, pred_gt), axis=0)
    ratio_max = np.amax(gt_pred_gt, axis=0)

    delta_1_sum = np.sum(ratio_max < 1.25)
    criteria['err_delta1'] += delta_1_sum
    delta_2_sum = np.sum(ratio_max < 1.25 ** 2)
    criteria['err_delta2'] += delta_2_sum
    delta_3_sum = np.sum(ratio_max < 1.25 ** 3)
    criteria['err_delta3'] += delta_3_sum
    return criteria


def select_index(img_size):
    """
    Extract the index of two points, to use instead of conducting the evaluation across the entire image
    @param img_size: The size of the image
    @return: dict with index of two points p1 and p2
    """
    p1 = np.random.choice(img_size, int(img_size * 0.6), replace=False)
    np.random.shuffle(p1)
    p2 = np.random.choice(img_size, int(img_size * 0.6), replace=False)
    np.random.shuffle(p2)

    mask = p1 != p2
    p1 = p1[mask]
    p2 = p2[mask]
    p12_index = {'p1': p1, 'p2': p2}
    return p12_index
