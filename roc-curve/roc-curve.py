import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort descending by score
    desc_idx = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_idx]
    y_true_sorted = y_true[desc_idx]

    # Find indices where score changes (unique threshold boundaries)
    distinct = np.concatenate(([True], np.diff(y_score_sorted) != 0))

    # Cumulative TP and FP at each position
    cum_tp = np.cumsum(y_true_sorted)
    cum_fp = np.cumsum(1 - y_true_sorted)

    # Take values only at distinct threshold boundaries (last occurrence of each tied group)
    # We want the cumulative counts *after* processing all ties at each unique score.
    # distinct marks the first occurrence; we need the last index of each group.
    # The last index of group i is (first index of group i+1) - 1, plus the final index.
    group_starts = np.where(distinct)[0]
    group_ends = np.concatenate((group_starts[1:] - 1, [len(y_score_sorted) - 1]))

    tp = cum_tp[group_ends]
    fp = cum_fp[group_ends]
    thresholds_vals = y_score_sorted[group_starts]

    total_pos = cum_tp[-1]
    total_neg = cum_fp[-1]

    tpr = tp / total_pos if total_pos > 0 else tp * 0.0
    fpr = fp / total_neg if total_neg > 0 else fp * 0.0

    # Prepend the (0, 0, inf) starting point
    fpr = np.concatenate(([0.0], fpr))
    tpr = np.concatenate(([0.0], tpr))
    thresholds = np.concatenate(([np.inf], thresholds_vals))

    return fpr.tolist(), tpr.tolist(), thresholds.tolist()
