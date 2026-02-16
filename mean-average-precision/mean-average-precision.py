import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    ap_list = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_score = np.asarray(y_score, dtype=np.float64)

        # Sort descending by score
        order = np.argsort(y_score)[::-1]
        rel = y_true[order]

        if k is not None:
            rel = rel[:k]

        R = y_true.sum()  # total relevant (full list, not truncated)
        if R == 0:
            ap_list.append(0.0)
            continue

        # Precision at each relevant rank: cumulative hits / rank position
        hits = np.cumsum(rel)
        ranks = np.arange(1, len(rel) + 1, dtype=np.float64)
        precision_at_k = hits / ranks

        ap = (precision_at_k * rel).sum() / R
        ap_list.append(float(ap))

    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    return (round(mAP, 4), [round(ap, 4) for ap in ap_list])