import numpy as np
from sklearn.metrics import precision_recall_curve

def find_best_thresholds(y_true, y_prob, categories):
    """
    Finds probability threshold that maximizes F1 score for each category
    Replaces default 0.5 cutoff for more sparse categories
    """
    thresholds = {} # Maps category to optimal threshold

    for i, cat in enumerate(categories):
        # Tests candidate probabilty thresholds and caclulates F1
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # Finds index where F1 peaks and maps it back to corresponding threshold
        best_idx = np.argmax(f1)
        thresholds[cat] = thresh[best_idx] if best_idx < len(thresh) else 0.5

    return thresholds

def apply_thresholds(y_proba, categories, thresholds):
    """
    Applies new threshold for predicted probabilities for each category
    Returns array with new predictions 
    """
    preds = np.zeros_like(y_proba) # Creates empty matrix of 0s with same shape

    # Applies new threshold
    for i, cat in enumerate(categories):
        preds[:, i] = (y_proba[:, i] >= thresholds[cat]).astype(int)
        
    return preds