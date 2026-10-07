import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Computes precision, recall, f1, and support for each category sorted by support
def compute_base_metrics(y_true, y_pred, categories):
    # Computes metrics
    precision, recall, fl, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Returns df with metrics sorted by support
    report_df = pd.DataFrame({
        "category": categories,
        "precision": precision.round(5),
        "recall": recall.round(5),
        "f1": fl.round(5),
        "support": support,
    })

    return report_df.sort_values("support", ascending=False).reset_index(drop=True)