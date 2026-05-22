import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def get_metrics(y_true, y_score) -> dict:
    """Main validation metrics used in notebooks."""
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def score_band_summary(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Fraud rate by score band."""
    df = pd.DataFrame({
        "isFraud": y_true,
        "score": y_score,
    })

    df["score_band"] = pd.qcut(df["score"], q=n_bins, duplicates="drop")

    return (
        df.groupby("score_band", observed=True)
        .agg(
            rows=("isFraud", "size"),
            fraud_rate=("isFraud", "mean"),
            fraud_count=("isFraud", "sum"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )