import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic features used across baseline notebooks."""
    df = df.copy()

    df["log_TransactionAmt"] = np.log1p(df["TransactionAmt"])

    identity_cols = [c for c in ["DeviceType", "DeviceInfo"] if c in df.columns]
    if identity_cols:
        df["has_identity"] = df[identity_cols].notna().any(axis=1).astype(int)

    return df


def add_velocity_features(
    df: pd.DataFrame,
    entity_col: str,
    windows,
    time_col: str = "TransactionDT",
    amt_col: str = "TransactionAmt",
):
    """Past-only rolling count and amount sum.

    `closed="left"` excludes the current row.
    """
    df = df.copy()
    df["_row_id"] = np.arange(len(df))

    # TransactionDT is seconds, convert to timedelta for rolling windows.
    df["_t"] = pd.to_timedelta(
        pd.to_numeric(df[time_col], errors="coerce").fillna(0),
        unit="s",
    )

    df["_amt"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)

    df = df.sort_values([entity_col, "_t", "_row_id"], na_position="last")
    df = df.set_index("_t")

    g = df.groupby(entity_col, sort=False, dropna=True)

    for window in windows:
        df[f"{entity_col}_cnt_{window}"] = (
            g["_amt"]
            .rolling(window, closed="left")
            .count()
            .reset_index(level=0, drop=True)
        )

        df[f"{entity_col}_amt_sum_{window}"] = (
            g["_amt"]
            .rolling(window, closed="left")
            .sum()
            .reset_index(level=0, drop=True)
        )

    df = df.reset_index().sort_values("_row_id")
    df = df.drop(columns=["_row_id", "_t", "_amt"])

    new_cols = [
        c for c in df.columns
        if c.startswith(f"{entity_col}_cnt_")
        or c.startswith(f"{entity_col}_amt_sum_")
    ]
    df[new_cols] = df[new_cols].fillna(0)

    return df.reset_index(drop=True)