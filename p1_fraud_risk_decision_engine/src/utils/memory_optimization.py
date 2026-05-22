import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage."""
    df = df.copy()

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df