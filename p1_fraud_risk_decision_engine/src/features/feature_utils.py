import numpy as np
import pandas as pd


def add_history_feature(df,entity_col,amt_col="TransactionAmt",time_col="TransactionDT",prefix="card1"):
    
    df = df.copy()
    
    # Sort data by time
    df = df.sort_values(time_col)

    # Dynamic column names using prefix
    cnt_col = f"{prefix}_cnt_txn_hist"
    avg_col = f"{prefix}_avg_txn_amt_hist"
    time_diff_col = f"{prefix}_time_secs_since_last_txn_hist"

    # Initialize columns
    df[cnt_col] = 0
    df[avg_col] = 0
    df[time_diff_col] = 0

    for entity, group in df.groupby(entity_col):
        past_amounts = []
        last_time = None

        for i in group.index:
            # Count of past transactions
            df.loc[i, cnt_col] = len(past_amounts)

            # Average of past amounts
            if past_amounts:
                df.loc[i, avg_col] = sum(past_amounts) / len(past_amounts)
            else:
                df.loc[i, avg_col] = None

            # Time since last transaction
            if last_time is not None:
                df.loc[i, time_diff_col] = df.loc[i, time_col] - last_time
            else:
                df.loc[i, time_diff_col] = None

            # Update trackers
            amt = df.loc[i, amt_col]
            time = df.loc[i, time_col]

            past_amounts.append(amt)
            last_time = time

    return df


def add_velocity_features(df, entity_col, windows, time_col="TransactionDT", amt_col="TransactionAmt", prefix=None):

    df = df.copy()
    prefix = prefix or entity_col

    #Coverting timestamp to ns so that pandas recognizes 1h, 24h kind of windows
    df["t_s"] = pd.to_datetime(df[time_col], unit="s").astype("datetime64[ns]")

    # Sort chronologically first and reset index to guarantee proper array alignment
    df = df.sort_values("t_s").reset_index(drop=True)

    # Issue: Timestamp can duplicate so we need to create oue own fix
    # Add a tiny, strictly increasing nanosecond offset to each row. (Ex. row one gets 1ns , second get 2ns etc)
    # This makes the DatetimeIndex 100% unique and strictly monotonic.
    # It satisfies all Pandas constraints instantly and prevents duplicate axis crashes.
    unique_offsets = pd.to_timedelta(np.arange(len(df)), unit="ns")
    df["t_s"] = df["t_s"] + unique_offsets

    # Set datetime index (Strictly required by Pandas for closed="left")
    df = df.set_index("t_s")

    df_g = df.groupby(entity_col, sort=False)

    for w in windows:
        roll = df_g[amt_col].rolling(w, closed="left")
        
        # Because the index is perfectly unique, Pandas aligns this without throwing errors
        df[f"{prefix}_cnt_{w}"] = roll.count().reset_index(level=0, drop=True)
        df[f"{prefix}_sum_{w}"] = roll.sum().reset_index(level=0, drop=True)

    # Restore original integer index (this automatically drops the t_s index)
    df = df.reset_index(drop=True)
    return df