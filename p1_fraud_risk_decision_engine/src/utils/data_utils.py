import pandas as pd

def time_split(df: pd.DataFrame, time_col: str = "TransactionDT", fraction: float=0.2):
    # Ensure the dataframe is sorted by time before splitting
    df = df.sort_values(time_col).reset_index(drop=True)

    split_index=int(len(df)*(1-fraction)) #get train end index
    
    #train will start from zero and go till split index
    train_df=df.iloc[:split_index].copy()
    valid_df=df.iloc[split_index:].copy()
    #Copy is created so that any changes in df will not impact change in train_df or valid_df

    return train_df, valid_df