import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def model_input_data(train_df,valid_df,feature_cols,target_col="isFraud"):

    #Splitting Target vs Predictors
    X_train=train_df[feature_cols].copy()
    X_valid=valid_df[feature_cols].copy()

    y_train=train_df[target_col].copy()
    y_valid=valid_df[target_col].copy()

    #Seperating Categorical and Numerical Columns
    cat_cols=X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols=X_train.select_dtypes(include="number").columns.tolist()

    #Filling Missing Values
    for col in num_cols:
        X_train[col]=X_train[col].fillna(X_train[col].median())
        X_valid[col]=X_valid[col].fillna(X_valid[col].median())

    for col in cat_cols:
        X_train[col]=X_train[col].fillna("Missing").astype(str)
        X_valid[col]=X_valid[col].fillna("Missing").astype(str)
    
    # Typecasting all strings into category
    for col in cat_cols:
        X_train[col]=X_train[col].astype("category")
        X_valid[col]=X_valid[col].astype("category")
    
    return X_train, X_valid, y_train, y_valid, cat_cols


def model_results(train_df,valid_df,feature_cols,model_name="lgbm"):
    X_train, X_valid, y_train, y_valid, cat_cols=model_input_data(train_df,valid_df,feature_cols)
    model=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=64,subsample=0.8,
                         colsample_bytree=0.8,random_state=42,class_weight="balanced")
    model.fit(X_train, y_train, categorical_feature=cat_cols)

    pred=model.predict_proba(X_valid)[:,1]

    results = {
        "n_features": len(feature_cols),
        "roc_auc": roc_auc_score(y_valid,pred),
        "pr_auc":average_precision_score(y_valid,pred),
        "pred":pred,
        "model":model
    }

    return results