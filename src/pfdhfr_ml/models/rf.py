"""Random Forest helpers."""
import joblib, pandas as pd

def load_rf(path:str):
    return joblib.load(path)

def top_feature_importances(rf, topn=25) -> pd.Series:
    imp = getattr(rf, "feature_importances_", None)
    if imp is None:
        raise AttributeError("RF model has no feature_importances_.")
    return pd.Series(imp, index=[f"FP_{i}" for i in range(len(imp))]).sort_values(ascending=False).head(topn)
