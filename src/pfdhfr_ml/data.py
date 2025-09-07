"""Data loading, curation, and column aliasing."""
import pandas as pd

def alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "smiles" and c != "smiles": rename[c] = "smiles"
        if lc in {"y","label","activity","active"} and c != "label": rename[c] = "label"
        if "ensemble" in lc and ("proba" in lc or "prob" in lc or "score" in lc): rename[c] = "Ensemble_pred_proba"
    return df.rename(columns=rename)
