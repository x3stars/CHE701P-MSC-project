"""Evaluation helpers (ROC/PR/EF)."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def _ef_at_frac(y_true: np.ndarray, scores: np.ndarray, frac=0.05) -> float:
    n = len(y_true); k = max(1, int(round(frac * n)))
    idx = np.argsort(-scores)[:k]
    tp_at_k = y_true[idx].sum()
    hitrate_at_k = tp_at_k / k
    base_rate = y_true.mean()
    return float(hitrate_at_k / base_rate) if base_rate > 0 else np.nan

def eval_from_files(scores_csv: str, labels_csv: str,
                    smiles_col_labels="Smiles", smiles_col_scores="smiles",
                    label_col="active") -> dict:
    labels = pd.read_csv(labels_csv, encoding="utf-8-sig")
    scores = pd.read_csv(scores_csv, encoding="utf-8-sig")
    labels.columns = [c.replace("\\ufeff","").strip() for c in labels.columns]
    scores.columns = [c.replace("\\ufeff","").strip() for c in scores.columns]
    L = labels[[smiles_col_labels, label_col]].rename(columns={smiles_col_labels:"smiles"})
    keep_cols = [c for c in ["RF_pred_proba","GCN_pred_proba","combo_mean"] if c in scores.columns]
    S = scores[[smiles_col_scores, *keep_cols]].rename(columns={smiles_col_scores:"smiles"})
    df = pd.merge(L, S, on="smiles", how="inner").dropna()
    y = df[label_col].astype(int).to_numpy()
    results = {"n_merged": int(df.shape[0]), "prevalence": float(y.mean()), "metrics": {}}
    for c in keep_cols:
        s = df[c].to_numpy()
        roc = roc_auc_score(y, s) if len(np.unique(y))>1 else np.nan
        pr  = average_precision_score(y, s) if len(np.unique(y))>1 else np.nan
        results["metrics"][c] = {"roc_auc": float(roc), "pr_auc": float(pr),
                                 "EF1%": _ef_at_frac(y, s, 0.01), "EF5%": _ef_at_frac(y, s, 0.05)}
    return results
