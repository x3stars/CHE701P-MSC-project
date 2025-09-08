# rf_gcn_eval_from_scores.py
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def ef_at_frac(y_true, scores, frac=0.05):
    n = len(y_true)
    k = max(1, int(round(frac * n)))
    idx = np.argsort(-scores)[:k]               # top-k by score
    tp_at_k = y_true[idx].sum()
    hitrate_at_k = tp_at_k / k
    base_rate = y_true.mean()
    return float(hitrate_at_k / base_rate) if base_rate > 0 else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",  required=True)  # scored_pfDHFR_rf_gcn.csv
    ap.add_argument("--labels",  required=True)  # pfDHFR_cleaned_with_potency_descriptors.csv
    ap.add_argument("--smiles-col-labels", default="Smiles")
    ap.add_argument("--smiles-col-scores", default="smiles")
    ap.add_argument("--label-col", default="active")
    ap.add_argument("--out", default="rf_gcn_eval_report.json")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    scores = pd.read_csv(args.scores, encoding="utf-8-sig")
    # clean possible BOMs
    labels.columns = [c.replace("\ufeff","").strip() for c in labels.columns]
    scores.columns = [c.replace("\ufeff","").strip() for c in scores.columns]

    # merge on SMILES
    L = labels[[args.smiles_col_labels, args.label_col]].rename(columns={args.smiles_col_labels:"smiles"})
    S = scores[[args.smiles_col_scores, "RF_pred_proba", "GCN_pred_proba", "combo_mean"]].rename(columns={args.smiles_col_scores:"smiles"})
    df = pd.merge(L, S, on="smiles", how="inner").dropna()

    y  = df[args.label_col].astype(int).to_numpy()
    cols = ["RF_pred_proba","GCN_pred_proba","combo_mean"]

    results = {
        "n_merged": int(df.shape[0]),
        "prevalence": float(y.mean()),
        "metrics": {}
    }
    for c in cols:
        s = df[c].to_numpy()
        roc = roc_auc_score(y, s) if len(np.unique(y))>1 else np.nan
        pr  = average_precision_score(y, s) if len(np.unique(y))>1 else np.nan
        ef1 = ef_at_frac(y, s, 0.01)
        ef5 = ef_at_frac(y, s, 0.05)
        results["metrics"][c] = {
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "EF1%": ef1,
            "EF5%": ef5
        }

    # save + pretty print
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
if __name__ == "__main__":
    main()
