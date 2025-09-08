# rf_gcn_eval_plots.py
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

LABELS = r"C:\Users\Finley\OneDrive\Masters code\Data\pfDHFR_cleaned_with_potency_descriptors.csv"
SCORES = r"C:\Users\Finley\OneDrive\Masters code\scored_pfDHFR_rf_gcn.csv"

L = pd.read_csv(LABELS, encoding="utf-8-sig")[["Smiles","active"]].rename(columns={"Smiles":"smiles"})
S = pd.read_csv(SCORES, encoding="utf-8-sig")[["smiles","RF_pred_proba","GCN_pred_proba","combo_mean"]]
df = L.merge(S, on="smiles").dropna()
y = df["active"].astype(int).to_numpy()

def plot_roc(df, y, cols, out):
    plt.figure(figsize=(5,5))
    for c in cols:
        fpr, tpr, _ = roc_curve(y, df[c].to_numpy())
        plt.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curves"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

def plot_pr(df, y, cols, out):
    plt.figure(figsize=(5,5))
    base = y.mean()
    for c in cols:
        prec, rec, _ = precision_recall_curve(y, df[c].to_numpy())
        plt.plot(rec, prec, label=f"{c} (AP={auc(rec,prec):.3f})")
    plt.hlines(base, 0, 1, linestyles="--", label=f"baseline={base:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

cols = ["RF_pred_proba","GCN_pred_proba","combo_mean"]
plot_roc(df, y, cols, r"C:\Users\Finley\OneDrive\Masters code\roc_rf_gcn.png")
plot_pr(df, y, cols, r"C:\Users\Finley\OneDrive\Masters code\pr_rf_gcn.png")
print("Saved roc_rf_gcn.png and pr_rf_gcn.png")
