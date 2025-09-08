import os, json, math, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix
)

def find_label_col(df, explicit=None):
    if explicit is not None:
        if explicit in df.columns:
            return explicit
        raise KeyError(f"--label '{explicit}' not found. Available columns: {list(df.columns)}")
    # auto-detect common names
    for c in ["active","Active","label","Label","y","activity","is_active"]:
        if c in df.columns:
            return c
    # fallback: derive from potency if present
    for pcol in ["pIC50","pic50","IC50_nM","ic50_nM","IC50","ic50"]:
        if pcol in df.columns:
            s = df[pcol]
            if pcol.lower().startswith("pic"):
                y = (s >= 6.0).astype(int)   # pIC50>=6 (~≤1µM) = active
                df["_derived_label"] = y
                return "_derived_label"
            else:
                y = (s <= 1000).astype(int)  # IC50 ≤ 1000 nM = active
                df["_derived_label"] = y
                return "_derived_label"
    raise KeyError(f"Could not find a label column. Available columns: {list(df.columns)}")

def ef_at_percent(y_true, y_score, pct):
    n = len(y_true)
    top_n = max(1, int(math.ceil(pct * n)))
    idx = np.argsort(-y_score)[:top_n]
    hits = int(np.asarray(y_true)[idx].sum())
    hit_rate = hits / top_n
    prevalence = float(np.mean(y_true))
    ef = hit_rate / prevalence if prevalence > 0 else np.nan
    return {"EF": ef, "hits": hits, "top_n": top_n, "prevalence": prevalence}

def precision_at_k(y_true, y_score, k):
    k = min(k, len(y_true))
    idx = np.argsort(-y_score)[:k]
    return float(np.asarray(y_true)[idx].mean())

def youden_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i])

def plot_confusion(y_true, y_score, thr, title, out_png):
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Inactive","Active"]); ax.set_yticklabels(["Inactive","Active"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    ax.set_title(title + f"\nthreshold={thr:.3f}")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def main(in_csv, out_dir, label_name):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(in_csv, encoding="utf-8-sig")
    label_col = find_label_col(df, explicit=label_name)
    y = df[label_col].astype(int).values

    model_cols = [c for c in ["RF_pred_proba","GCN_pred_proba","combo_mean"] if c in df.columns]
    if not model_cols:
        raise SystemExit("No model score columns found (need at least one of RF_pred_proba / GCN_pred_proba / combo_mean).")

    metrics, ef_rows = {}, []
    for col in model_cols:
        s = df[col].values.astype(float)
        roc = roc_auc_score(y, s)
        pr  = average_precision_score(y, s)
        ef1  = ef_at_percent(y, s, 0.01)
        ef5  = ef_at_percent(y, s, 0.05)
        ef10 = ef_at_percent(y, s, 0.10)
        p100 = precision_at_k(y, s, 100)
        p250 = precision_at_k(y, s, 250)
        thr, tpr, tnr = youden_threshold(y, s)
        cm = plot_confusion(y, s, thr, f"{col} — Confusion matrix", os.path.join(out_dir, f"cm_{col}.png"))
        metrics[col] = {
            "roc_auc": roc, "pr_auc": pr,
            "EF1%": ef1["EF"], "EF5%": ef5["EF"], "EF10%": ef10["EF"],
            "Precision@100": p100, "Precision@250": p250,
            "threshold_youden": thr, "TPR_at_thr": tpr, "TNR_at_thr": tnr,
            "confusion": cm,
        }
        for name, ef in [("EF1%",ef1), ("EF5%",ef5), ("EF10%",ef10)]:
            ef_rows.append({"model": col, "metric": name, "EF": ef["EF"],
                            "hits": ef["hits"], "top_n": ef["top_n"], "prevalence": ef["prevalence"]})

    with open(os.path.join(out_dir, "metrics_extended.json"), "w", encoding="utf-8") as f:
        json.dump({"n": int(len(df)), "label_col": label_col, "metrics": metrics}, f, indent=2)
    pd.DataFrame(ef_rows).to_csv(os.path.join(out_dir, "ef_table.csv"), index=False, encoding="utf-8")

    lines = [f"n={len(df)} | label={label_col} | prevalence={np.mean(y):.3f}"]
    for m in model_cols:
        mm = metrics[m]
        lines.append(f"{m}: ROC-AUC={mm['roc_auc']:.3f} | PR-AUC={mm['pr_auc']:.3f} | "
                     f"EF1%={mm['EF1%']:.2f} EF5%={mm['EF5%']:.2f} EF10%={mm['EF10%']:.2f} | "
                     f"P@100={mm['Precision@100']:.2f} P@250={mm['Precision@250']:.2f} | "
                     f"thr*={mm['threshold_youden']:.3f}")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", required=False, default=None, help="Name of the binary label column (e.g., 'active').")
    a = ap.parse_args()
    main(a.input, a.outdir, a.label)
