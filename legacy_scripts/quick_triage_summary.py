import argparse, pandas as pd, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); a = ap.parse_args()
df = pd.read_csv(a.csv, encoding="utf-8-sig")

print("Rows:", len(df))
print("Keep (passes filters):", int(df['keep'].sum()))
print("Novel scaffolds:", int(df['novel_scaffold'].sum()))
q = df['nn_tanimoto_to_train'].quantile([0.1,0.25,0.5,0.75,0.9], interpolation="linear")
print("NN tanimoto quantiles:\n", q.to_string())

cols = ["smiles","combo_mean","RF_pred_proba","GCN_pred_proba",
        "keep","novel_scaffold","nn_tanimoto_to_train","mw","tpsa","clogP","pains"]
print("\nTop 15 (key columns):")
print(df[cols].head(15).to_string(index=False))
