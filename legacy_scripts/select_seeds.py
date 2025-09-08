import os, pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

inp   = r"C:\Users\Finley\OneDrive\Masters code\scored_pfDHFR_rf_gcn_labeled.csv"
outd  = r"C:\Users\Finley\OneDrive\Masters code\artifacts\selection"
os.makedirs(outd, exist_ok=True)

df = pd.read_csv(inp, encoding="utf-8-sig")
assert "smiles" in df.columns and "RF_pred_proba" in df.columns, f"Missing needed columns in {inp}"

score_col = "RF_pred_proba"
thr = 0.506  # from your evaluation summary

# 1) All predicted actives (hit list)
hit = df[df[score_col] >= thr].copy()
hit = hit[["smiles", score_col]].sort_values(score_col, ascending=False)
hit_path = os.path.join(outd, "hitlist_rf.csv")
hit.to_csv(hit_path, index=False)
print(f"Wrote {hit_path}  (n={len(hit)})")

# 2) Top-500 by RF score
top = df.sort_values(score_col, ascending=False).head(500).copy()
top_path = os.path.join(outd, "top500_rf.csv")
top[["smiles", score_col]].to_csv(top_path, index=False)
print(f"Wrote {top_path}  (n={len(top)})")

# 3) Scaffold-diverse seeds from the top set (Bemis–Murcko)
def murcko(smi: str):
    m = Chem.MolFromSmiles(smi)
    if not m: 
        return None
    scaff = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(scaff, canonical=True) if scaff else None

top["murcko"] = top["smiles"].apply(murcko).fillna("NO_SCAFFOLD")
best_per_scaff = (top.sort_values(score_col, ascending=False)
                    .drop_duplicates(subset=["murcko"]))
seeds = best_per_scaff.head(200)["smiles"].tolist()

seeds_path = os.path.join(outd, "seeds_murcko_top200.smi")
with open(seeds_path, "w", encoding="utf-8") as f:
    for s in seeds: f.write(s + "\n")
print(f"Wrote {seeds_path}  (n={len(seeds)})")
