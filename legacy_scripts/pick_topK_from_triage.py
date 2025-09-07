import argparse, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdmd, Descriptors

def add_props(df):
    # Compute HBD/HBA/RotBonds for rows missing them (from SMILES)
    need = [c for c in ["hbd","hba","rotb"] if c not in df.columns]
    if not need: return df
    HBD, HBA, RB = [], [], []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(s) if isinstance(s,str) else None
        if m is None:
            HBD.append(None); HBA.append(None); RB.append(None)
        else:
            HBD.append(int(rdmd.CalcNumHBD(m)))
            HBA.append(int(rdmd.CalcNumHBA(m)))
            RB.append(int(Descriptors.NumRotatableBonds(m)))
    if "hbd" not in df:  df["hbd"]  = HBD
    if "hba" not in df:  df["hba"]  = HBA
    if "rotb" not in df: df["rotb"] = RB
    return df

def lipinski_ok(r):
    # Use columns present in your triage CSV: mw, clogP, tpsa; and computed hbd/hba if needed
    return (
        (r.get("mw", 0)   <= 500) and
        (r.get("clogP", 0) <= 5) and
        (r.get("hbd", 0)  <= 5) and
        (r.get("hba", 0)  <= 10) and
        (r.get("tpsa", 0) <= 140)
    )

def main(inp, out, k, score_col):
    df = pd.read_csv(inp, encoding="utf-8-sig")
    if score_col not in df.columns:
        # fall back if combo isn’t there
        score_col = "RF_pred_proba" if "RF_pred_proba" in df.columns else df.columns[-1]
    # basic columns expected from your triage: keep, novel_scaffold, pains, smiles, mw,tpsa,clogP, nn_tanimoto_to_train
    if "keep" in df.columns: df = df[df["keep"]==True]
    if "novel_scaffold" in df.columns: df = df[df["novel_scaffold"]==True]
    if "pains" in df.columns: df = df[df["pains"]==False]

    df = add_props(df).dropna(subset=["smiles"])
    df = df[df.apply(lipinski_ok, axis=1)]

    if len(df)==0:
        raise SystemExit("No molecules pass Lipinski/filters; relax filters or check input.")

    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    top = df.head(k).copy()

    cols = [c for c in ["smiles","combo_mean","RF_pred_proba","GCN_pred_proba",
                        "mw","tpsa","clogP","hbd","hba","rotb","pains",
                        "novel_scaffold","nn_tanimoto_to_train"] if c in top.columns]
    top[cols].to_csv(out, index=False, encoding="utf-8")

    print(f"Selected {len(top)} molecules → {out}")
    for i,row in top.iterrows():
        score = row.get(score_col, float("nan"))
        print(f"{i+1:>2}. {row['smiles']}  | {score_col}={score:.3f}  "
              f"MW={row.get('mw','?')}, cLogP={row.get('clogP','?')}, "
              f"HBD={row.get('hbd','?')}, HBA={row.get('hba','?')}, TPSA={row.get('tpsa','?')}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--score-col", default="combo_mean")
    a = ap.parse_args()
    main(a.input, a.output, a.k, a.score_col)
