import argparse, os, sys, json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED, FilterCatalog

try:
    import joblib
except Exception:
    joblib = None

try:
    from gcn_molscore_sf import GCNDeepChemScorer
except Exception:
    GCNDeepChemScorer = None

def largest_parent_smiles(smi: str):
    m = Chem.MolFromSmiles(smi)
    if not m: return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    parent = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(parent, canonical=True)

def ecfp4_bitvect(mol, nBits=2048):
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def nearest_active_tanimoto(mol, actives_fps):
    if mol is None or not actives_fps: return np.nan
    from rdkit import DataStructs
    fp = ecfp4_bitvect(mol)
    return float(max(DataStructs.BulkTanimotoSimilarity(fp, actives_fps))) if actives_fps else np.nan

def build_filter_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)

def add_props(m):
    return dict(
        MW = Descriptors.MolWt(m),
        cLogP = Crippen.MolLogP(m),
        TPSA = rdMolDescriptors.CalcTPSA(m),
        HBD = rdMolDescriptors.CalcNumHBD(m),
        HBA = rdMolDescriptors.CalcNumHBA(m),
        RotBonds = rdMolDescriptors.CalcNumRotatableBonds(m),
        FormalCharge = int(sum(a.GetFormalCharge() for a in m.GetAtoms())),
        HeavyAtoms = int(m.GetNumHeavyAtoms()),
        QED = float(QED.qed(m))
    )

def within_lipinski(row):
    return (row["MW"] <= 500 and row["cLogP"] <= 5 and row["HBD"] <= 5 and row["HBA"] <= 10)

def within_veber(row):
    return (row["TPSA"] <= 140 and row["RotBonds"] <= 10)

def main(inp, out_csv, rf_model=None, smiles_col="smiles", train_csv=None, train_smiles_col="Smiles", train_label_col="active", try_gcn=True):
    df = pd.read_csv(inp, encoding="utf-8-sig")
    if smiles_col not in df.columns:
        raise KeyError(f"'{smiles_col}' column not found in {inp}. Found: {list(df.columns)}")

    df["parent_smiles"] = df[smiles_col].map(largest_parent_smiles)

    cat = build_filter_catalog()
    pains_counts, brenk_counts, props_rows, mols = [], [], [], []
    for s in df["parent_smiles"]:
        m = Chem.MolFromSmiles(s) if isinstance(s,str) else None
        mols.append(m)
        if m is None:
            props_rows.append({k: np.nan for k in ["MW","cLogP","TPSA","HBD","HBA","RotBonds","FormalCharge","HeavyAtoms","QED"]})
            pains_counts.append(np.nan); brenk_counts.append(np.nan)
            continue
        props_rows.append(add_props(m))
        matches = cat.GetMatches(m)
        pains = sum("PAINS" in ma.GetDescription() for ma in matches)
        brenk = sum("Brenk" in ma.GetDescription() for ma in matches)
        pains_counts.append(int(pains)); brenk_counts.append(int(brenk))

    props_df = pd.DataFrame(props_rows)
    df = pd.concat([df, props_df], axis=1)
    df["PAINS_alerts"] = pains_counts
    df["Brenk_alerts"] = brenk_counts
    df["Lipinski_ok"] = df.apply(within_lipinski, axis=1)
    df["Veber_ok"] = df.apply(within_veber, axis=1)

    actives_fps = []
    if train_csv and os.path.isfile(train_csv):
        tr = pd.read_csv(train_csv, encoding="utf-8-sig")
        if train_label_col in tr.columns and train_smiles_col in tr.columns:
            actives = tr[tr[train_label_col]==1][train_smiles_col].dropna().unique().tolist()
            actives_mols = [Chem.MolFromSmiles(s) for s in actives]
            actives_mols = [m for m in actives_mols if m is not None]
            actives_fps = [ecfp4_bitvect(m) for m in actives_mols]
    df["NN_Tanimoto_to_active"] = [nearest_active_tanimoto(m, actives_fps) for m in mols] if actives_fps else np.nan

    if rf_model and joblib and os.path.isfile(rf_model):
        clf = joblib.load(rf_model)
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        fp_bits = []
        for m in mols:
            if m is None:
                fp_bits.append(None); continue
            bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(bv, arr)
            fp_bits.append(arr)
        X = np.stack([x for x in fp_bits if x is not None]) if any(x is not None for x in fp_bits) else None
        rf_scores = np.full(len(df), np.nan, dtype=float)
        if X is not None:
            idxs = [i for i,x in enumerate(fp_bits) if x is not None]
            proba = clf.predict_proba(X)[:,1]
            for i,p in zip(idxs, proba): rf_scores[i]=float(p)
        df["RF_pred_proba"] = rf_scores

    if try_gcn and GCNDeepChemScorer is not None:
        try:
            gcn_dir_guess = os.path.join(os.path.dirname(inp), "artifacts", "metrics_gcn")
            scorer = GCNDeepChemScorer(prefix="GCN", gcn_dir=gcn_dir_guess if os.path.isdir(gcn_dir_guess) else None)
            gcn_out = scorer([s if isinstance(s,str) else "" for s in df["parent_smiles"]])
            sm2p = {d["smiles"]: d.get("GCN_pred_proba", np.nan) for d in gcn_out}
            df["GCN_pred_proba"] = [sm2p.get(s, np.nan) for s in df["parent_smiles"]]
        except Exception:
            pass

    if "RF_pred_proba" in df.columns and "GCN_pred_proba" in df.columns:
        df["combo_mean"] = df[["RF_pred_proba","GCN_pred_proba"]].mean(axis=1)

    score_col = "combo_mean" if "combo_mean" in df.columns else ("RF_pred_proba" if "RF_pred_proba" in df.columns else ("GCN_pred_proba" if "GCN_pred_proba" in df.columns else "QED"))
    df = df.sort_values(score_col, ascending=False)

    df.to_csv(out_csv, index=False)
    keep = df[(df["PAINS_alerts"]==0) & (df["Brenk_alerts"]==0) & df["Lipinski_ok"] & df["Veber_ok"]]
    print(json.dumps({
        "n": len(df),
        "scored_by": score_col,
        "kept_no_alerts_druglike": int(keep.shape[0]),
        "median_QED": float(df["QED"].median(skipna=True)),
        "median_cLogP": float(df["cLogP"].median(skipna=True)),
        "median_TPSA": float(df["TPSA"].median(skipna=True))
    }, indent=2))
    print(f"Wrote {os.path.abspath(out_csv)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rf-model")
    ap.add_argument("--train-csv")
    ap.add_argument("--smiles-col", default="smiles")
    args = ap.parse_args()
    main(args.input, args.output, rf_model=args.rf_model, smiles_col=args.smiles_col,
         train_csv=args.train_csv)
