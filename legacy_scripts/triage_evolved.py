import argparse, os, pandas as pd, numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

def mol(s): 
    m = Chem.MolFromSmiles(s)
    if m: Chem.SanitizeMol(m)
    return m

def ecfp4(m, nBits=2048):
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return bv

def scaffold_smiles(m):
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc else None

def pains_catalog():
    p = FilterCatalogParams(); p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(p)

def read_training_scaffolds(path, col="Smiles"):
    scaffs=set()
    if path and os.path.isfile(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        if col not in df.columns:
            for c in ["smiles","SMILES","canonical_smiles","can_smiles"]:
                if c in df.columns: col=c; break
        for s in df[col].dropna().astype(str):
            m = mol(s)
            if m:
                sc = scaffold_smiles(m)
                if sc: scaffs.add(sc)
    return scaffs

def nearest_tanimoto(train_bvs, q_bv):
    if not train_bvs: return np.nan
    sims = [DataStructs.TanimotoSimilarity(q_bv, tbv) for tbv in train_bvs]
    return float(np.max(sims)) if sims else np.nan

def load_train_bvs(path, col="Smiles"):
    bvs=[]
    if path and os.path.isfile(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        if col not in df.columns:
            for c in ["smiles","SMILES","canonical_smiles","can_smiles"]:
                if c in df.columns: col=c; break
        for s in df[col].dropna().astype(str):
            m = mol(s)
            if m: bvs.append(ecfp4(m))
    return bvs

def main(inp, out, train, train_col):
    df = pd.read_csv(inp, encoding="utf-8-sig")
    pains = pains_catalog()
    train_scaffs = read_training_scaffolds(train, train_col)
    train_bvs = load_train_bvs(train, train_col)

    rows=[]
    for r in df.itertuples(index=False):
        s = r.smiles
        m = mol(s)
        if not m:
            continue
        sc = scaffold_smiles(m)
        novel = (sc is not None and sc not in train_scaffs)
        bv = ecfp4(m)
        nn = nearest_tanimoto(train_bvs, bv)

        rows.append({
            "smiles": s,
            "RF_pred_proba": getattr(r, "RF_pred_proba", np.nan),
            "GCN_pred_proba": getattr(r, "GCN_pred_proba", np.nan),
            "combo_mean": getattr(r, "combo_mean", np.nan),
            "mw": Descriptors.MolWt(m),
            "clogP": Crippen.MolLogP(m),
            "tpsa": rdMolDescriptors.CalcTPSA(m),
            "hbd": rdMolDescriptors.CalcNumHBD(m),
            "hba": rdMolDescriptors.CalcNumHBA(m),
            "rotb": rdMolDescriptors.CalcNumRotatableBonds(m),
            "pains": pains.HasMatch(m),
            "scaffold": sc,
            "novel_scaffold": novel,
            "nn_tanimoto_to_train": nn
        })
    res = pd.DataFrame(rows)

    # simple keep filter (very permissive) + rank by combo_mean then RF
    keep = (res["pains"]==False) & (res["mw"].between(150,600)) & (res["tpsa"]<=140) & (res["hbd"]<=5) & (res["hba"]<=12) & (res["rotb"]<=12)
    res["keep"] = keep
    res = res.sort_values(["combo_mean","RF_pred_proba"], ascending=False)
    res.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {out}  (n={len(res)})  kept={int(res.keep.sum())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="rf_gcn_scored_top500.csv")
    ap.add_argument("--output", required=True, help="triaged csv output")
    ap.add_argument("--train", required=True, help="training CSV for novelty (pfDHFR_cleaned_with_potency_descriptors.csv)")
    ap.add_argument("--train-col", default="Smiles")
    a = ap.parse_args()
    main(a.input, a.output, a.train, a.train_col)
