"""Triage & candidate review."""
import os, json, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED, FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold

def add_basic_props(df: pd.DataFrame, smiles_col="smiles"):
    def calc(sm):
        m = Chem.MolFromSmiles(sm)
        if not m: return pd.Series({"MW": np.nan, "logP": np.nan})
        return pd.Series({"MW": Descriptors.MolWt(m), "logP": Descriptors.MolLogP(m)})
    props = df[smiles_col].apply(calc)
    return pd.concat([df, props], axis=1)

def save_rank1_png(df: pd.DataFrame, out_png="figures/pfDHFR_rank1.png",
                   smiles_col="smiles", score_col="Ensemble_pred_proba"):
    from rdkit.Chem import Draw
    row = df.sort_values(score_col, ascending=False).iloc[0] if score_col in df.columns else df.iloc[0]
    m = Chem.MolFromSmiles(row[smiles_col])
    Draw.MolToImage(m, size=(600,450)).save(out_png)
    return float(row[score_col]) if score_col in df.columns else None

def triage_evolved(input_csv: str, output_csv: str, train_csv: str, train_col: str = "Smiles") -> dict:
    """Mirror triage_evolved.py behaviour (filters, novelty vs training, ECFP similarity)."""
    from rdkit import DataStructs
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    def _mol(s): 
        m = Chem.MolFromSmiles(s) if isinstance(s,str) else None
        if m: Chem.SanitizeMol(m)
        return m
    def _scaf(m):
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc else None
    def _ecfp4(m, nBits=2048):
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=nBits)
    def _nearest_tanimoto(train_bvs, q_bv):
        if not train_bvs: return np.nan
        sims = [DataStructs.TanimotoSimilarity(q_bv, tbv) for tbv in train_bvs]
        return float(np.max(sims)) if sims else np.nan

    tr_scaffs=set(); tr_bvs=[]
    if train_csv and os.path.isfile(train_csv):
        tr = pd.read_csv(train_csv, encoding="utf-8-sig")
        if train_col not in tr.columns:
            for c in ["smiles","SMILES","canonical_smiles","can_smiles"]:
                if c in tr.columns: train_col=c; break
        for s in tr[train_col].dropna().astype(str):
            m = _mol(s); 
            if not m: continue
            sc = _scaf(m)
            if sc: tr_scaffs.add(sc)
            tr_bvs.append(_ecfp4(m))

    pains = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams())
    pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)

    rows=[]
    for r in df.itertuples(index=False):
        s = getattr(r, "smiles", None)
        m = _mol(s) if isinstance(s,str) else None
        if not m: 
            continue
        sc = _scaf(m)
        novel = (sc is not None and sc not in tr_scaffs)
        bv = _ecfp4(m)
        nn = _nearest_tanimoto(tr_bvs, bv)
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
    keep = (res["pains"]==False) & (res["mw"].between(150,600)) & (res["tpsa"]<=140) & (res["hbd"]<=5) & (res["hba"]<=12) & (res["rotb"]<=12)
    res["keep"] = keep
    res = res.sort_values(["combo_mean","RF_pred_proba"], ascending=False)
    res.to_csv(output_csv, index=False, encoding="utf-8")
    return {"n": int(len(res)), "kept": int(res.keep.sum()), "output_csv": os.path.abspath(output_csv)}

def review_candidates(input_csv: str, output_csv: str,
                      rf_model: str | None = None,
                      smiles_col: str = "smiles") -> dict:
    """Port of review_candidates.py: parent, props, alerts, optional RF+GCN, ensemble -> sorted CSV."""
    try:
        import joblib
    except Exception:
        joblib = None
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if smiles_col not in df.columns:
        raise KeyError(f"'{smiles_col}' not found in {input_csv}")

    # Parent smiles
    def _largest_parent_smiles(smi: str):
        m = Chem.MolFromSmiles(smi)
        if not m: return None
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
        parent = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        return Chem.MolToSmiles(parent, canonical=True)
    df["parent_smiles"] = df[smiles_col].map(_largest_parent_smiles)

    # Props + alerts
    def _props_row(m):
        return dict(
            MW=Descriptors.MolWt(m),
            cLogP=Crippen.MolLogP(m),
            TPSA=rdMolDescriptors.CalcTPSA(m),
            HBD=rdMolDescriptors.CalcNumHBD(m),
            HBA=rdMolDescriptors.CalcNumHBA(m),
            RotBonds=rdMolDescriptors.CalcNumRotatableBonds(m),
            FormalCharge=int(sum(a.GetFormalCharge() for a in m.GetAtoms())),
            HeavyAtoms=int(m.GetNumHeavyAtoms()),
            QED=float(QED.qed(m)),
        )
    cat = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams())
    cat.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    cat.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    cat.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    cat.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)

    props_rows, pains_counts, brenk_counts, mols = [], [], [], []
    for s in df["parent_smiles"]:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        mols.append(m)
        if m is None:
            props_rows.append({k: np.nan for k in ["MW","cLogP","TPSA","HBD","HBA","RotBonds","FormalCharge","HeavyAtoms","QED"]})
            pains_counts.append(np.nan); brenk_counts.append(np.nan); continue
        props_rows.append(_props_row(m))
        matches = cat.GetMatches(m)
        pains = sum("PAINS" in ma.GetDescription() for ma in matches)
        brenk = sum("Brenk" in ma.GetDescription() for ma in matches)
        pains_counts.append(int(pains)); brenk_counts.append(int(brenk))

    df = pd.concat([df, pd.DataFrame(props_rows)], axis=1)
    df["PAINS_alerts"] = pains_counts
    df["Brenk_alerts"] = brenk_counts
    df["Lipinski_ok"] = df.apply(lambda r: (r["MW"]<=500 and r["cLogP"]<=5 and r["HBD"]<=5 and r["HBA"]<=10), axis=1)
    df["Veber_ok"]    = df.apply(lambda r: (r["TPSA"]<=140 and r["RotBonds"]<=10), axis=1)

    # Optional RF scoring
    if rf_model and joblib and os.path.isfile(rf_model):
        from rdkit.Chem import AllChem, DataStructs
        bits = []
        for m in mols:
            if m is None: bits.append(None); continue
            bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.uint8); DataStructs.ConvertToNumpyArray(bv, arr)
            bits.append(arr)
        rf_scores = np.full(len(df), np.nan, dtype=float)
        X_idx = [i for i,x in enumerate(bits) if x is not None]
        if X_idx:
            X = np.stack([bits[i] for i in X_idx])
            clf = joblib.load(rf_model)
            proba = clf.predict_proba(X)[:,1]
            for i,p in zip(X_idx, proba): rf_scores[i]=float(p)
        df["RF_pred_proba"] = rf_scores

    # Ensemble if GCN added later
    if "RF_pred_proba" in df.columns and "GCN_pred_proba" in df.columns:
        df["combo_mean"] = df[["RF_pred_proba","GCN_pred_proba"]].mean(axis=1)

    score_col = "combo_mean" if "combo_mean" in df.columns else ("RF_pred_proba" if "RF_pred_proba" in df.columns else ("GCN_pred_proba" if "GCN_pred_proba" in df.columns else "QED"))
    df = df.sort_values(score_col, ascending=False)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    keep = df[(df["PAINS_alerts"]==0) & (df["Brenk_alerts"]==0) & df["Lipinski_ok"] & df["Veber_ok"]]
    return {"n": int(len(df)), "scored_by": score_col, "kept_no_alerts_druglike": int(keep.shape[0]), "output_csv": os.path.abspath(output_csv)}
