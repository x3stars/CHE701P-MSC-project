# scripts/qc_and_filter_generated.py  (REV=ultra-compat-no-standardize v2)
print("[QC] Loaded qc_and_filter_generated.py (REV=v2) from:", __file__)

import argparse, os, sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors as Desc, rdMolDescriptors as rdDesc, QED

# Optional SA score; ok if missing
try:
    from rdkit.Chem import SA_Score as sascorer
    HAVE_SA = True
except Exception:
    HAVE_SA = False

ALLOWED_ELEMENTS = {"H","C","N","O","S","F","Cl","Br","I"}  # add "P" if you want to allow phosphorus

def largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    return max(frags, key=lambda x: x.GetNumAtoms())

def clean_parent_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "bad_smiles"
    mol = largest_fragment(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "sanitize_fail"
    for a in mol.GetAtoms():
        if a.GetSymbol() not in ALLOWED_ELEMENTS:
            return None, f"disallowed_element:{a.GetSymbol()}"
    if len(Chem.GetMolFrags(mol, asMols=False)) > 1:
        return None, "multi_fragment"
    total_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if total_charge not in (-1, 0, 1):
        return None, f"weird_charge:{total_charge}"
    for a in mol.GetAtoms():
        if a.GetSymbol() == "B" and a.GetFormalCharge() != 0:
            return None, "boron_cation"
    return mol, None

def physchem(mol):
    return {
        "MW": float(Desc.MolWt(mol)),
        "cLogP": float(rdDesc.CalcCrippenDescriptors(mol)[0]),
        "HBD": int(rdDesc.CalcNumHBD(mol)),
        "HBA": int(rdDesc.CalcNumHBA(mol)),
        "TPSA": float(rdDesc.CalcTPSA(mol)),
        "RB": int(rdDesc.CalcNumRotatableBonds(mol)),
        "QED": float(QED.qed(mol)),
    }

def lipinski_ok(p):
    return (p["MW"] <= 500 and p["cLogP"] <= 5 and p["HBD"] <= 5 and p["HBA"] <= 10)

def sa_score(mol):
    if not HAVE_SA:
        return None
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None

def main(inp, out_csv, rf_model, gcn_dir, smiles_col="smiles", top_k=50):
    # Make project root + scripts importable
    here = os.path.dirname(__file__) or "."
    proj = os.path.dirname(here)
    for p in (here, proj):
        if p not in sys.path:
            sys.path.append(p)

    import joblib
    from rf_scorer import ecfp4_vec
    try:
        from gcn_molscore_sf import GCNDeepChemScorer
    except Exception as e:
        raise ImportError(
            "Could not import GCNDeepChemScorer. Place gcn_molscore_sf.py in project root or scripts.\n"
            f"Original error: {e}"
        )

    df = pd.read_csv(inp, encoding="utf-8-sig")
    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found in {inp}")

    rf = joblib.load(rf_model)
    gcn = GCNDeepChemScorer(prefix="GCN", gcn_dir=gcn_dir)

    keep_rows, rejected = [], []

    for smi in df[smiles_col].astype(str):
        mol, why = clean_parent_from_smiles(smi)
        if mol is None:
            rejected.append({"smiles": smi, "reason": why})
            continue
        props = physchem(mol)
        props["Lipinski_ok"] = lipinski_ok(props)
        sa = sa_score(mol)
        if sa is not None and sa > 6.0:
            rejected.append({"smiles": smi, "reason": f"SA>{sa:.2f}"})
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        keep_rows.append({"smiles": can, **props, "SA": sa})

    clean = pd.DataFrame(keep_rows).drop_duplicates(subset=["smiles"])
    if clean.empty:
        raise SystemExit("No molecules passed feasibility filters.")

    # RF scores
    X = np.asarray([ecfp4_vec(s) for s in clean["smiles"]], dtype=float)
    clean["RF_pred_proba"] = rf.predict_proba(X)[:, 1]

    # GCN scores
    gcn_out = gcn(clean["smiles"].tolist())
    gcn_scores = [d.get("GCN_pred_proba", np.nan) for d in gcn_out]
    # --- ChatGPT patch start ---
    if a.top_k is not None and a.top_k > 0:
    _k = min(top_k, len(clean))
    top = clean.nlargest(_k, "RF_pred_proba").copy()
    top["GCN_pred_proba"] = np.asarray(gcn_scores, dtype=float)
    out_df = top
else:
    try:
\ \ \ \ target_df\ =\ out_df
except\ NameError:
\ \ \ \ target_df\ =\ clean
if\ len\(gcn_scores\)\ ==\ 0:
\ \ \ \ pass
elif\ len\(gcn_scores\)\ ==\ len\(target_df\):
\ \ \ \ target_df\["GCN_pred_proba"]\ =\ np\.asarray\(gcn_scores,\ dtype=float\)
else:
\ \ \ \ print\(f"\[WARN]\ GCN\ scores\ length\ \{len\(gcn_scores\)}\ !=\ target_df\ rows\ \{len\(target_df\)};\ skipping\ GCN\ column\."\)
    out_df = clean
# --- ChatGPT patch end ---

    clean["combo_mean"] = 0.5 * (clean["RF_pred_proba"] + clean["GCN_pred_proba"])
    clean.sort_values(["Lipinski_ok", "SA", "combo_mean"],
                      ascending=[False, True, False],
                      inplace=True, kind="mergesort")

    top = clean.head(int(top_k)).copy()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    top.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if rejected:
        rej_out = os.path.splitext(out_csv)[0] + "_rejected.csv"
        pd.DataFrame(rejected).to_csv(rej_out, index=False, encoding="utf-8-sig")
        print(f"Rejected {len(rejected)} structures -> {rej_out}")

    print(f"Wrote top {len(top)} cleaned & rescored -> {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with a 'smiles' column (or use --col)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rf-model", required=True)
    ap.add_argument("--gcn-dir", required=True)
    ap.add_argument("--col", default="smiles")
    ap.add_argument("--top-k", type=int, default=50)
    a = ap.parse_args()
    main(a.input, a.output, a.rf_model, a.gcn_dir, smiles_col=a.col, top_k=a.top_k)

