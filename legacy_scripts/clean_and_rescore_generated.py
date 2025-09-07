import os, sys; sys.path.append(os.path.dirname(__file__))
import os, pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import joblib

# --- utils ---
def largest_fragment(smiles):
    m = Chem.MolFromSmiles(smiles)
    if not m: return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    if not frags: return None
    biggest = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(biggest, isomericSmiles=True)

def neutralize(mol):
    try:
        uncharger = rdMolStandardize.Uncharger()  # removes charges while keeping valence
        return uncharger.uncharge(mol)
    except Exception:
        return mol

ALLOWED = set(["H","C","N","O","S","F","Cl","Br","I"])  # tighten as you like

def allowed_elements(mol):
    return all(atom.GetSymbol() in ALLOWED for atom in mol.GetAtoms())

def props(m):
    return dict(
        mw = Descriptors.MolWt(m),
        tpsa = rdMolDescriptors.CalcTPSA(m),
        clogP = Crippen.MolLogP(m),
        hbd = rdMolDescriptors.CalcNumHBD(m),
        hba = rdMolDescriptors.CalcNumHBA(m),
        rotb = rdMolDescriptors.CalcNumRotatableBonds(m)
    )

# RF featurization
from rdkit.Chem import AllChem, DataStructs
def ecfp4_array(m, nbits=2048):
    bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

# load models
ROOT = r"C:\Users\Finley\OneDrive\Masters code"
RF_PATH = r"C:\Users\Finley\OneDrive\Masters code\artifacts\metrics_rf\rf_ecfp2048.joblib"
GCN_DIR = r"C:\Users\Finley\OneDrive\Masters code\artifacts\metrics_gcn"
rf = joblib.load(RF_PATH)

# GCN scorer from your adapter
from gcn_molscore_sf import GCNDeepChemScorer
gcn = GCNDeepChemScorer(prefix="GCN", gcn_dir=GCN_DIR)

# input
IN = r"C:\Users\Finley\OneDrive\Masters code\artifacts\evolve_rf\rf_gcn_triaged_top500.csv"
df = pd.read_csv(IN, encoding="utf-8-sig")

# standardize: largest fragment, neutralize, element filter, dedupe
clean = []
for smi in df["smiles"].astype(str):
    try:
        smi1 = largest_fragment(smi)
        if not smi1: continue
        m = Chem.MolFromSmiles(smi1)
        if not m: continue
        m = neutralize(m)
        if not allowed_elements(m): continue
        smi2 = Chem.MolToSmiles(m, isomericSmiles=True)
        clean.append(smi2)
    except Exception:
        continue

clean = pd.Series(clean, name="smiles").drop_duplicates().reset_index(drop=True)
print(f"[clean] kept {len(clean)} unique molecules")

# rescore
# RF
X = np.stack([ecfp4_array(Chem.MolFromSmiles(s)) for s in clean])
rf_scores = rf.predict_proba(X)[:,1]

# GCN
gcn_out = gcn(clean.tolist())
gcn_map = {d["smiles"]: d["GCN_pred_proba"] for d in gcn_out}
gcn_scores = clean.map(gcn_map).values

out = pd.DataFrame({
    "smiles": clean,
    "RF_pred_proba": rf_scores,
    "GCN_pred_proba": gcn_scores
})
out["combo_mean"] = out[["RF_pred_proba","GCN_pred_proba"]].mean(axis=1)

# recompute simple properties post-clean
prop_rows = []
for s in out["smiles"]:
    m = Chem.MolFromSmiles(s)
    prop_rows.append(props(m))
props_df = pd.DataFrame(prop_rows)
out = pd.concat([out, props_df], axis=1)

# sort and save
out = out.sort_values("combo_mean", ascending=False).reset_index(drop=True)
os.makedirs(os.path.join(ROOT,"artifacts","evolve_rf"), exist_ok=True)
OUT = os.path.join(ROOT, "artifacts", "evolve_rf", "rf_gcn_triaged_top500_CLEAN.csv")
out.to_csv(OUT, index=False)
print("Wrote", OUT, "rows=", len(out))
