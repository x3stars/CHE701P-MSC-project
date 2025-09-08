# make_figure_pfDHFR_from_csv.py
import os, textwrap
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image, ImageDraw, ImageFont

CSV  = r"C:\Users\Finley\OneDrive\Masters code\artifacts\evolve_rf\rf_gcn_triaged_top500.csv"
OUT  = r"C:\Users\Finley\OneDrive\Masters code\figures\Figure_X_pfDHFR_rank1.png"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# --- load ---
df = pd.read_csv(CSV, encoding="utf-8-sig")

# --- find SMILES column ---
smiles_candidates = ["smiles","SMILES","canonical_smiles","Smiles","mol_smiles"]
sm_col = next((c for c in smiles_candidates if c in df.columns), None)
if sm_col is None:
    # last resort: any column that looks like smiles content
    for c in df.columns:
        if df[c].astype(str).str.contains(r"[=#()\[\]NOPSFrclI]", regex=True, na=False).mean() > 0.5:
            sm_col = c
            break
if sm_col is None:
    raise ValueError(f"Could not find a SMILES column. Available columns: {list(df.columns)}")

# --- find score column ---
lower_map = {c.lower(): c for c in df.columns}
preferred = [
    "ensemble_pred_proba","ensemble_proba","ensemble_score","ensemble",
    "pred_proba","probability","proba","score","model_score",
    "rf_gcn_ensemble","rf_gcn_score"
]
score_col = None
for key in preferred:
    if key in lower_map:
        score_col = lower_map[key]
        break

# if no explicit ensemble score, try to build it
if score_col is None:
    rf = None; gcn = None
    for k in lower_map:
        if "rf" in k and ("proba" in k or "score" in k): rf = lower_map[k]
        if "gcn" in k and ("proba" in k or "score" in k): gcn = lower_map[k]
    if rf and gcn:
        score_col = "Ensemble_pred_proba_auto"
        df[score_col] = 0.5*pd.to_numeric(df[rf], errors="coerce") + 0.5*pd.to_numeric(df[gcn], errors="coerce")

# if still nothing, try rank==1
use_rank = False
if score_col is None and "rank" in lower_map:
    use_rank = True
    score_col = lower_map["rank"]

# sanity print
print("Detected SMILES column:", sm_col)
print("Detected score column:", score_col, "(rank fallback)" if use_rank else "")
print("Available columns:", list(df.columns))

# --- choose the top row ---
if use_rank:
    row = df.sort_values(score_col, ascending=True).iloc[0]  # rank 1 is best
else:
    # make sure numeric
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    row = df.sort_values(score_col, ascending=False).iloc[0]

smiles = str(row[sm_col])
score_txt = f"{float(row[score_col]):.3f}" if not use_rank else f"rank={int(row[score_col])}"

# --- draw molecule ---
m = Chem.MolFromSmiles(smiles)
if m is None:
    raise ValueError("RDKit could not parse the top SMILES. Value was:\n" + smiles)
AllChem.Compute2DCoords(m)

legend = f"Rank 1 | {'Ensemble score' if not use_rank else 'Top candidate'} = {score_txt}"
mol_img = Draw.MolToImage(m, size=(900, 650), legend=legend)

cap = (f"Figure X. Representative example of a top-5 de novo generated PfDHFR candidate "
       f"(Rank 1, {'ensemble score' if not use_rank else 'selection'} = {score_txt}). "
       f"Structure rendered from SMILES using RDKit.")
wrap = textwrap.fill(cap, width=100)

W, H = mol_img.size
caption_h = 110
canvas = Image.new("RGB", (W, H + caption_h), "white")
canvas.paste(mol_img, (0, 0))
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()
d.text((10, H + 12), wrap, fill="black", font=font)

canvas.save(OUT, dpi=(300, 300))
print("Saved:", OUT)
