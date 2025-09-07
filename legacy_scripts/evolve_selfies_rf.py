import os, sys, json, random, argparse, math
import numpy as np
import pandas as pd

# deps
try:
    import selfies as sf
except ModuleNotFoundError:
    sys.exit("Missing dependency: selfies. Install once with:\n  pip install selfies")

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, FilterCatalog, FilterCatalogParams
import joblib

def load_smiles_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def smiles_ok(s):
    m = Chem.MolFromSmiles(s)
    if not m: return False
    mw   = Descriptors.MolWt(m)
    logp = Descriptors.MolLogP(m)
    hbd  = rdMolDescriptors.CalcNumHBD(m)
    hba  = rdMolDescriptors.CalcNumHBA(m)
    tpsa = rdMolDescriptors.CalcTPSA(m)
    # soft drug-likeness window
    return (120 <= mw <= 600) and (logp <= 5.5) and (hbd <= 6) and (hba <= 12) and (tpsa <= 160)

def pains_filter():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)

PAINS = pains_filter()

def pass_filters(smi):
    m = Chem.MolFromSmiles(smi)
    if not m: return False
    if not smiles_ok(smi): return False
    if PAINS.HasMatch(m): return False
    return True

def ecfp_bits(smi, nBits=2048, radius=2):
    m = Chem.MolFromSmiles(smi)
    if not m: return None
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def score_rf(model, smiles):
    fps = []
    keep = []
    for s in smiles:
        arr = ecfp_bits(s)
        if arr is not None:
            fps.append(arr); keep.append(s)
    if not fps:
        return {}
    X = np.stack(fps).astype(np.uint8)
    proba = model.predict_proba(X)[:,1]
    return {s: float(p) for s,p in zip(keep, proba)}

# --- SELFIES mutate ---
SEM_ALPH = list(sf.get_semantic_robust_alphabet())  # safe default alphabet

def mutate_smiles(smi, n_mut=1):
    try:
        base = sf.encoder(smi)
    except Exception:
        return None
    tokens = list(sf.split_selfies(base))
    if len(tokens)==0: return None
    for _ in range(n_mut):
        op = random.choice(("replace","insert","delete"))
        if op=="replace" and tokens:
            i = random.randrange(len(tokens))
            tokens[i] = random.choice(SEM_ALPH)
        elif op=="insert":
            i = random.randrange(len(tokens)+1)
            tokens.insert(i, random.choice(SEM_ALPH))
        elif op=="delete" and len(tokens)>1:
            i = random.randrange(len(tokens))
            del tokens[i]
    new_s = sf.decoder("".join(tokens))
    # RDKit sanitization
    if new_s and Chem.MolFromSmiles(new_s):
        return Chem.MolToSmiles(Chem.MolFromSmiles(new_s))  # canonicalize
    return None

def murcko(s):
    m = Chem.MolFromSmiles(s)
    if not m: return "NO_SCAFFOLD"
    from rdkit.Chem.Scaffolds import MurckoScaffold
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc else "NO_SCAFFOLD"

def main():
    ap = argparse.ArgumentParser(description="Evolve molecules with SELFIES + RF scorer.")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--train", help="CSV with historical molecules (for novelty).")
    ap.add_argument("--train-col", default="Smiles")
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--children", type=int, default=3, help="children per parent per gen")
    ap.add_argument("--keep", type=int, default=200, help="survivors per gen")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.makedirs(args.outdir, exist_ok=True)

    model = joblib.load(args.model)
    seeds = list({s for s in load_smiles_file(args.seeds) if pass_filters(s)})
    if not seeds:
        sys.exit("No valid seeds after filtering.")

    # optional novelty ref
    train_scaff = set()
    if args.train and os.path.exists(args.train):
        try:
            df = pd.read_csv(args.train, encoding="utf-8-sig")
            col = args.train_col if args.train_col in df.columns else ("smiles" if "smiles" in df.columns else None)
            if col:
                train_scaff = {murcko(s) for s in df[col].dropna().astype(str).tolist()}
        except Exception:
            pass

    population = seeds[:args.pop]
    seen = set(population)

    for g in range(1, args.gens+1):
        children = []
        for p in population:
            for _ in range(args.children):
                c = mutate_smiles(p, n_mut=random.choice((1,1,2)))
                if c and c not in seen and pass_filters(c):
                    children.append(c); seen.add(c)
        # score and select
        cand = list({*population, *children})
        scores = score_rf(model, cand)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        population = [s for s,_ in ranked[:args.keep]]

        # write snapshot
        snap = pd.DataFrame(ranked[:args.topk], columns=["smiles","RF_pred_proba"])
        snap["generation"] = g
        snap["scaffold"] = [murcko(s) for s in snap["smiles"]]
        snap["scaffold_novel"] = [sc not in train_scaff for sc in snap["scaffold"]]
        out_csv = os.path.join(args.outdir, f"gen{g:02d}_top{args.topk}.csv")
        snap.to_csv(out_csv, index=False)
        print(f"[gen {g}/{args.gens}] children={len(children)} | unique={len(cand)} | wrote {out_csv}")

    # final top-500
    final = pd.DataFrame(sorted(score_rf(model, list(seen)).items(),
                                key=lambda kv: kv[1], reverse=True)[:500],
                         columns=["smiles","RF_pred_proba"])
    final["scaffold"] = [murcko(s) for s in final["smiles"]]
    final["scaffold_novel"] = [sc not in train_scaff for sc in final["scaffold"]]
    final_path = os.path.join(args.outdir, "rf_evolved_top500.csv")
    final.to_csv(final_path, index=False)
    print(f"Wrote {final_path} (n={len(final)})")

if __name__ == "__main__":
    main()
