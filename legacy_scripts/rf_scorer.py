# rf_scorer.py  — ECFP4 (2048) + optional 11 RDKit descriptors, auto-detect model input size
import argparse, json, os, sys, traceback
from typing import List, Dict
import numpy as np

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen

# joblib for loading sklearn models/pipelines
try:
    import joblib
except Exception:
    from sklearn.externals import joblib  # very old sklearn fallback

DESC_ORDER = [
    "mol_weight", "logP", "TPSA", "HBD", "HBA", "RotatableBonds",
    "Fsp3", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "HeavyAtomCount"
]
N_BITS = 2048

def read_smiles(args):
    def clean(s: str) -> str:
        # strip whitespace + any UTF-8 BOM that may appear on the first line
        return s.strip().lstrip("\ufeff")

    smi = []
    if args.input:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"SMILES file not found: {args.input}")
        # utf-8-sig auto-removes BOM if present
        with open(args.input, "r", encoding="utf-8-sig") as f:
            smi = [clean(ln) for ln in f if clean(ln) and not clean(ln).startswith("#")]
    elif not sys.stdin.isatty():
        smi = [clean(ln) for ln in sys.stdin if clean(ln) and not clean(ln).startswith("#")]
    elif args.smiles:
        smi = [clean(s) for s in args.smiles if clean(s)]
    else:
        raise SystemExit("No SMILES provided. Use --input FILE or pipe, or --smiles.")
    return smi


def smiles_to_mol(s: str):
    m = Chem.MolFromSmiles(s)
    if m is None:
        raise ValueError(f"Bad SMILES: {s}")
    return m

def ecfp4_array(mol, n_bits=N_BITS) -> np.ndarray:
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def descriptor_vector(mol) -> np.ndarray:
    return np.array([
        Descriptors.MolWt(mol),                                  # mol_weight
        Crippen.MolLogP(mol),                                     # logP
        rdMolDescriptors.CalcTPSA(mol),                           # TPSA
        rdMolDescriptors.CalcNumHBD(mol),                         # HBD
        rdMolDescriptors.CalcNumHBA(mol),                         # HBA
        rdMolDescriptors.CalcNumRotatableBonds(mol),              # RotatableBonds
        rdMolDescriptors.CalcFractionCSP3(mol),                   # Fsp3
        rdMolDescriptors.CalcNumAromaticRings(mol),               # NumAromaticRings
        rdMolDescriptors.CalcNumAliphaticRings(mol),              # NumAliphaticRings
        rdMolDescriptors.CalcNumSaturatedRings(mol),              # NumSaturatedRings
        mol.GetNumHeavyAtoms(),                                    # HeavyAtomCount
    ], dtype=np.float32)

def featurize(smiles: List[str]):
    mols = [smiles_to_mol(s) for s in smiles]
    X_fp = np.vstack([ecfp4_array(m) for m in mols])                 # (N,2048) uint8
    X_desc = np.vstack([descriptor_vector(m) for m in mols])         # (N,11)  float32
    X_combo = np.hstack([X_fp.astype(np.float32), X_desc])           # (N,2059) float32
    return X_fp, X_desc, X_combo

def predict_with_model(model, smiles: List[str], class_index: int) -> Dict[str, float]:
    """
    Strategy:
      A) Try direct predict_proba(smiles) (in case you saved a Pipeline)
      B) Try 2059-feature (ECFP+desc)
      C) Fallback to 2048-feature (ECFP only)
    """
    # A) direct
    try:
        proba = model.predict_proba(smiles)
        return {s: float(proba[i][class_index]) for i, s in enumerate(smiles)}
    except Exception:
        pass

    # prepare features once
    X_fp, X_desc, X_combo = featurize(smiles)

    # If model exposes n_features_in_, try to match
    nfi = getattr(model, "n_features_in_", None)
    if isinstance(nfi, (int, np.integer)):
        if nfi == X_combo.shape[1]:
            proba = model.predict_proba(X_combo)
            return {smiles[i]: float(proba[i][class_index]) for i in range(len(smiles))}
        if nfi == X_fp.shape[1]:
            proba = model.predict_proba(X_fp)
            return {smiles[i]: float(proba[i][class_index]) for i in range(len(smiles))}
        # otherwise fall through to trial-and-error

    # B) try ECFP+desc (2059)
    try:
        proba = model.predict_proba(X_combo)
        return {smiles[i]: float(proba[i][class_index]) for i in range(len(smiles))}
    except Exception:
        pass

    # C) try ECFP only (2048)
    proba = model.predict_proba(X_fp)
    return {smiles[i]: float(proba[i][class_index]) for i in range(len(smiles))}

def main():
    p = argparse.ArgumentParser(description="Score SMILES with a saved RF/Sklearn model (ECFP4 ± descriptors).")
    p.add_argument("--model", required=True, help="Path to joblib model/pipeline")
    p.add_argument("--input", help="SMILES file (one per line). If omitted, read stdin.")
    p.add_argument("--smiles", nargs="*", help="SMILES given directly on the command line")
    p.add_argument("--output", help="Write JSON to this path (else print to stdout)")
    p.add_argument("--class-index", type=int, default=1, help="Index of positive class in predict_proba (default=1)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    try:
        if args.verbose:
            print(f"[rf_scorer] loading model: {args.model}", file=sys.stderr)
        model = joblib.load(args.model)

        smiles = read_smiles(args)
        if args.verbose:
            print(f"[rf_scorer] read {len(smiles)} SMILES", file=sys.stderr)

        scores = predict_with_model(model, smiles, args.class_index)

        payload = json.dumps(scores, ensure_ascii=False)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(payload)
            if args.verbose:
                print(f"[rf_scorer] wrote {len(scores)} scores -> {args.output}", file=sys.stderr)
        else:
            print(payload)
    except Exception as e:
        print("[rf_scorer] ERROR:", str(e), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
