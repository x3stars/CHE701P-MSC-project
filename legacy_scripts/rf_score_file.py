import argparse, csv, json, os
import numpy as np
from molscore.scoring_functions.sklearn_model import SKLearnModel

def read_smiles(path):
    # supports .smi (one SMILES per line, optional whitespace) and .csv with column named 'Smiles' or 'smiles'
    ext = os.path.splitext(path)[1].lower()
    if ext == ".smi":
        with open(path, "r", encoding="utf-8-sig") as f:
            return [line.strip().split()[0] for line in f if line.strip()]
    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        col = "Smiles" if "Smiles" in df.columns else ("smiles" if "smiles" in df.columns else None)
        if col is None: raise ValueError("CSV must have a 'Smiles' or 'smiles' column")
        return df[col].astype(str).tolist()
    else:
        raise ValueError("Input must be .smi or .csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to rf_molscore.json")
    ap.add_argument("--input",  required=True, help="SMILES file (.smi or .csv)")
    ap.add_argument("--output", required=True, help="Output CSV")
    args = ap.parse_args()

    # load molscore objective config
    with open(args.config, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    params = cfg["objectives"][0]["parameters"]
    prefix = params.get("prefix", "RF")

    scorer = SKLearnModel(**params)

    smiles = read_smiles(args.input)
    # call the scorer
    for mname in ("__call__", "score"):
        fn = getattr(scorer, mname, None)
        if callable(fn):
            out = fn(smiles)
            break
    else:
        raise RuntimeError("SKLearnModel is not callable")

    # handle list-of-dicts like [{'smiles': s, 'RF_pred_proba': p}, ...]
    if isinstance(out, (list, tuple)) and all(isinstance(d, dict) for d in out):
        key_candidates = [f"{prefix}_pred_proba", "pred_proba", "proba", "score", "values", "pred", "y"]
        key = next((k for k in key_candidates if all(k in d for d in out)), None)
        if key is None:
            raise RuntimeError(f"Cannot find score key in returned dicts; saw keys {sorted(set().union(*[set(d.keys()) for d in out]))}")
        by_smiles = {d.get("smiles", ""): float(d[key]) for d in out}
        scores = [by_smiles[s] if s in by_smiles else np.nan for s in smiles]
        score_key = key
    else:
        # fallback: try to coerce to numeric vector
        arr = np.asarray(out, dtype=float).ravel()
        if arr.size != len(smiles):
            raise RuntimeError(f"Unexpected score length: got {arr.size}, expected {len(smiles)}")
        scores = arr.tolist()
        score_key = "score"

    # write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["smiles", score_key])
        for s, p in zip(smiles, scores):
            w.writerow([s, p])

    print(f"Wrote {args.output}  ({len(smiles)} rows)")

if __name__ == "__main__":
    main()
