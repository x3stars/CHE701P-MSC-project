# rf_gcn_score_file.py — unified config (simple or MolScore-style)
import argparse, json, os, sys, math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib

# our GCN adapter you already created
from gcn_molscore_sf import GCNDeepChemScorer


def ecfp4_bits(smi: str, nbits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    return np.array(list(bv), dtype=np.uint8)


def read_cfg(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    out = {}
    # MolScore-style
    if isinstance(cfg, dict) and "objectives" in cfg:
        # RF (SKLearnModel)
        rf_obj = next((o for o in cfg["objectives"]
                       if "sklearn" in o.get("class", "").lower()
                       or "sklearn" in o.get("name", "").lower()), None)
        if rf_obj:
            p = rf_obj.get("parameters", {})
            out["rf_model"]  = p.get("model_path")
            out["rf_nBits"]  = int(p.get("nBits", 2048))
            out["rf_prefix"] = p.get("prefix", "RF")

        # GCN (your custom scorer)
        gcn_obj = next((o for o in cfg["objectives"]
                        if "gcn" in o.get("class", "").lower()
                        or "gcn" in o.get("name", "").lower()), None)
        if gcn_obj:
            p = gcn_obj.get("parameters", {})
            out["gcn_dir"]   = p.get("gcn_dir")
            out["gcn_prefix"]= p.get("prefix", "GCN")

        # optional weights/aggregate
        out["weights"]   = cfg.get("weights", {"RF_pred_proba": 0.5, "GCN_pred_proba": 0.5})
        out["aggregate"] = cfg.get("aggregate", "mean")
        return out

    # Simple style
    out["rf_model"]   = cfg.get("rf_model")
    out["rf_nBits"]   = int(cfg.get("rf_nBits", cfg.get("nBits", 2048)))
    out["rf_prefix"]  = cfg.get("prefixes", {}).get("rf", "RF")
    out["gcn_dir"]    = cfg.get("gcn_dir")
    out["gcn_prefix"] = cfg.get("prefixes", {}).get("gcn", "GCN")
    out["weights"]    = cfg.get("weights", {"RF_pred_proba": 0.5, "GCN_pred_proba": 0.5})
    out["aggregate"]  = cfg.get("aggregate", "mean")
    return out


def score_rf(model, smiles, nbits):
    X = []
    idx = []
    for i, s in enumerate(smiles):
        x = ecfp4_bits(s, nbits)
        if x is not None:
            X.append(x)
            idx.append(i)
    out = [math.nan] * len(smiles)
    if not X:
        return out
    X = np.asarray(X, dtype=np.uint8)
    proba = model.predict_proba(X)[:, 1]
    for j, i in enumerate(idx):
        out[i] = float(proba[j])
    return out


def score_gcn(gcn, smiles, prefix):
    raw = gcn(smiles)
    key = f"{prefix}_pred_proba"
    # adapter returns list[dict] with that key
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict) and key in raw[0]:
            return [float(d.get(key, math.nan)) for d in raw]
        # or plain list of floats
        try:
            return [float(x) for x in raw]
        except Exception:
            pass
    return [math.nan] * len(smiles)


def normalize_columns(df):
    clean = []
    for c in df.columns:
        c2 = c.replace("\ufeff", "").strip()
        clean.append(c2)
    df.columns = clean
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--col", default="Smiles")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = read_cfg(args.config)
    if not cfg.get("rf_model") or not cfg.get("gcn_dir"):
        print("Config is missing 'rf_model' or 'gcn_dir'.", file=sys.stderr)
        sys.exit(2)

    # load models
    rf = joblib.load(cfg["rf_model"])
    gcn = GCNDeepChemScorer(prefix=cfg.get("gcn_prefix", "GCN"),
                            gcn_dir=cfg["gcn_dir"])

    # data
    df = pd.read_csv(args.input)
    df = normalize_columns(df)
    if args.col not in df.columns:
        print(f"Column '{args.col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)
    smiles = df[args.col].astype(str).tolist()

    # score
    rf_scores  = score_rf(rf, smiles, cfg.get("rf_nBits", 2048))
    gcn_scores = score_gcn(gcn, smiles, cfg.get("gcn_prefix","GCN"))

    # combine (weighted mean)
    w_rf  = float(cfg.get("weights", {}).get("RF_pred_proba", 0.5))
    w_gcn = float(cfg.get("weights", {}).get("GCN_pred_proba", 0.5))
    combo = []
    for r, g in zip(rf_scores, gcn_scores):
        num = (0 if math.isnan(r) else w_rf*r) + (0 if math.isnan(g) else w_gcn*g)
        den = (0 if math.isnan(r) else w_rf)   + (0 if math.isnan(g) else w_gcn)
        combo.append(num/den if den > 0 else math.nan)

    out = pd.DataFrame({
        "smiles": smiles,
        "RF_pred_proba": rf_scores,
        "GCN_pred_proba": gcn_scores,
        "combo_mean": combo
    })
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {args.output}  ({len(out)} rows)")
    # show head
    try:
        print(out.head(5).to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
