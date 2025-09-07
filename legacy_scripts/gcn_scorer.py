# gcn_scorer.py
import os, sys, json, argparse, tempfile, shutil, gc, warnings
import numpy as np

# Silence TF/DeepChem chatter + force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")

from rdkit import Chem
import deepchem as dc
from deepchem.models import GraphConvModel
import tensorflow as tf

def dc_proba(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, 0, 1]
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return arr[:, 1]
    return arr.reshape(-1)

def load_summary(gcn_dir):
    path = os.path.join(gcn_dir, "summary_metrics.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"summary_metrics.json not found in {gcn_dir}")
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    return js["best_params"]

def build_model(params, model_dir):
    return GraphConvModel(
        n_tasks=1, mode="classification",
        graph_conv_layers=params["graph_conv_layers"],
        dense_layer_size=params["dense_layer_size"],
        dropout=params["dropout"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        weight_decay_penalty=params["weight_decay"],
        use_queue=False, model_dir=model_dir
    )

def load_seed_dirs(gcn_dir):
    # auto-detect final seed subdirs
    seeds = []
    for name in sorted(os.listdir(gcn_dir)):
        if name.startswith("gcn_final_seed"):
            full = os.path.join(gcn_dir, name)
            if os.path.isdir(full):
                seeds.append(full)
    if not seeds:
        raise RuntimeError(f"No seed directories like 'gcn_final_seed*' found under {gcn_dir}")
    return seeds

def smiles_to_dataset(smiles, featurizer):
    # DeepChem CSVLoader wants a tasks column; create a temp CSV with dummy labels
    tmpd = tempfile.mkdtemp(prefix="_gcnscore_")
    csvp = os.path.join(tmpd, "to_score.csv")
    with open(csvp, "w", encoding="utf-8") as f:
        f.write("Smiles,active\n")
        for s in smiles:
            f.write(f"{s},0\n")
    loader = dc.data.CSVLoader(tasks=["active"], smiles_field="Smiles", featurizer=featurizer)
    ds = loader.create_dataset(csvp)
    return ds, tmpd

def validate_smiles(smiles):
    good = []
    bad = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        (good if m is not None else bad).append(s)
    return good, bad

def main():
    ap = argparse.ArgumentParser(description="Score SMILES with DeepChem GCN ensemble.")
    ap.add_argument("--gcn-dir", required=True, help="Path to artifacts/metrics_gcn")
    ap.add_argument("--input", help="SMILES file (one per line). If omitted, read stdin.")
    ap.add_argument("--smiles", nargs="*", help="SMILES on the CLI")
    ap.add_argument("--output", help="Write JSON scores here (else print)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # collect smiles
    if args.smiles:
        smiles = [s.strip() for s in args.smiles if s.strip()]
    elif args.input:
        with open(args.input, "r", encoding="utf-8-sig") as f:
            smiles = [ln.strip() for ln in f if ln.strip()]
    else:
        if sys.stdin.isatty():
            sys.exit("No SMILES. Use --smiles, --input, or pipe them in.")
        smiles = [ln.strip() for ln in sys.stdin if ln.strip()]

    if args.verbose:
        print(f"[gcn_scorer] {len(smiles)} SMILES read")

    valid, bad = validate_smiles(smiles)
    if not valid:
        sys.exit("All SMILES failed RDKit parsing.")
    if bad and args.verbose:
        print(f"[gcn_scorer] Skipping {len(bad)} invalid SMILES")

    params = load_summary(args.gcn_dir)
    seed_dirs = load_seed_dirs(args.gcn_dir)
    if args.verbose:
        print(f"[gcn_scorer] best params: {params}")
        print(f"[gcn_scorer] seeds: {seed_dirs}")

    # featurizer
    try:
        featurizer = dc.feat.ConvMolFeaturizer(chirality=True)
    except TypeError:
        try:
            featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
        except TypeError:
            featurizer = dc.feat.ConvMolFeaturizer()

    ds, tmpd = smiles_to_dataset(valid, featurizer)

    # ensemble predictions
    preds = []
    try:
        for sd in seed_dirs:
            try:
                model = build_model(params, sd)
                model.restore()  # load weights from model_dir
                p = model.predict(ds)
                preds.append(dc_proba(p))
                if args.verbose:
                    print(f"[gcn_scorer] restored {sd} → ok")
            except Exception as e:
                if args.verbose:
                    print(f"[gcn_scorer] WARNING: seed {sd} failed: {e}")
            finally:
                tf.keras.backend.clear_session()
                gc.collect()
        if not preds:
            raise RuntimeError("No seed restored successfully.")
        mean_scores = np.mean(np.vstack(preds), axis=0)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    # map back to original list (invalid SMILES omitted)
    out = {s: float(sc) for s, sc in zip(valid, mean_scores)}
    if args.output:                                                                                                                                                                                                                        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        if args.verbose:
            print(f"[gcn_scorer] wrote {len(out)} scores -> {args.output}")
    else:
        print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
