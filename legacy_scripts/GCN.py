# ---- quiet mode (keep GPU off + silence spam) ----
import os, warnings, gc, shutil, traceback, sys, json, hashlib, time, subprocess, tempfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF INFO/WARN logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

warnings.filterwarnings("ignore", message=r".*Converting sparse IndexedSlices.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

N_ENSEMBLE = 2          
param_list[0]["bs"] = 24
param_list[1]["bs"] = 24
param_list[2]["bs"] = 24
param_list[3]["bs"] = 24

# ---- end quiet mode ----

import random
import numpy as np
import pandas as pd

import deepchem as dc
from deepchem.models import GraphConvModel

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Small helpers used by both orchestrator and worker
# -----------------------------
def make_loader():
    try:
        featurizer = dc.feat.ConvMolFeaturizer(chirality=True)
    except TypeError:
        try:
            featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
        except TypeError:
            featurizer = dc.feat.ConvMolFeaturizer()
    loader = dc.data.CSVLoader(tasks=["active"], smiles_field="Smiles", featurizer=featurizer)
    return loader

def add_class_weights(dataset):
    y = np.asarray(dataset.y).reshape(-1).astype(int)
    n = y.size; pos = max(1, int(y.sum())); neg = max(1, n - pos)
    w_pos = n / (2.0 * pos); w_neg = n / (2.0 * neg)
    w = np.where(y == 1, w_pos, w_neg).astype(np.float32).reshape(-1, 1)
    set_w = getattr(dataset, "set_w", None)
    if callable(set_w):
        dataset.set_w(w); return dataset
    return dc.data.NumpyDataset(X=dataset.X, y=dataset.y, w=w, ids=dataset.ids)

def build_gcn(cfg, model_dir, seed):
    return GraphConvModel(
        n_tasks=1, mode="classification",
        graph_conv_layers=cfg["graph_conv_layers"],
        dense_layer_size=cfg["dense"],
        dropout=cfg["dropout"],
        batch_size=cfg["bs"],
        learning_rate=cfg["lr"],
        weight_decay_penalty=cfg["wd"],
        use_queue=False, model_dir=model_dir, random_seed=seed,
    )

def dc_proba(model, ds):
    out = np.asarray(model.predict(ds))
    if out.ndim == 3 and out.shape[-1] == 2: return out[:, 0, 1]
    if out.ndim == 2 and out.shape[1] == 2: return out[:, 1]
    return out.reshape(-1)

def clear_mem(*objs, paths=None):
    for o in objs:
        try: del o
        except Exception: pass
    if paths:
        for p in paths:
            try: shutil.rmtree(p, ignore_errors=True)
            except Exception: pass
    tf.keras.backend.clear_session()
    gc.collect()

# ===========================================================
#                    WORKER MODE
# Trains once (possibly retrying with half batch) in a fresh
# process, predicts on provided CSV, writes metrics & probs.
# ===========================================================
def worker_main(payload_path: str):
    with open(payload_path, "r") as f:
        P = json.load(f)

    SEED = int(P["seed"])
    random.seed(SEED); np.random.seed(SEED)
    try: tf.random.set_seed(SEED)
    except Exception: pass

    loader = make_loader()

    tr_ds = loader.create_dataset(P["train_csv"])
    tr_ds = add_class_weights(tr_ds)
    pred_ds = loader.create_dataset(P["predict_csv"])

    cfg = P["cfg"]
    nb_epoch = int(P["nb_epoch"])
    model_dir = P["model_dir"]

    # Try at cfg["bs"]; on OOM, retry once at half (min 8).
    used_bs = int(cfg["bs"])
    tried_half = False
    while True:
        try:
            cfg_try = dict(cfg); cfg_try["bs"] = used_bs
            model = build_gcn(cfg_try, model_dir, SEED)
            model.fit(tr_ds, nb_epoch=nb_epoch, all_losses=None)
            y_proba = dc_proba(model, pred_ds)
            np.save(P["out_proba_npy"], y_proba)
            # Compute AUCPR if labels exist (tuning/val); else set None.
            aupr = None
            if P.get("has_labels", True):
                y_true = pred_ds.y.reshape(-1).astype(int)
                if y_true.sum() > 0 and y_true.sum() < len(y_true):
                    aupr = float(average_precision_score(y_true, y_proba))
            with open(P["out_metrics_json"], "w") as g:
                json.dump({"status": "ok", "used_bs": used_bs, "aupr": aupr}, g)
            return 0
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError, MemoryError):
            if tried_half or used_bs <= 8:
                with open(P["out_metrics_json"], "w") as g:
                    json.dump({"status": "oom"}, g)
                return 0  # signal handled OOM
            used_bs = max(8, used_bs // 2)
            tried_half = True
            clear_mem(locals().get("model"))
        except Exception as e:
            with open(P["out_metrics_json"], "w") as g:
                json.dump({"status": "error", "msg": str(e)}, g)
            return 1

# ===========================================================
#                   ORCHESTRATOR MODE
# Your original pipeline, now calling the worker per fold.
# ===========================================================
def main():
    # -----------------------------
    # Config
    # -----------------------------
    SEED = 42
    FAST_DEBUG = False
    random.seed(SEED); np.random.seed(SEED)
    try: tf.random.set_seed(SEED)
    except Exception: pass

    DATA_CSV = r"C:\Users\Finley\OneDrive\Masters code\Data\pfDHFR_cleaned_with_potency_descriptors.csv"
    OUT_DIR = "artifacts/metrics_gcn"
    TMP_DIR = os.path.join(OUT_DIR, "_tmp")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    if not os.path.isfile(DATA_CSV):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_CSV}")

    NB_EPOCH_TUNE  = 8  if FAST_DEBUG else 16
    NB_EPOCH_CV    = 10 if FAST_DEBUG else 32
    NB_EPOCH_FINAL = 12 if FAST_DEBUG else 48
    N_ENSEMBLE     = 1  if FAST_DEBUG else 3  # trimmed for stability

    # -----------------------------
    # Load and prep data
    # -----------------------------
    print("Loading data… (DeepChem=", getattr(dc, "__version__", "unknown"), ")", flush=True)
    df = pd.read_csv(DATA_CSV, low_memory=False)
    if not {"Smiles", "active"}.issubset(df.columns):
        raise KeyError("Input CSV must contain 'Smiles' and 'active' columns.")

    df = df[["Smiles", "active"]].dropna(subset=["Smiles"]).reset_index(drop=True)

    def smi_to_mol(s):
        try: return Chem.MolFromSmiles(s)
        except Exception: return None

    print("Parsing SMILES with RDKit…", flush=True)
    df["mol"] = df["Smiles"].apply(smi_to_mol)
    df = df[df["mol"].notna()].reset_index(drop=True)

    print("Computing Murcko scaffolds…", flush=True)
    def mol_to_scaffold_smi(mol):
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaf) if scaf is not None else "NO_SCAFFOLD"
        except Exception:
            return "NO_SCAFFOLD"
    df["scaffold_smiles"] = df["mol"].apply(mol_to_scaffold_smi)
    print(f"Valid molecules: {len(df)} | Unique scaffolds: {df['scaffold_smiles'].nunique()}", flush=True)

    print("Creating deterministic external scaffold hold-out…", flush=True)
    def scaffold_hash_bucket(smi: str, buckets: int = 5) -> int:
        h = hashlib.md5(smi.encode("utf-8")).hexdigest()
        return int(h, 16) % buckets

    scaffold_buckets = df["scaffold_smiles"].apply(lambda s: scaffold_hash_bucket(s, 5))
    TEST_BUCKET = 0
    test_mask = scaffold_buckets == TEST_BUCKET
    train_mask = ~test_mask

    train_df = df[train_mask].reset_index(drop=True)
    test_df  = df[test_mask].reset_index(drop=True)
    print(f"Train molecules: {len(train_df)} | Test molecules (external scaffolds): {len(test_df)}", flush=True)

    with open(os.path.join(OUT_DIR, "scaffold_split.json"), "w") as f:
        json.dump({
            "test_scaffolds": sorted(test_df["scaffold_smiles"].unique().tolist()),
            "train_scaffolds": sorted(train_df["scaffold_smiles"].unique().tolist()),
            "hash_bucket_rule": {"buckets": 5, "TEST_BUCKET": 0}
        }, f, indent=2)

    loader = make_loader()

    # -----------------------------
    # Hyperparameter sweep (3-fold GroupKFold) — safer configs
    # -----------------------------
    param_list = [
        {"graph_conv_layers": [128, 128],     "dense": 256, "dropout": 0.25, "lr": 1e-3,  "wd": 1e-5, "bs": 32},
        {"graph_conv_layers": [128, 128, 64], "dense": 256, "dropout": 0.30, "lr": 7e-4, "wd": 1e-5, "bs": 32},
        {"graph_conv_layers": [96,  96],      "dense": 192, "dropout": 0.20, "lr": 1e-3,  "wd": 5e-6, "bs": 32},
        {"graph_conv_layers": [64,  64],      "dense": 128, "dropout": 0.25, "lr": 1e-3,  "wd": 1e-5, "bs": 32},
    ]

    def run_worker(train_csv, predict_csv, cfg, nb_epoch, model_dir, seed, has_labels, tmp_prefix):
        payload = {
            "train_csv": train_csv,
            "predict_csv": predict_csv,
            "cfg": cfg,
            "nb_epoch": nb_epoch,
            "model_dir": model_dir,
            "seed": seed,
            "has_labels": has_labels,
            "out_proba_npy": tmp_prefix + "_proba.npy",
            "out_metrics_json": tmp_prefix + "_metrics.json",
        }
        payload_path = tmp_prefix + "_payload.json"
        with open(payload_path, "w") as f:
            json.dump(payload, f)
        cmd = [sys.executable, sys.argv[0], "--worker", payload_path]
        try:
            ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as e:
            return None, {"status": "error", "msg": str(e)}
        # try to read metrics
        try:
            with open(payload["out_metrics_json"], "r") as g:
                metrics = json.load(g)
        except Exception:
            metrics = {"status": "error", "msg": "worker crashed", "stderr": ret.stderr[-500:]}
        # load proba if exists
        proba = None
        if os.path.isfile(payload["out_proba_npy"]):
            try:
                proba = np.load(payload["out_proba_npy"])
            except Exception:
                proba = None
        # cleanup worker temp files but keep model_dir for final models
        for p in [payload_path, payload["out_metrics_json"], payload["out_proba_npy"]]:
            try: os.remove(p)
            except Exception: pass
        return proba, metrics

    print("Tuning hyperparameters with 3-fold GroupKFold on training scaffolds…", flush=True)
    train_groups = train_df["scaffold_smiles"].values
    train_indices = np.arange(len(train_df))
    inner_gkf = GroupKFold(n_splits=3)

    best_cfg, best_cv_aupr = None, -np.inf

    for i, cfg in enumerate(param_list, 1):
        fold_scores = []
        print(f"Config {i}/{len(param_list)}: {cfg}", flush=True)
        for fold, (tr_idx, va_idx) in enumerate(inner_gkf.split(train_indices, np.zeros(len(train_indices)), groups=train_groups), 1):
            tr_rows = train_df.iloc[tr_idx][["Smiles", "active"]]
            va_rows = train_df.iloc[va_idx][["Smiles", "active"]]

            tr_tmp = os.path.join(TMP_DIR, f"_tr_cfg{i}_fold{fold}.csv")
            va_tmp = os.path.join(TMP_DIR, f"_va_cfg{i}_fold{fold}.csv")
            tr_rows.to_csv(tr_tmp, index=False); va_rows.to_csv(va_tmp, index=False)

            tmp_prefix = os.path.join(TMP_DIR, f"cfg{i}_fold{fold}")
            model_dir  = os.path.join(TMP_DIR, f"gcn_tune_cfg{i}_fold{fold}")
            proba, m = run_worker(tr_tmp, va_tmp, cfg, NB_EPOCH_TUNE, model_dir, SEED+fold+i*10, True, tmp_prefix)
            if m.get("status") == "ok" and m.get("aupr") is not None:
                fold_scores.append(float(m["aupr"]))
                print(f"  Fold {fold}: AUCPR={m['aupr']:.3f} (bs={m.get('used_bs','?')})", flush=True)
            elif m.get("status") == "oom":
                print(f"  Fold {fold}: skipped after OOM.", flush=True)
                fold_scores.append(np.nan)
            else:
                msg = m.get("msg", "unknown error")
                print(f"  Fold {fold}: ERROR -> {msg}", flush=True)
                fold_scores.append(np.nan)

            # clean CSVs and model_dir
            clear_mem(paths=[tr_tmp, va_tmp, model_dir])

        mean_aupr = float(np.nanmean(fold_scores)) if np.any(~np.isnan(fold_scores)) else -np.inf
        print(f"  Mean AUCPR (cfg {i}): {mean_aupr:.3f}", flush=True)
        if mean_aupr > best_cv_aupr:
            best_cv_aupr = mean_aupr; best_cfg = cfg

    print("Best config:", best_cfg, "CV AUCPR=", round(best_cv_aupr, 3), flush=True)

    # -----------------------------
    # 5-fold scaffold CV PR curve (selected config)
    # -----------------------------
    print("5-fold GroupKFold PR curve on training set…", flush=True)
    outer_gkf = GroupKFold(n_splits=5)
    cv_pred = np.full(len(train_df), np.nan, dtype=float)
    cv_true = train_df["active"].values.astype(int)

    for fold, (tr_idx, va_idx) in enumerate(outer_gkf.split(train_indices, np.zeros(len(train_indices)), groups=train_groups), 1):
        tr_rows = train_df.iloc[tr_idx][["Smiles", "active"]]
        va_rows = train_df.iloc[va_idx][["Smiles", "active"]]
        tr_tmp = os.path.join(TMP_DIR, f"_cv_tr_fold{fold}.csv")
        va_tmp = os.path.join(TMP_DIR, f"_cv_va_fold{fold}.csv")
        tr_rows.to_csv(tr_tmp, index=False); va_rows.to_csv(va_tmp, index=False)

        tmp_prefix = os.path.join(TMP_DIR, f"cv_fold{fold}")
        model_dir  = os.path.join(TMP_DIR, f"gcn_cv_fold{fold}")
        proba, m = run_worker(tr_tmp, va_tmp, best_cfg, NB_EPOCH_CV, model_dir, SEED+fold+123, True, tmp_prefix)
        if proba is not None:
            cv_pred[va_idx] = proba
            print(f"  CV fold {fold} done (bs={m.get('used_bs','?')}).", flush=True)
        else:
            print(f"  CV fold {fold}: skipped (worker error/oom).", flush=True)

        clear_mem(paths=[tr_tmp, va_tmp, model_dir])

    mask = ~np.isnan(cv_pred)
    if not np.any(mask):
        raise RuntimeError("All CV predictions are NaN; check earlier logs.")
    cv_pr_auc = float(average_precision_score(cv_true[mask], cv_pred[mask]))
    prec, rec, thr = precision_recall_curve(cv_true[mask], cv_pred[mask])
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_thr = 0.0 if best_idx == 0 else float(thr[best_idx-1])
    np.save(os.path.join(OUT_DIR, "best_threshold.npy"), np.array([best_thr]))
    print(f"CV-derived best threshold ~ {best_thr:.3f}", flush=True)

    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"GCN (PR-AUC={cv_pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (Scaffold-CV)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cv_precision_recall.png"), dpi=300); plt.close()

    # -----------------------------
    # Final model (seed ensemble, each in its own worker) + held-out eval
    # -----------------------------
    print("Training final GCN seed-ensemble on the full training set…", flush=True)
    # Export datasets once
    test_tmp = os.path.join(TMP_DIR, "_final_test.csv")
    test_df[["Smiles", "active"]].to_csv(test_tmp, index=False)
    full_train_tmp = os.path.join(TMP_DIR, "_final_train.csv")
    train_df[["Smiles", "active"]].to_csv(full_train_tmp, index=False)

    test_probas = []
    for s in range(N_ENSEMBLE):
        model_dir = os.path.join(OUT_DIR, f"gcn_final_seed{s}")  # keep final models
        tmp_prefix = os.path.join(TMP_DIR, f"final_seed{s}")
        proba, m = run_worker(full_train_tmp, test_tmp, best_cfg, NB_EPOCH_FINAL, model_dir, SEED+s, True, tmp_prefix)
        if proba is not None:
            test_probas.append(proba)
            print(f"  Ensemble member {s+1}/{N_ENSEMBLE} done (bs={m.get('used_bs','?')}).", flush=True)
        else:
            print(f"  Ensemble member {s+1}: skipped (worker error/oom).", flush=True)

    if len(test_probas) == 0:
        raise RuntimeError("No ensemble members finished; check earlier logs.")
    y_proba_test = np.mean(np.vstack(test_probas), axis=0)

    # Build a DeepChem dataset once to get y_true in same order
    loader_local = make_loader()
    test_ds = loader_local.create_dataset(test_tmp)
    y_true_test = test_ds.y.reshape(-1)

    # -----------------------------
    # Metrics on external test scaffolds
    # -----------------------------
    ho_roc = float(roc_auc_score(y_true_test, y_proba_test))
    ho_pr  = float(average_precision_score(y_true_test, y_proba_test))
    prevalence = float(y_true_test.mean())
    print(f"Hold-out prevalence (AUCPR baseline): {prevalence:.3f}", flush=True)

    y_hat_test = (y_proba_test >= best_thr).astype(int)
    prec_h = float(precision_score(y_true_test, y_hat_test))
    rec_h  = float(recall_score(y_true_test, y_hat_test))
    f1_h   = float(f1_score(y_true_test, y_hat_test))
    tn, fp, fn, tp = confusion_matrix(y_true_test, y_hat_test).ravel()

    def enrichment_factor(y_true, scores, top_frac):
        n = len(y_true); k = max(1, int(np.ceil(top_frac * n)))
        idx = np.argsort(scores)[::-1][:k]
        hits = int(y_true[idx].sum())
        hit_rate_top = hits / k
        hit_rate_all = y_true.sum() / n if y_true.sum() > 0 else 0
        ef = (hit_rate_top / hit_rate_all) if hit_rate_all > 0 else np.nan
        return ef, hits

    ef1, hits1 = enrichment_factor(y_true_test, y_proba_test, 0.01)
    ef5, hits5 = enrichment_factor(y_true_test, y_proba_test, 0.05)
    k100 = min(100, len(y_true_test))
    hits_at_100 = int(y_true_test[np.argsort(y_proba_test)[::-1][:k100]].sum())

    prob_true, prob_pred = calibration_curve(y_true_test, y_proba_test, n_bins=10, strategy="quantile")
    brier = float(brier_score_loss(y_true_test, y_proba_test))

    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker="o", label="GCN (held-out)")
    plt.plot([0,1],[0,1], "--", color="gray", label="Perfect")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed active fraction")
    plt.title("Calibration (Scaffold Hold-out)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "holdout_calibration.png"), dpi=300); plt.close()

    pd.DataFrame({"prob_pred_bin": prob_pred, "prob_true": prob_true}).to_csv(
        os.path.join(OUT_DIR, "holdout_calibration_points.csv"), index=False
    )

    # Per-scaffold AUC
    print("Per-scaffold AUC on external test…", flush=True)
    res_rows = []
    for scaf, g in test_df.assign(y_true=y_true_test, y_proba=y_proba_test).groupby("scaffold_smiles"):
        y_g = g["y_true"].values; p_g = g["y_proba"].values
        auc = np.nan
        if len(np.unique(y_g)) == 2:
            auc = float(roc_auc_score(y_g, p_g))
        res_rows.append({
            "scaffold_smiles": scaf, "n": int(len(g)),
            "n_active": int(y_g.sum()), "active_frac": float(y_g.mean()), "AUC": auc,
        })
    res_df = pd.DataFrame(res_rows)
    res_df.to_csv(os.path.join(OUT_DIR, "holdout_per_scaffold_auc.csv"), index=False)

    valid_aucs = res_df["AUC"].dropna().values
    plt.figure(figsize=(7,5))
    plt.boxplot(valid_aucs, vert=True, labels=["Per-scaffold AUC"])
    plt.ylabel("AUC"); plt.title("Per-scaffold AUC (Scaffold Hold-out)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "holdout_per_scaffold_auc_box.png"), dpi=300); plt.close()

    # Per-compound preds
    pd.DataFrame({
        "Smiles": test_df["Smiles"].values,
        "active": y_true_test.astype(int),
        "proba": y_proba_test,
        "pred": y_hat_test.astype(int)
    }).to_csv(os.path.join(OUT_DIR, "holdout_predictions.csv"), index=False)

    # Summary
    summary = {
        "cv": {"pr_auc": float(cv_pr_auc), "best_threshold_f1": float(best_thr)},
        "holdout": {
            "roc_auc": ho_roc, "pr_auc": ho_pr, "aupr_baseline": prevalence,
            "brier_score": brier, "precision_at_thr": float(prec_h), "recall_at_thr": float(rec_h),
            "f1_at_thr": float(f1_h), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "ef@1percent": float(ef1) if ef1 == ef1 else None, "hits_in_top1percent": int(hits1),
            "ef@5percent": float(ef5) if ef5 == ef5 else None, "hits_in_top5percent": int(hits5),
            "hits@100": int(hits_at_100),
            "per_scaffold_auc_count": int(len(valid_aucs)),
            "per_scaffold_auc_median": float(np.nanmedian(valid_aucs)) if len(valid_aucs) > 0 else None,
            "per_scaffold_auc_iqr": (
                float(np.nanpercentile(valid_aucs, 75) - np.nanpercentile(valid_aucs, 25)) if len(valid_aucs) > 0 else None
            ),
        },
        "best_params": {
            "graph_conv_layers": best_cfg["graph_conv_layers"],
            "dense_layer_size": best_cfg["dense"],
            "dropout": best_cfg["dropout"],
            "learning_rate": best_cfg["lr"],
            "weight_decay": best_cfg["wd"],
            "batch_size": best_cfg["bs"],
            "epochs_cv": NB_EPOCH_CV, "epochs_final": NB_EPOCH_FINAL, "ensemble_members": N_ENSEMBLE,
        },
    }
    with open(os.path.join(OUT_DIR, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved GCN metrics to:", OUT_DIR, flush=True)
    print(json.dumps(summary, indent=2))

    # cleanup tmp files
    try: shutil.rmtree(TMP_DIR, ignore_errors=True)
    except Exception: pass


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(worker_main(sys.argv[2]))
    else:
        main()
