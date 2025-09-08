import os
import warnings
warnings.filterwarnings("ignore")

# ---- plotting in headless mode & make output dir early ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ART_DIR = "artifacts/metrics_rf"
os.makedirs(ART_DIR, exist_ok=True)

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from scipy.stats import randint
import joblib


USE_DESCRIPTORS = False  #keep False for MolScore (ECFP-only, 2048 bits)

#Load and prepare data
print("Loading data...")
CSV = r"C:\Users\Finley\OneDrive\Masters code\Data\pfDHFR_cleaned_with_potency_descriptors.csv"
df = pd.read_csv(CSV)
df["mol"] = df["Smiles"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notna()].reset_index(drop=True)
print(f"Valid RDKit molecules: {len(df)}")

#ECFP fingerprints 
def mol_to_fp_array(mol, radius=2, nbits=2048):
    try:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr
    except Exception:
        return None

fp_arrays = df["mol"].apply(mol_to_fp_array)
valid_mask = fp_arrays.notna()
df = df[valid_mask].reset_index(drop=True)
fp_arrays = fp_arrays[valid_mask].reset_index(drop=True)

if len(fp_arrays) == 0:
    raise ValueError("All fingerprint generations failed.")

X_fp = np.stack(fp_arrays.values).astype(np.uint8)
print(f"Fingerprint matrix: {X_fp.shape}")

#Murcko scaffolds (groups) 
def mol_to_scaffold_smiles(mol):
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaff) if scaff is not None else None
    except Exception:
        return None

df["scaffold_smiles"] = df["mol"].apply(mol_to_scaffold_smiles).fillna("NO_SCAFFOLD")
groups = df["scaffold_smiles"].values

#Optional descriptors (not used when USE_DESCRIPTORS=False)
descriptor_cols = [
    "mol_weight","logP","TPSA","HBD","HBA","RotatableBonds",
    "Fsp3","NumAromaticRings","NumAliphaticRings",
    "NumSaturatedRings","HeavyAtomCount"
]
if USE_DESCRIPTORS:
    missing = [c for c in descriptor_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing descriptor columns: {missing}")
    df[descriptor_cols] = df[descriptor_cols].fillna(0)
    X_desc = df[descriptor_cols].to_numpy(dtype=np.float32)
    X = np.hstack([X_fp, X_desc])
    feature_names = [f"FP_{i}" for i in range(X_fp.shape[1])] + descriptor_cols
else:
    X = X_fp
    feature_names = [f"FP_{i}" for i in range(X_fp.shape[1])]

y = df["active"].astype(int).values
print(f"Feature matrix used for training: {X.shape} (USE_DESCRIPTORS={USE_DESCRIPTORS})")

#GroupKFold CV 
unique_scaffolds = pd.unique(df["scaffold_smiles"])
print(f"Unique scaffolds: {len(unique_scaffolds)}")
group_cv = GroupKFold(n_splits=5)
cv_splits = list(group_cv.split(X, y, groups=groups))

#RandomizedSearchCV 
param_dist = {
    "n_estimators": randint(200, 1000),
    "max_depth": [None] + list(range(6, 31, 6)),
    "min_samples_split": randint(2, 30),
    "min_samples_leaf": randint(1, 20),
    "max_features": ["sqrt", "log2", None]
}
print("Running RandomizedSearchCV (scaffold-aware CV)...")
rf_base = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=200,
    scoring="roc_auc",
    n_jobs=-1,
    cv=cv_splits,
    verbose=2,
    random_state=42
)
search.fit(X, y)
print("Randomized search complete.")
print("\nBest Parameters (RandomizedSearch):")
print(search.best_params_)
print("\nBest Cross-Validated ROC AUC (scaffold CV): {:.3f}".format(search.best_score_))

#GridSearchCV refinement 
best_params = search.best_params_
param_grid = {
    "n_estimators": [
        max(100, best_params["n_estimators"] - 100),
        best_params["n_estimators"],
        best_params["n_estimators"] + 100
    ],
    "max_depth": (
        [best_params["max_depth"]] if best_params["max_depth"] is None else
        [max(3, best_params["max_depth"] - 3), best_params["max_depth"], best_params["max_depth"] + 3]
    ),
    "min_samples_split": [
        max(2, best_params["min_samples_split"] - 2),
        best_params["min_samples_split"],
        best_params["min_samples_split"] + 2
    ],
    "min_samples_leaf": [
        max(1, best_params["min_samples_leaf"] - 1),
        best_params["min_samples_leaf"],
        best_params["min_samples_leaf"] + 1
    ],
    "max_features": [best_params["max_features"]]
}
print("Running GridSearchCV (scaffold-aware CV)...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv_splits,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X, y)
print("Grid search complete.")
print("\nBest Parameters (GridSearch):")
print(grid_search.best_params_)
print("\nBest ROC AUC (GridSearch, scaffold CV): {:.3f}".format(grid_search.best_score_))

#Scaffold-held-out evaluation 
print("\nEvaluating on scaffold-held-out split...")
rng = np.random.RandomState(42)
scaffs = df["scaffold_smiles"].unique()
rng.shuffle(scaffs)
n_test = max(1, int(0.2 * len(scaffs)))
test_scaffs = set(scaffs[:n_test])
test_mask = df["scaffold_smiles"].isin(test_scaffs)
train_mask = ~test_mask

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

final_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42,
                                  class_weight="balanced", n_jobs=-1)
final_rf.fit(X_train, y_train)
y_proba = final_rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"Scaffold-Held-Out ROC AUC: {auc:.3f}")

#Per-scaffold analysis 
df_test = df[test_mask].copy()
df_test["y_true"] = y_test
df_test["y_proba"] = y_proba

def per_group_auc(g):
    if g["y_true"].nunique() < 2:
        return np.nan
    return roc_auc_score(g["y_true"], g["y_proba"])

scaffold_perf = (df_test
    .groupby("scaffold_smiles")
    .apply(lambda g: pd.Series({
        "n": len(g),
        "n_active": int(g["y_true"].sum()),
        "active_frac": float(g["y_true"].mean()),
        "AUC": per_group_auc(g)
    }))
    .sort_values("AUC", ascending=False))

print("\nTop 10 scaffold performance (by per-scaffold AUC):\n")
print(scaffold_perf.head(10))

#Feature importances (save fig) 
importances = final_rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_k = min(20, len(importances))

plt.figure(figsize=(10, 6))
plt.bar(range(top_k), importances[indices[:top_k]], align="center")
plt.xticks(range(top_k), [feature_names[i] for i in indices[:top_k]], rotation=90)
plt.ylabel("Importance")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "rf_feature_importances.png"), dpi=300)
plt.close()

# If we didn't use descriptors, skip contribution summary
if USE_DESCRIPTORS:
    desc_importance = sum(importances[-len(descriptor_cols):])
    fp_importance = sum(importances[:-len(descriptor_cols)])
    print(f"Total importance from descriptors: {desc_importance:.3f}")
    print(f"Total importance from fingerprints: {fp_importance:.3f}")

#Calibration curve (save fig) 
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="RF (scaffold hold-out)")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Fraction of actives")
plt.title("Calibration Curve (Random Forest)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "rf_calibration.png"), dpi=300)
plt.close()

#Save deployable RF (fit on ALL data)
deploy_rf = RandomForestClassifier(
    **grid_search.best_params_, random_state=42,
    class_weight="balanced", n_jobs=-1
)
deploy_rf.fit(X, y)

model_name = "rf_ecfp2048.joblib" if not USE_DESCRIPTORS else "rf_model.joblib"
out_path = os.path.abspath(os.path.join(ART_DIR, model_name))
joblib.dump(deploy_rf, out_path)
print("Saved:", out_path)
if not USE_DESCRIPTORS:
    print("MolScore-ready: Use SKLearnModel with fp='ECFP4', nBits=2048 and this model path.")
