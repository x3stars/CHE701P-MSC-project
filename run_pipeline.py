import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pfdhfr_ml.models import rf as RF
from pfdhfr_ml.analysis import triage as TA
import os
import pandas as pd
import matplotlib.pyplot as plt

# make output folders
os.makedirs("reports", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 1) Load pre-trained RF and save top-25 fingerprint importances
rf = RF.load_rf("artifacts/metrics_rf/rf_ecfp2048.joblib")
top = RF.top_feature_importances(rf, 25)
top.to_csv("reports/rf_feature_importances_top25.csv", header=["importance"])

ax = top.plot(kind="bar", figsize=(10,4))
ax.set_ylabel("Gini importance")
ax.set_title("RF Feature Importance (Top 25)")
plt.tight_layout()
plt.savefig("figures/rf_feature_importances_top25.png", dpi=200)
plt.close()

print("Top-5 importances:\n", top.head(5))

# 2) Triage evolved candidates (novelty vs. training + filters)
tri = TA.triage_evolved(
    "artifacts/evolve_rf/rf_gcn_triaged_top500.csv",
    "reports/triaged_pfDHFR.csv",
    "data/pfDHFR_cleaned_with_potency_descriptors.csv",
    train_col="Smiles"
)
print("Triage summary:", tri)

# 3) Review candidates (props/alerts + optional RF score + ensemble if present)
rev = TA.review_candidates(
    "reports/triaged_pfDHFR.csv",
    "reports/reviewed_candidates.csv",
    rf_model="artifacts/metrics_rf/rf_ecfp2048.joblib"
)
print("Review summary:", rev)
