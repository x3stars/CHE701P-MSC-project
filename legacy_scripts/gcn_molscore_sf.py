# gcn_molscore_sf.py — minimal DeepChem GCN scorer adapter
from __future__ import annotations
import os, json
from typing import List, Dict
import numpy as np

import deepchem as dc

def _read_best_params(gcn_dir: str) -> dict:
    candidates = [
        os.path.join(gcn_dir, "gcn_cfg.json"),
        os.path.join(gcn_dir, "gcn_scores.json"),
        os.path.join(gcn_dir, "gcn_eval_report.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    j = json.load(f)
                if isinstance(j, dict) and "best_params" in j:
                    return j["best_params"]
                if isinstance(j, dict) and all(k in j for k in ["graph_conv_layers","dense_layer_size"]):
                    return j
            except Exception:
                pass
    return {
        "graph_conv_layers": [96, 96],
        "dense_layer_size": 192,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 5e-6,
        "batch_size": 32,
    }

class GCNDeepChemScorer:
    def __init__(self, prefix: str, gcn_dir: str):
        self.prefix = prefix
        self.gcn_dir = gcn_dir
        self.params = _read_best_params(gcn_dir)

        self.seed_dirs = [d for d in sorted(os.listdir(gcn_dir))
                          if d.startswith("gcn_final_seed") and os.path.isdir(os.path.join(gcn_dir, d))]
        if not self.seed_dirs:
            raise FileNotFoundError(f"No GCN seed directories found in {gcn_dir}")

        self.models = []
        for sd in self.seed_dirs:
            mdir = os.path.join(gcn_dir, sd)
            try:
                model = dc.models.GraphConvModel(
                    n_tasks=1,
                    mode="classification",
                    graph_conv_layers=self.params.get("graph_conv_layers", [96,96]),
                    dense_layer_size=self.params.get("dense_layer_size", 192),
                    dropout=self.params.get("dropout", 0.2),
                    learning_rate=self.params.get("learning_rate", 1e-3),
                    weight_decay=self.params.get("weight_decay", 5e-6),
                    batch_size=self.params.get("batch_size", 32),
                    model_dir=mdir,
                )
                model.restore()
                self.models.append(model)
            except Exception:
                pass

        if not self.models:
            raise RuntimeError(f"Failed to restore any GCN models from {gcn_dir}")

        self.featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)

    def _make_dataset(self, smiles: List[str]) -> dc.data.NumpyDataset:
        X = self.featurizer.featurize(smiles)
        return dc.data.NumpyDataset(X=X, ids=smiles)

    def __call__(self, smiles: List[str]) -> List[Dict[str, float]]:
        ds = self._make_dataset(smiles)
        preds = []
        for model in self.models:
            p = model.predict(ds)
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                preds.append(p[:, 1])
            else:
                preds.append(1 / (1 + np.exp(-p.squeeze())))
        mean_proba = np.mean(np.vstack(preds), axis=0)
        label = f"{self.prefix}_pred_proba"
        return [{"smiles": s, label: float(p)} for s, p in zip(smiles, mean_proba)]

