"""GCN utilities: summarise JSON logs if present (no DeepChem import required)."""
import os, glob, json
def latest_json(gcn_dir:str):
    js = glob.glob(os.path.join(gcn_dir, "**", "*.json"), recursive=True)
    return sorted(js, key=len)[-1] if js else None
def summarise(json_path:str):
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "best_params": cfg.get("best_params"),
        "split_sizes": cfg.get("split_sizes"),
        "test_metrics": cfg.get("test_metrics"),
    }
