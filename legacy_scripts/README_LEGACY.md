# Legacy scripts (verbatim)

This folder preserves **all original .py scripts** exactly as they were, for full provenance.
Use `Mapping_Plan.csv` to track where each script/function is being migrated inside `src/pfdhfr_ml/`.

Columns to fill:
- TargetModule: e.g. `pfdhfr_ml.models.rf` or `pfdhfr_ml.analysis.triage`
- TargetFunction: e.g. `train_rf`, `plot_feature_importances`, `summarise_triaged_set`
- Status: TODO / MIGRATED / DEPRECATED

Do not delete legacy files until the mapping shows everything is migrated.
