# results/

Committed outputs of every stage. **The report takes all figures and numbers from here**, so the report always matches the repo.
Model weights are not stored here; they're in the team Google Drive folder (see the main README).

## Conventions
- One metrics file per stage: `stageN_metrics.csv`, with the agreed columns (issue #21, row 8):
  `model, data_setting, split, accuracy, precision, recall, f1, auc, tn, fp, fn, tp, attention_in_lung`.
  Write it with `src.metrics.save_metrics` so the format stays identical. `data_setting` is `unmasked` or `masked`.
- Everything else from a stage goes in a folder named `stageN_<topic>/`.
- Predictions are saved per `image_id`, so later stages can join them with the split CSVs and lung masks.

## Stage 1 (Abhishek)
| Path | Made by | Contents |
|---|---|---|
| `stage1_eda/` | `notebooks/01_eda.ipynb` | `class_counts.csv/.png`, `patients.csv`, `images_per_patient.png`, `image_size_summary.csv`, `image_modes.csv`, `image_sizes.png`, `sample_grid.png` |
| `stage1_loader/` | `python -m src.data.check_loader` | `loader_check.json` (all splits load, channels, reproducibility), `augmentation_samples.png` |
| `stage1_metrics.csv` | `python -m src.train_baseline` | Custom CNN, val and test rows |
| `stage1_baseline/` | `python -m src.train_baseline` | `history.csv`, `settings.json`, `val_predictions.csv`, `test_predictions.csv` (`image_id, label, prob_pneumonia, pred`), `training_curves.png`, `confusion_matrices.png`, `roc.png` |

## Later stages
| Path | Stage | Expected contents |
|---|---|---|
| `stage2_metrics.csv` | 2 (Ajit) | DenseNet-121 val/test rows + comparison table with the baseline |
| `stage3/` | 3 (Aman) | Grad-CAM heatmaps, attention-metric table, shortcut examples |
| `stage4_*` | 4 (Himanshu) | Masked-model metrics, cross-test, final comparison table |
