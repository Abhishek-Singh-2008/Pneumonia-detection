# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## Project
P03, Team 3: **Trustworthy Pneumonia Detection from Chest X-rays with Grad-CAM Validation**.
Train a pneumonia classifier (custom CNN → DenseNet-121) on the Kaggle dataset `paultimothymooney/chest-xray-pneumonia`,
then measure **shortcut learning**: the % of Grad-CAM attention inside the lungs, and how lung-masked training changes accuracy.

- Repo: `Abhishek-Singh-2008/Pneumonia-detection` (lowercase `d`; the team guide's suggested name was not used).
- Four sequential stages, one member each: 1 Abhishek (data + baseline CNN), 2 Ajit (DenseNet-121),
  3 Aman (Grad-CAM, lung masks, attention metric), 4 Himanshu (masked training, final table, report).
- **All design decisions are final in the closing comment of issue #21.** Changes go through a new issue, never silently.
  EDA findings on possible shortcuts: issue #22.

## Hard rules
- **Never push to `main`.** It is protected; work on the stage branch and merge through a PR
  (`stage1-data-baseline`, `stage2-densenet`, `stage3-gradcam`, `stage4-masking-report`).
  PR reviews: Ajit → Stage 1, Aman → Stage 2, Himanshu → Stage 3, Abhishek → Stage 4.
- **No AI attribution** in commits, PRs or issues: no `Co-Authored-By: Claude`, no "Generated with Claude Code".
- Commit messages: short prefix style: `feat:`, `fix:`, `docs:`, `chore:` (e.g. `feat: add DenseNet-121 head training`).
- Never commit the dataset, model weights (`*.pth`, `checkpoints/`) or secrets (`kaggle.json`, `access_token`).
  Weights, masks and other large files go to the team Google Drive folder linked in the README.
- **Never make new splits.** Use `data/splits/{train,val,test}.csv` (`src.data.splits.load_split`).
- **The test set is never used to choose anything** (model, epoch, threshold, hyperparameters); it is only evaluated
  once with the final model. Model selection uses val only.
- Every number in the report must come from a file in `results/`.

## Fixed settings (from `src/config.py`; do not hardcode them elsewhere)
| Setting | Value |
|---|---|
| Seed | 42; call `src.utils.set_seed()` at the start of every script |
| Labels | NORMAL = 0, PNEUMONIA = 1 (positive class) |
| Output head | one logit + `BCEWithLogitsLoss(pos_weight)`, `pos_weight` = 0.347263 from `data/splits/split_summary.json` |
| Threshold | 0.5 (if ever tuned: on val only, recorded in config) |
| Preprocessing | grayscale → 3 channels, plain resize 224×224 (no pad/crop), ImageNet mean/std |
| Augmentation | train only, rotation ±10°, **no horizontal flip** |
| Patient ID | `PNEUMONIA_person{N}_{bacteria\|virus}`, `NORMAL_IM-{XXXX}`, `NORMAL_NORMAL2-IM-{XXXX}` |
| Image ID | `{class}_{official_folder}_{file_stem}`; lung masks are `{image_id}.png` (binary, 224×224, nearest-neighbour) |
| Metrics CSV | `results/stageN_metrics.csv` with columns `model, data_setting, split, accuracy, precision, recall, f1, auc, tn, fp, fn, tp, attention_in_lung` |
| Grad-CAM layers | `LAST_CONV_LAYER` in config: `custom_cnn` → `features.block5.conv2`; `densenet121` is set in Stage 2 |

## Code map
- `src/config.py`: all shared settings; `get_data_root()` finds the data on Kaggle, via `PNEUMONIA_DATA_ROOT`, or in `data/chest_xray/`.
- `src/utils.py`: `set_seed`, `seed_worker`.
- `src/data/index.py`: `build_index(root)` (one row per image, patient-ID rule).
- `src/data/splits.py`: split method, `load_split`, `load_pos_weight`; `python -m src.data.splits` regenerates identical CSVs.
- `src/data/dataset.py`: `ChestXrayDataset`, `build_transforms`, `get_dataloaders(..., cache=True)`; items are `(image, label, image_id)`.
- `src/metrics.py`: `compute_metrics`, `metrics_row`, `save_metrics`. Reuse them so every stage reports the same way.
- `src/train.py`: `fit` (early stopping on val loss, AMP, checkpoints), `predict`, `make_criterion`. Reuse for DenseNet-121.
- `src/models/custom_cnn.py`, `src/train_baseline.py`: Stage 1 baseline.
- `src/explain/` (Stage 3), `src/masking/` (Stage 4): to be filled.
- `notebooks/0N_*.ipynb`: one or more notebooks per stage. `results/`: committed outputs (see `results/README.md`).

## Running
- **Compute is Kaggle** (free T4 GPU). Notebooks/scripts clone the repo, so the Kaggle account needs phone
  verification for Internet. The dataset mounts at `/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray`
  (the old `/kaggle/input/chest-xray-pneumonia/chest_xray` also works); `get_data_root()` finds either.
- Kaggle image: Python 3.13, torch 2.11.0, torchvision 0.26.0, scikit-learn 1.6.1 (pinned in `requirements.txt`).
  Code must not depend on library-version-specific randomness (this is why the split uses NumPy's `RandomState`).
- Run modules from the repo root: `python -m src.train_baseline`, `python -m src.data.check_loader`.
- Load checkpoints with `torch.load(path, map_location="cpu", weights_only=False)`. They are dicts with
  `model_state`, `epoch`, `val_loss`, `val_auc` and `extra` (all training settings).

## Known findings to keep in mind
- Val–test gap: the baseline gets 0.97 val accuracy but 0.78 test accuracy (many NORMAL test images are predicted as pneumonia). The official test set has a different distribution.
- Possible shortcuts in the raw data (#22): aspect ratio differs by class (~1.2 normal vs ~1.5 pneumonia), and the plain resize distorts the classes differently. This is kept as agreed and documented as a confound. All RGB files are PNEUMONIA; this cue is removed by converting to grayscale first.
- Zero patient overlap is strong evidence, not proof, that `person{N}_bacteria` and `person{N}_virus` are different children. Mention it in the limitations section.
