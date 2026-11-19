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

## Work boundaries: who does what
Each member owns one stage (team project guide). **Work only on your own stage's issues, branch and files.**

| | Abhishek Singh | Ajit Yadav | Aman Singh | Himanshu Kumar |
|---|---|---|---|---|
| GitHub login | `Abhishek-Singh-2008` | `Ajit2908` | `amansingh1207` | `Himanshu-86` |
| Stage | 1: data pipeline + baseline CNN | 2: DenseNet-121 transfer learning | 3: Grad-CAM, lung masks, attention metric, shortcut analysis | 4: lung-masked training, final comparison, README, report |
| Branch | `stage1-data-baseline` | `stage2-densenet` | `stage3-gradcam` | `stage4-masking-report` |
| Issues | #1–#5, #22 | #6 (parent), #7–#9 | #10 (parent), #11–#14 | #15 (parent), #16–#20 |
| Owns (code) | `src/config.py`, `src/utils.py`, `src/metrics.py`, `src/train.py`, `src/train_baseline.py`, `src/data/`, `src/models/custom_cnn.py` | `src/models/densenet*.py`, its training script (e.g. `src/train_densenet.py`) | `src/explain/`, `src/masking/lung_masks.py` (mask generation and loading) | `src/masking/` except `lung_masks.py` (masked dataset, masked training, cross-test) |
| Owns (data/results) | `data/splits/`, `notebooks/01_*`, `results/stage1_*` | `notebooks/02_*`, `results/stage2_*` | `notebooks/03_*`, `results/stage3/`; lung masks on Drive | `notebooks/04_*`, `results/stage4_*`; final comparison table |
| Report sections | Introduction; dataset, preprocessing and baseline | DenseNet-121 method, training setup, model comparison | Grad-CAM, lung masks, attention metric, shortcut findings | Masking experiment, final results, discussion, abstract, conclusion; Overleaf setup and final read-through |
| Reviews the PR of | Stage 4 (Himanshu) | Stage 1 (Abhishek) | Stage 2 (Ajit) | Stage 3 (Aman) |
| Hands over | splits, seed, loader, baseline results → Ajit (and Aman) | best model + `LAST_CONV_LAYER["densenet121"]` → Aman | lung masks, attention-metric function, Grad-CAM function → Himanshu | final table, README, report → whole team |

**Shared by everyone:** one Related Work paragraph each (on the papers used in your stage), your own entries in the shared `.bib`;
in `README.md` each member edits only their own stage's section (Himanshu finalises the whole README in Stage 4.4);
`requirements.txt` is add-only (add a library your stage needs, never change existing pins without an issue).
Repo administration (labels, branch rules, collaborators, Drive folder) is done by Abhishek.

**Order and dependencies.** A stage starts when the previous member posts the handover note on their parent issue.
Exceptions from the guide: Stages 3.1 and 3.2 can start after the Stage 1 handover; Stage 4.5 (Overleaf setup) can start on day one;
Stage 4.1 needs Stage 3.2 (masks) and the Stage 2 handover. While waiting, read your papers and draft your report section.
If blocked: add the `blocked` label to your issue, comment what you need, and tag the person responsible.

### Shared Stage 1 code is add-only
`src/config.py`, `src/utils.py`, `src/metrics.py`, `src/train.py`, `src/data/*` are used by every stage and owned by Abhishek.
- Others may **add** their own entries (e.g. Ajit sets `LAST_CONV_LAYER["densenet121"]`; a new helper used only by their stage).
- **Changing existing behaviour** (splits, preprocessing, augmentation, metrics, the training loop, existing settings) needs an issue first,
  and Abhishek reviews the change. Prefer wrapping or extending in your own files over editing shared ones.
- `data/splits/` is never edited by anyone.

### Rules for Claude (and other AI assistants)
1. **Know who you are working for.** Before changing files, issues or PRs, identify the user with `gh api user --jq .login`
   (fallback: `git config user.name`) and map it to the table above. If it can't be determined, ask.
2. **Stay in the user's lane.** Work on that member's stage: their issues, their branch, the files they own.
3. **Someone else's task:** if asked to do work that belongs to another member (their files, issues, branch or report section),
   first **warn** the user, naming the owner and the issue (e.g. "Grad-CAM is Aman's task, #11"). Continue only if the user
   explicitly confirms, and suggest telling the owner on their issue so the work isn't duplicated.
4. **Shared code:** follow the add-only rule above. Don't silently change existing behaviour, even when asked; open or point to an issue.
5. Read-only help is always fine: explaining code, answering questions, and reviewing a PR the user was asked to review.
6. Close only issues that belong to the user, and only when their checklist and "Done when" points are true.

## Code comments (required in every file)
The code is read by teammates, reviewers and the instructor, so **every file must be understandable without asking its author.**
- **Module docstring** at the top of every `.py` file: what the file does, which stage and issue it belongs to,
  which #21 decisions it implements, how to run it (`python -m ...`) and what it writes.
- **Docstring on every function and class:** purpose, arguments (with shapes/types for tensors and DataFrames),
  return value, and any side effects (files written, global RNG used).
- **Inline comments explain *why*, not just *what*:** the reason for a choice (e.g. "no horizontal flip: it mirrors the
  L/R markers we study"), links to the decision (`#21 row 5`, `#22`, `PR #23`), and anything non-obvious
  (tensor shapes, seeding, why the test set is not touched, units).
- Comment every non-trivial block (loops over data, training steps, metric maths, file formats).
- Write comments in plain English that a classmate new to the code can follow. Keep them up to date when the code changes;
  a wrong comment is worse than none.
- Notebooks: a markdown cell before each step explaining what it does and why, plus comments inside code cells.

## Hard rules
- **Never push to `main`.** It is protected; work on your stage branch and merge through a PR
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
| Patient ID | `PNEUMONIA_person{N}_{bacteria\|virus}`, `NORMAL_IM-{XXXX}`, `NORMAL_NORMAL2-IM-{XXXX}`; the train/val split groups by `person{N}` (both types together, `split_group()`) |
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
- `src/explain/` (Aman, Stage 3: Grad-CAM, attention metric), `src/masking/lung_masks.py` (Aman, Stage 3.2: lung masks),
  rest of `src/masking/` (Himanshu, Stage 4: masked dataset and training): to be filled.
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
- Val–test gap: the baseline gets 0.97 val accuracy but 0.80 test accuracy (many NORMAL test images are predicted as pneumonia). The official test set has a different distribution.
- Possible shortcuts in the raw data (#22): aspect ratio differs by class (~1.2 normal vs ~1.5 pneumonia), and the plain resize distorts the classes differently. This is kept as agreed and documented as a confound. All RGB files are PNEUMONIA; this cue is removed by converting to grayscale first.
- Zero patient overlap is strong evidence, not proof, that `person{N}_bacteria` and `person{N}_virus` are different children. Mention it in the limitations section.
