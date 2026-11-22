# Trustworthy Pneumonia Detection from Chest X-rays with Grad-CAM Validation

**Project P03 · Team 3 (section-a)**

We train a pneumonia classifier on paediatric chest X-rays and check whether it looks at the **lungs** or relies on **shortcuts** such as text markers, corners and image borders.

> **Status:** Stages 1–3 are complete (Stage 3 merged in [PR #25](https://github.com/Abhishek-Singh-2008/Pneumonia-detection/pull/25)). Stage 4.1 masked training and the Stage 4.2/4.3 Kaggle evaluation are complete; the final comparison and cross-test are recorded in `results/`. The evaluator tests and existing Stage 3 checks pass from a fresh clone; Kaggle data and model weights remain external. The project report remains local. All design decisions are listed in the closing comment of [issue #21](https://github.com/Abhishek-Singh-2008/Pneumonia-detection/issues/21).

---

## Research question

Measure **shortcut learning**:
1. What percentage of Grad-CAM attention falls **inside the lung region**?
2. How does training on **lung-masked** images (everything outside the lungs set to black) change accuracy and attention?

## Approach

| Step | Method |
|---|---|
| Baseline | Custom CNN trained from scratch |
| Stronger model | DenseNet-121 transfer learning, CheXNet style (ImageNet-pretrained, head training, then fine-tuning) |
| Explainability | Grad-CAM heatmaps on both models |
| Attention metric | Sum of Grad-CAM heatmap inside the lung mask ÷ sum over the whole image |
| Masking experiment | Retrain on lung-masked images; cross-test the unmasked model on masked test images |

## Dataset

**Chest X-Ray Images (Pneumonia)**, Kaggle: [`paultimothymooney/chest-xray-pneumonia`](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- 5,856 images, two classes: `NORMAL` and `PNEUMONIA` (paediatric patients).

  | Official folder | NORMAL | PNEUMONIA |
  |---|---|---|
  | `train/` | 1,341 | 3,875 |
  | `val/` | 8 | 8 |
  | `test/` | 234 | 390 |

- The dataset is **not stored in this repo** (see [Setup](#setup-and-how-to-run)).
- The download contains a duplicate copy at `chest_xray/chest_xray/` and a `__MACOSX/` folder. The code reads only the top-level `chest_xray/{train,val,test}`.
- The official `val/` folder has only 16 images, so we build a new **patient-level** validation split (12.5%) from `train/` + `val/`. The official `test/` folder stays the test set. The split CSVs live in `data/splits/`.
- **Patient ID** (used for the split): `PNEUMONIA_person{N}_{bacteria|virus}`, `NORMAL_IM-{XXXX}`, `NORMAL_NORMAL2-IM-{XXXX}`. With this rule the official folders share no patients. Zero overlap is strong evidence, not proof, that `person{N}_bacteria` and `person{N}_virus` are different children.
- **Image ID**: `{class}_{official_folder}_{file_stem}`, e.g. `NORMAL_train_IM-0115-0001`. It is unique across the dataset and names the lung masks (`{image_id}.png`).

## Team and stages

The project runs as four stages in a fixed order. Each member starts when the previous one posts a handover note on their parent issue.

| Stage | Member | Work | Branch | Hands over |
|---|---|---|---|---|
| 1 | Abhishek Singh | Data pipeline, splits, preprocessing, baseline CNN | `stage1-data-baseline` | Splits, seed, baseline results |
| 2 | Ajit Yadav | DenseNet-121 transfer learning, comparison with baseline | `stage2-densenet` | Best model, last conv layer name |
| 3 | Aman Singh | Grad-CAM, lung masks, attention metric, shortcut analysis | `stage3-gradcam` | Lung masks, metric function |
| 4 | Himanshu Kumar | Lung-masked training, final comparison, README, report | `stage4-masking-report` | Final table, README, report |

## Repository structure

The repo is named `Pneumonia-detection`. The team guide suggested `pneumonia-gradcam-shortcut-learning`; we kept the existing name (issue #21, row 11). Branch names are unchanged.

```
Pneumonia-detection/
├── README.md              # project summary, how to run, results
├── CLAUDE.md              # rules and fixed settings for AI coding assistants (useful summary for everyone)
├── requirements.txt       # all libraries with versions (pinned to the Kaggle image)
├── data/                  # dataset NOT pushed to GitHub (see data/README.md)
│   └── splits/            # committed split CSVs + split_summary.json, used by every stage
├── src/
│   ├── config.py          # shared settings: seed, data root, image size, labels, metrics columns, Grad-CAM layers
│   ├── utils.py           # seeding helpers
│   ├── metrics.py         # shared metrics + metrics CSV writer
│   ├── train.py           # shared training loop (early stopping on val loss) and predict()
│   ├── train_baseline.py  # Stage 1.4: train + evaluate the custom CNN
│   ├── train_densenet.py  # Stage 2.1: DenseNet-121 Phase A (head only)
│   ├── finetune_densenet.py  # Stage 2.2: DenseNet-121 Phase B (fine-tuning + tuning runs)
│   ├── evaluate_densenet.py  # Stage 2.3: test evaluation + comparison with the baseline
│   ├── data/              # Stage 1: index.py, splits.py, dataset.py, check_loader.py
│   ├── models/            # Stage 1 & 2: custom_cnn.py, DenseNet-121
│   ├── explain/           # Stage 3 (Aman): Grad-CAM, lung-attention metric
│   └── masking/           # lung_masks.py: Stage 3.2 (Aman); masked dataset/training: Stage 4 (Himanshu)
├── notebooks/             # notebooks per stage: 01_, 02_, 03_, 04_
├── results/               # committed metrics tables and figures (see results/README.md)
└── report/                # draft LaTeX sections + notes for the Overleaf report (see report/README.md)
```

## Setup and how to run

**Framework:** PyTorch. **Where we run:** Kaggle notebooks (free GPU, dataset attaches directly). The same code runs on Colab or locally by pointing it at the data.

### On Kaggle (recommended)
1. Create a new notebook, or upload one from `notebooks/`.
2. **Add Input** → search `chest-xray-pneumonia` → add *Chest X-Ray Images (Pneumonia)* by Paul Mooney.
3. Settings: turn **Internet on** (the first cell clones this repo). Turn a GPU on for training notebooks.
4. Run All. The first cell clones the repo and the data root is found automatically.

### On Colab or locally
```bash
git clone https://github.com/Abhishek-Singh-2008/Pneumonia-detection.git
cd Pneumonia-detection
pip install -r requirements.txt
# Download the dataset (needs a Kaggle API token in ~/.kaggle/)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
```
After unzipping, `data/chest_xray/{train,val,test}/` must exist. Data stored elsewhere can be used by setting `PNEUMONIA_DATA_ROOT` to the folder that contains `train/`, `val/` and `test/`.

### Notebooks
| Notebook | Stage | What it does |
|---|---|---|
| `notebooks/01_eda.ipynb` | 1.1 | Class counts, patients, image sizes and modes, sample grid → `results/stage1_eda/` |
| `notebooks/03_gradcam_masks.ipynb` | 3 | Self-checks, lung masks for all images, Grad-CAM + attention metric for both models, figures → `results/stage3/` |

### Splits (Stage 1.2)
The split CSVs are already committed in `data/splits/`. **Every stage uses these files; do not make new splits.**

| Split | Images | NORMAL | PNEUMONIA | Patients |
|---|---|---|---|---|
| train | 4,578 | 1,180 | 3,398 (74.2%) | 3,218 |
| val | 654 | 169 | 485 (74.2%) | 451 |
| test (official) | 624 | 234 | 390 (62.5%) | 428 |

- Patient-level and stratified: no patient appears in two splits. The val split is 12.5% of each class in the train + official-val pool.
- Stricter grouping for train/val (PR #23 review): all pneumonia images of `person{N}` (bacterial **and** viral) are kept on the same side, so train and val share no `person{N}` even if both types belong to one child. The `patient_id` column keeps the #21 rule.
- Columns: `image_id, filepath, label, label_name, patient_id, official_folder, split`. `filepath` is relative to the data root.
- `data/splits/split_summary.json` holds the counts and the class weight `pos_weight` = 0.347263 (#NORMAL / #PNEUMONIA in train), for `BCEWithLogitsLoss`.
- In code: `from src.data.splits import load_split, load_pos_weight`.
- To regenerate (gives identical files on any OS and library version): `python -m src.data.splits`.

### Data loader (Stage 1.3)
```python
from src.data.dataset import get_dataloaders
loaders = get_dataloaders(batch_size=32)          # dict: train / val / test
for images, labels, image_ids in loaders["train"]:  # images: (B, 3, 224, 224)
    loss = criterion(model(images).squeeze(1), labels.float())
```
- Every image: grayscale → 3 channels → plain resize to 224×224 → ImageNet mean/std.
- Train only: random rotation ±10°. No horizontal flip.
- Only train is shuffled. Shuffling and workers are seeded; call `src.utils.set_seed()` at the start of every script.
- Check: `python -m src.data.check_loader` loads every image once → `results/stage1_loader/`.

### Baseline CNN (Stage 1.4)
```bash
python -m src.train_baseline          # GPU recommended; a few minutes on a Kaggle T4
```
- Model: `src/models/custom_cnn.py`, 5 conv blocks → global average pooling → 1 logit (2.35 M parameters).
- Grad-CAM layer: `features.block5.conv2` (`src.config.LAST_CONV_LAYER["custom_cnn"]`).
- Training: `BCEWithLogitsLoss(pos_weight)`, Adam (lr 1e-3, weight decay 1e-4), batch 32, ReduceLROnPlateau on val loss, early stopping on **val loss** (patience 5). The test set is evaluated once with the best checkpoint.
- Outputs: `results/stage1_metrics.csv`; `results/stage1_baseline/` (training history, val/test predictions per `image_id`, curves, confusion matrices, ROC, settings).
- Weights: `custom_cnn_best.pth` in the [shared Google Drive folder](https://drive.google.com/drive/folders/1-IldsdQnswOB-PVVcwLyYlijN_4TlTYA?usp=sharing). Load with:
  ```python
  ckpt = torch.load("custom_cnn_best.pth", map_location="cpu", weights_only=False)
  model = CustomCNN(); model.load_state_dict(ckpt["model_state"])
  ```
- The shared training loop `src/train.py` and metrics `src/metrics.py` are reused in Stage 2.

### DenseNet-121 (Stage 2)
CheXNet-style transfer learning on the same splits, loader and metrics as the baseline. Run in this order (GPU recommended; each step takes a few minutes on a Kaggle T4):
```bash
python -m src.train_densenet                                           # 2.1 Phase A: backbone frozen, only the new 1-logit layer
python -m src.finetune_densenet --run-name lr1e-4_bs16 --batch-size 16 # 2.2 Phase B: whole network, low learning rate
python -m src.finetune_densenet --run-name lr1e-4                      # other tuning runs (see the tuning table below)
python -m src.finetune_densenet --run-name lr5e-5 --lr 5e-5
python -m src.evaluate_densenet                                        # 2.3 test evaluation + comparison with the baseline
```
- Model: `src/models/densenet121.py`, ImageNet-pretrained DenseNet-121, final 1000-class layer replaced by one logit (6.95 M parameters). Same preprocessing as the baseline (ImageNet mean/std, grayscale → 3 channels), so nothing in the pipeline changes.
- **Phase A** (`train_densenet.py`): backbone frozen (BatchNorm kept in eval mode), only 1,025 parameters trained. Adam lr 1e-3, batch 32, early stopping on val loss. Val AUC 0.990.
- **Phase B** (`finetune_densenet.py`): starts from the Phase A weights, all layers trained with a low learning rate (Adam, weight decay 1e-4, ReduceLROnPlateau, early stopping on val loss, patience 4).
- **Tuning (val only):** three runs, the best one is chosen by the lowest **val loss**. The val AUCs (0.998 to 0.9995) are too close to separate the runs, which is why val loss is the selection criterion. The test set is not used for any choice and is evaluated once, for the chosen model.

  | Run | Learning rate | Batch size | Best epoch | Val loss | Val AUC |
  |---|---|---|---|---|---|
  | `lr1e-4_bs16` (chosen) | 1e-4 | 16 | 3 | 0.0139 | 0.9995 |
  | `lr1e-4` | 1e-4 | 32 | 2 | 0.0206 | 0.9994 |
  | `lr5e-5` | 5e-5 | 32 | 8 | 0.0278 | 0.9984 |

- Grad-CAM layer: `relu`, the named ReLU module after the last feature map. Its output is the **post-ReLU** 1024 × 7 × 7 map (`src.config.LAST_CONV_LAYER["densenet121"]`; changed from `features.norm5` after the PR #24 review). It is not in-place, so Grad-CAM hooks see unmodified values. Get it with `model.get_submodule("relu")`. The module has no parameters, so the saved weights load unchanged.
- Outputs: `results/stage2_metrics.csv` (val and test, agreed columns), `results/stage2_comparison.csv` (baseline vs DenseNet-121 on the test set), `results/stage2_phaseA/`, `results/stage2_phaseB/` (every run + `tuning.csv`), `results/stage2_densenet/` (predictions per `image_id`, confusion matrices, ROC, settings).
- Weights: `densenet121_phaseA_best.pth` and `densenet121_phaseB_lr1e-4_bs16_best.pth` (final model) in the shared Google Drive folder. Load with:
  ```python
  from src.models.densenet121 import DenseNet121Classifier
  ckpt = torch.load("densenet121_phaseB_lr1e-4_bs16_best.pth", map_location="cpu", weights_only=False)
  model = DenseNet121Classifier(pretrained=False); model.load_state_dict(ckpt["model_state"])
  ```

### Lung masks, Grad-CAM and the attention metric (Stage 3)
Easiest: run `notebooks/03_gradcam_masks.ipynb` on Kaggle (GPU and Internet on). The same steps from a terminal:
```bash
pip install torchxrayvision                       # pretrained lung segmentation model (not preinstalled on Kaggle)
python -m src.explain.check_explain               # self-checks on synthetic data, no dataset needed
python -m src.masking.lung_masks --zip            # 3.2 lung masks for all 5,856 images + eye-check figures
python -m src.explain.run_gradcam --checkpoint-dir <folder with both .pth files> --mask-dir <lung mask folder>   # 3.3
```
- **Lung masks** (`src/masking/lung_masks.py`): torchxrayvision's pretrained PSPNet (ChestX-Det, adult X-rays); its left and right lung outputs are joined. The whole image is resized without cropping, exactly like the classifier input, so masks line up pixel by pixel with the heatmaps. Saved as `{image_id}.png`, 224 × 224, binary (0 = background, 255 = lung, #21 row 14). Read them with `load_mask(mask_dir, image_id)` / `load_masks(...)`. The masks are on the team Drive ([`lung_masks.zip`](https://drive.google.com/file/d/1ijzEW5sEwMmEbyUQfpTLJ7Oehl5IKqpr/view?usp=sharing)), not in the repo.
  - **Known limitation:** the segmentation model was trained on adult X-rays. On a pneumonia image where a lung is white from consolidation, it can miss most of that lung (seen on `PNEUMONIA_test_person100_bacteria_475` during testing). Heatmap mass on the missed part then counts as *outside* the lung, so `attention_in_lung` can be too low for pneumonia images. Such masks are flagged in `results/stage3/lung_mask_summary.csv` (one lung under 5% of the image or under 40% of the other lung) and shown in `lung_masks_flagged.png` for the eye check.
- **Grad-CAM** (`src/explain/gradcam.py`): one function `grad_cam(model, images, layer)` for both models. It explains the **predicted** class: +logit when the image is predicted PNEUMONIA, −logit when predicted NORMAL (#21). Layers (post-ReLU, `GRADCAM_LAYER`): custom CNN `features.block5.relu2` (256 × 14 × 14), DenseNet-121 `relu` (1024 × 7 × 7). The custom CNN entry in `src/config.py` (`features.block5.conv2`) is the same convolution before its BatchNorm and ReLU; the post-ReLU output is the standard Grad-CAM input, so Stage 3 uses `relu2`.
- **Attention metric** (`src/explain/attention.py`): `attention_in_lung(cams, masks)` = heatmap inside the lung mask / whole heatmap. It is reported next to its chance level `lung_area` (what an evenly spread heatmap would score) and `border_fraction` (share in the outer 10% band, where text markers sit). Stage 4 calls the same function on the masked model.
- `run_gradcam.py` runs both models on all 624 test images, first checks that its test metrics equal the committed Stage 1/2 numbers (right weights loaded), and fills `attention_in_lung` in `results/stage3_metrics.csv`. Figures use a fixed sample of 50 test images (25 per class, up to 8 per class misclassified by at least one model, the same images for both models): `results/stage3/sample_images.csv`.

### Lung-masked DenseNet-121 and cross-test (Stage 4)
Stage 4.1 trained and selected the masked DenseNet-121 using validation loss only. The saved training summary confirms the Stage 2 settings and the 4,578/654/624 split alignment; no test metrics were computed during training.

On Kaggle, attach the Chest X-Ray Images (Pneumonia) dataset and run the complete Stage 4.1 pipeline from a fresh checkout. The default is a one-epoch-per-phase smoke run; set `STAGE4_SMOKE=0` for the full training:
```bash
git clone --branch stage4-masking-report https://github.com/Abhishek-Singh-2008/Pneumonia-detection.git /kaggle/working/Pneumonia-detection
STAGE4_SMOKE=0 STAGE4_REPO=/kaggle/working/Pneumonia-detection \
  python /kaggle/working/Pneumonia-detection/stage_4_masked_densenet_121_training.py
```
The runner reuses or downloads the Stage 3 masks, creates masked copies, trains Phase A then Phase B, and writes `/kaggle/working/stage4_bundle.zip`. Download that bundle to retain the checkpoints and results. The masked image copies and masks are temporary Kaggle scratch data under `/kaggle/temp/stage4/`; they are not included in the bundle and can be regenerated from the original dataset and shared masks.

Run the final masked test evaluation, the Stage 2 unmasked-checkpoint cross-test on masked test images, and Grad-CAM attention for the masked model while the scratch data is still available:
```bash
python -m src.evaluate_masked --data-root /kaggle/temp/stage4/masked_chest_xray --mask-dir /kaggle/temp/stage4/lung_masks --checkpoint-root /kaggle/working/checkpoints
```
The checkpoint folder must contain `densenet121_masked_phaseB_best.pth` and `densenet121_phaseB_lr1e-4_bs16_best.pth`. The script verifies their saved training metadata and prediction alignment with the committed test split. The masked model's single test pass also produces Grad-CAM, avoiding a second pass for its metrics.

Outputs: `results/stage4_metrics.csv` (matched masked-model test metrics and lung attention), `results/stage4_cross_test_metrics.csv` (unmasked-trained model evaluated on masked images), `results/stage4_comparison.csv` (baseline, unmasked DenseNet and masked DenseNet), and per-image predictions/attention in `results/stage4_masked_densenet/`. The evaluator ran on Kaggle GPU over the 624-image committed test split; the test set was not used for training or model selection.

## Shared files (Google Drive)

Team folder: https://drive.google.com/drive/folders/1-IldsdQnswOB-PVVcwLyYlijN_4TlTYA?usp=sharing

| File | Stage | What it is |
|---|---|---|
| `custom_cnn_best.pth` | 1 | Custom CNN baseline, best epoch by val loss |
| `densenet121_phaseA_best.pth` | 2 | DenseNet-121 after Phase A (head only) |
| `densenet121_phaseB_lr1e-4_bs16_best.pth` | 2 | DenseNet-121 final model (fine-tuned, chosen by val loss) |
| [`lung_masks.zip`](https://drive.google.com/file/d/1ijzEW5sEwMmEbyUQfpTLJ7Oehl5IKqpr/view?usp=sharing) | 3 | Lung masks for all 5,856 images: `{image_id}.png`, 224 × 224, 0 = background, 255 = lung, plus `areas.csv` (left/right lung area). Unzip and pass the folder as `mask_dir` to `src.masking.lung_masks.load_mask` |

Stage 4 checkpoints are not stored in GitHub; the Kaggle runner includes them in `stage4_bundle.zip`. Share that bundle with the team if the weights are needed later.

## Results

*Final test-set metrics on the committed 624-image test split.*

| Model | Training data | Accuracy | Recall | F1 | AUC | Lung attention % |
|---|---|---|---|---|---|---|
| Custom CNN | Unmasked | 0.8013 | 0.9821 | 0.8607 | 0.9016 | 43.4% |
| DenseNet-121 | Unmasked | 0.8734 | 0.9974 | 0.9078 | 0.9713 | 39.5% |
| DenseNet-121 | Lung-masked | 0.8702 | 0.9949 | 0.9055 | 0.9759 | 51.8% |

Full metrics (with precision and the confusion matrix, val and test) are in `results/stageN_metrics.csv`.
Lung attention = mean share of the Grad-CAM heatmap inside the lung mask over the 624 test images (`results/stage3_metrics.csv`, `results/stage4_metrics.csv`). The lungs cover 37.7% of the image on average, which is what an evenly spread heatmap would score.

### Findings so far
- **Val–test gap (Stage 1):** the baseline reaches 0.968 accuracy on val but 0.801 on test. It predicts 117 of 234 NORMAL test images as pneumonia while missing only 7 of 390 pneumonia images. The val split comes from the same distribution as train; the official test set differs. Nothing was tuned on test.
- **DenseNet-121 vs baseline (Stage 2):** on the same test set, DenseNet-121 raises accuracy from 0.801 to 0.873 and AUC from 0.902 to 0.971. False positives drop from 117 to 78 and only 1 of 390 pneumonia images is missed (baseline: 7). The ImageNet-pretrained backbone gives better features than a CNN trained from scratch on about 4,500 images. A val–test gap remains, as in Stage 1: val accuracy 0.985 vs test 0.873, and precision 0.998 on val vs 0.833 on test. 78 of 234 NORMAL test images (about a third) are still predicted as pneumonia, which makes them a good target for the Grad-CAM analysis. The val numbers (AUC 0.9995) are very high because val comes from the same distribution as train, so we do not rely on val alone: the test set is the number that matters.
- **Possible shortcuts in the raw data ([#22](https://github.com/Abhishek-Singh-2008/Pneumonia-detection/issues/22)):** NORMAL images are wider than tall by about 1.2×, PNEUMONIA images by about 1.5×, so the plain 224×224 resize distorts the two classes differently. All RGB files are PNEUMONIA; this cue is removed by converting every image to grayscale first.
- **Where the models look (Stage 3):** both models put only a little more of their Grad-CAM heatmap on the lungs than an evenly spread heatmap would: 43.4% (custom CNN) and 39.5% (DenseNet-121) against a lung area of 37.7% (`results/stage3/attention_summary.csv`). The more accurate DenseNet-121 is the closer of the two to chance, especially on PNEUMONIA images (37.5% vs 36.7% lung area). Its most confident false positives (NORMAL test images predicted PNEUMONIA with p = 1.00) put the heatmap on burned-in text: the "KV58"/"KV59"/"KV60" exposure labels in the bottom-right corner and timestamps in the top-right corner (`results/stage3/false_positives_densenet121.png`). This is direct evidence of shortcut learning. The same corner text also drives some *correct* PNEUMONIA predictions (right answer for the wrong reason), and the custom CNN often puts a hot spot on the "R" laterality marker; eight hand-picked examples are in `results/stage3/shortcut_examples.png`. Lung masks: 226 of 5,856 are flagged as implausible, 225 of them PNEUMONIA images where consolidated lung was missed; only 19 are test images, and leaving them out changes the averages by less than 0.01 (`results/stage3/attention_robustness.csv`). The flag misses partial errors, though: by eye, 19 of 24 random masks are correct and the 5 imperfect ones (all PNEUMONIA) include 4 unflagged ones (`results/stage3/lung_mask_eye_check.csv`), so lung attention on PNEUMONIA images is probably underestimated.
- **Effect of lung masking (Stage 4):** on their matched test inputs, masked and unmasked DenseNet-121 are similar: accuracy is 0.8702 vs 0.8734, F1 is 0.9055 vs 0.9078, and AUC is 0.9759 vs 0.9713. Mean lung attention rises from 39.52% to 51.79%; the masked model's attention is 1.40 times the mean lung area (37.70%) (`results/stage4_metrics.csv`, `results/stage4_comparison.csv`, `results/stage4_masked_densenet/masked_attention_summary.json`). In the cross-test, the unmasked checkpoint on masked images reaches 0.6314 accuracy and 0.8625 AUC, with 230 false positives among 234 NORMAL images (`results/stage4_cross_test_metrics.csv`). This indicates sensitivity to removing extra-pulmonary image context, but does not by itself establish which feature caused the change. The 19 flagged test masks remain included, so the segmentation limitation still applies.

## Working rules

- `main` is kept working at all times. Each member works on their own branch and merges through a pull request.
- PR reviews: Ajit reviews Stage 1, Aman reviews Stage 2, Himanshu reviews Stage 3, Abhishek reviews Stage 4.
- Never commit the dataset or large model weights. They're shared through the [team Google Drive folder](https://drive.google.com/drive/folders/1-IldsdQnswOB-PVVcwLyYlijN_4TlTYA?usp=sharing).
- One fixed random seed (42), kept in `src/config.py`; call `src.utils.set_seed()` at the start of every script.
- All shared settings come from `src/config.py`; never hardcode paths or values.
- Use the committed split CSVs; never make new splits. The test set is only used to report final numbers, never to choose anything.
- Every number in the report comes from a file in `results/`.
- Every stage has one parent issue split into child issues, labelled `stage-N` + `parent` / `child`.
- **Work boundaries:** everyone works only on their own stage's issues, branch and files. Shared Stage 1 code (`src/config.py`, `src/train.py`, `src/metrics.py`, `src/data/`) is add-only; changing existing behaviour needs an issue. Details on who owns which files: [CLAUDE.md](CLAUDE.md#work-boundaries-who-does-what).
- Before submission, the code must run from a clean clone using only this README.
- **Every code file has detailed comments:** a module docstring, a docstring on every function and class, and inline comments that explain *why* (rules in [CLAUDE.md](CLAUDE.md#code-comments-required-in-every-file)).
- Commit messages follow a short prefix style (`feat:`, `fix:`, `docs:`, `chore:`), e.g. `docs: update setup notes`.

## References

1. P. Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, 2017. [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)
2. R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
