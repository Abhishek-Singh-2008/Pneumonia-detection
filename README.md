# Trustworthy Pneumonia Detection from Chest X-rays with Grad-CAM Validation

**Project P03 · Team 3 (section-a)**

We train a pneumonia classifier on paediatric chest X-rays and check whether it looks at the **lungs** or relies on **shortcuts** such as text markers, corners and image borders.

> **Status:** Stage 1 in progress. All design decisions are final and listed in the closing comment of [issue #21](https://github.com/Abhishek-Singh-2008/Pneumonia-detection/issues/21). The setup, run steps and results below get filled in as each stage is finished.

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
├── README.md           # project summary, how to run, results
├── requirements.txt    # all libraries with versions
├── data/               # dataset NOT pushed to GitHub; split CSVs in data/splits/ are
├── src/
│   ├── config.py       # shared settings: seed, data root, image size, labels, metrics columns
│   ├── utils.py        # seeding helpers
│   ├── data/           # Stage 1: image index, splits, loading, augmentation
│   ├── models/         # Stage 1 & 2: custom CNN, DenseNet-121
│   ├── explain/        # Stage 3: Grad-CAM, lung-attention metric
│   └── masking/        # Stage 4: lung masks, masked training
├── notebooks/          # one notebook per stage: 01_, 02_, 03_, 04_
├── results/            # metrics tables, plots, heatmaps
└── report/             # figures and notes for the Overleaf report
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

## Results

*Filled in as each stage is completed. The final comparison table is added in Stage 4.*

| Model | Training data | Accuracy | Recall | F1 | AUC | Lung attention % |
|---|---|---|---|---|---|---|
| Custom CNN | Unmasked | – | – | – | – | – |
| DenseNet-121 | Unmasked | – | – | – | – | – |
| DenseNet-121 | Lung-masked | – | – | – | – | – |

## Working rules

- `main` is kept working at all times. Each member works on their own branch and merges through a pull request.
- PR reviews: Ajit reviews Stage 1, Aman reviews Stage 2, Himanshu reviews Stage 3, Abhishek reviews Stage 4.
- Never commit the dataset or large model weights. They're shared through Google Drive (link will be added here).
- One fixed random seed (42), kept in a shared config file and used everywhere.
- Every stage has one parent issue split into child issues, labelled `stage-N` + `parent` / `child`.
- Before submission, the code must run from a clean clone using only this README.
- Commit messages follow a short prefix style (`feat:`, `fix:`, `docs:`, `chore:`), e.g. `docs: update setup notes`.

## References

1. P. Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, 2017. [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)
2. R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
