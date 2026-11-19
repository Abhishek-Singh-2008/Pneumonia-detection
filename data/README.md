# data/

The dataset is **not** stored in Git (see the main README, "Dataset" and "Setup").

## Where the images come from
- **Kaggle (recommended):** attach the dataset `paultimothymooney/chest-xray-pneumonia`; nothing goes in this folder.
  It is mounted at `/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray/`, and `src.config.get_data_root()` finds it automatically.
- **Colab or local:** extract the dataset so this folder contains `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`,
  or set `PNEUMONIA_DATA_ROOT` to any folder that contains `train/`, `val/` and `test/`.
- Only the top-level `chest_xray/{train,val,test}` is used; the duplicate `chest_xray/chest_xray/` and `__MACOSX/` are ignored.

## `splits/` (committed, used by every stage)
| File | Contents |
|---|---|
| `train.csv` | 4,578 images (1,180 NORMAL / 3,398 PNEUMONIA) |
| `val.csv` | 654 images (169 / 485) |
| `test.csv` | 624 images (234 / 390), the official test folder unchanged |
| `split_summary.json` | counts per split, patients per split, seed, method and `pos_weight` = 0.347263 |

- Columns: `image_id, filepath, label, label_name, patient_id, official_folder, split`. `filepath` is relative to the data root.
- Patient-level and stratified: no patient is in two splits. The val split is 12.5% of each class from official `train/` + `val/`.
- **Do not make new splits.** Load them with `src.data.splits.load_split("train")`.
- `python -m src.data.splits` regenerates the same split on any OS and library version (method: issue #21).
