# data/

The dataset is **not** stored in Git. See the main README, "Dataset" section.

- On Kaggle, attach the dataset `paultimothymooney/chest-xray-pneumonia`; nothing goes in this folder.
- On Colab or locally, extract the dataset so this folder contains `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`.
- `data/splits/` holds the split CSVs (`train.csv`, `val.csv`, `test.csv`). They are committed so every stage uses identical images.
