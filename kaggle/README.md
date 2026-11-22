# kaggle/ — Stage 4 Kaggle runner

One runner that does the whole Stage 4.1 chain on a T4 and returns the checkpoints and
results in a single zip.

| File | Purpose |
|---|---|
| `run_masked_training.py` | The whole chain: clone → masks → masked copies → train → bundle |
| `stage4_kaggle.ipynb` | Ready-to-upload notebook for the browser flow (no API token needed) |
| `kernel-metadata.json` | Kernel settings for the optional API flow (**replace `KAGGLE_USERNAME`**) |

## What the runner does

1. Clones `Abhishek-Singh-2008/Pneumonia-detection` at `STAGE4_REF` (default `stage4-masking-report`).
2. Gets the 5,856 Stage 3 masks: an attached Kaggle dataset if present, otherwise the team
   Drive `lung_masks.zip` (7.2 MB), otherwise regenerates them with `src.masking.lung_masks`.
3. Builds `masked_chest_xray/` and checks the split counts are 4,578 / 654 / 624.
4. Trains Phase A then Phase B with the Stage 2 settings. Two guards run first: the
   Stage 2 settings check and the pixel/mask alignment check. The test split is never touched.
5. Writes `/kaggle/working/stage4_bundle.zip` with both checkpoints and `results/stage4*`.

Environment switches: `STAGE4_SMOKE=1` (one epoch per phase), `STAGE4_REF=<branch|tag|commit>`,
`STAGE4_REPO=<existing checkout>` (reuse a clone instead of cloning again).

## Browser flow (no API token)

1. Download `kaggle/stage4_kaggle.ipynb` from GitHub.
2. kaggle.com → **Create → Notebook** → **File → Import Notebook** → upload that file.
3. **Settings**: Accelerator **GPU T4 x2**, **Internet On**, and **Add Data → Datasets →
   `paultimothymooney/chest-xray-pneumonia`**. Internet + GPU need a phone-verified account.
4. **Run all**. Cell 5 is a one-epoch smoke test; cell 7 is the real training.
5. Download `stage4_bundle.zip` from the **Output** panel.

The imported notebook is a copy: if the repository changes, download the notebook again.
It clones the branch itself, so the code inside it is always the current branch tip
(printed in the log as `git log --oneline -1`).

## API flow (optional)

```bash
pip install kaggle                 # once
cd kaggle/
# put your username in kernel-metadata.json first
kaggle kernels push -p .
kaggle kernels status KAGGLE_USERNAME/stage4-masked-training
kaggle kernels output KAGGLE_USERNAME/stage4-masked-training -p ../kaggle_out
```

Needs `~/.kaggle/kaggle.json` (Kaggle → Account → Create New Token).

## Notes

- The kernel never evaluates the test split, so it cannot influence model selection.
- Checkpoints and masks are not committed; they go to the team Drive folder.
- The smoke run's numbers are a pipeline check, not a result: one epoch cannot learn anything.
- Masked copies are reused when they already match the mask folder, so the full run after a
  smoke test does not rebuild them.
