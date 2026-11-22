"""Kaggle kernel: Stage 4 lung-masked DenseNet-121 training (issues #16 and #17).

Runs the whole Stage 4.1 chain on a Kaggle GPU and leaves everything needed for
Stage 4.2 in /kaggle/working:

1. Clone the repository at STAGE4_REF (default: stage4-masking-report).
   Optionally overlay a newer copy of the Stage 4 modules from an attached
   Kaggle dataset ("stage4-code"), so the kernel can run code that is not pushed.
2. Get the Stage 3 lung masks: reuse an attached Kaggle dataset if one is present,
   otherwise download the team Drive lung_masks.zip, otherwise generate them with
   src.masking.lung_masks (needs Internet for the PSPNet weights).
3. Build the masked image copies with src.masking.masked_dataset.
4. Train Phase A then Phase B with src.train_masked_densenet, exactly the Stage 2
   settings, validating only. The Stage 2 settings check and the pixel/mask
   alignment check run first, so a drifted setting or a bad mask stops the kernel.
   The test split is never touched here.
5. Zip the checkpoints and Stage 4 results into stage4_bundle.zip for download.

Environment switches (set them in the kernel metadata or at the top of this file):
    STAGE4_REF=stage4-masking-report   git branch, tag or commit to run
    STAGE4_SMOKE=1                     1 epoch per phase, to check the pipeline only
    STAGE4_REPO=/path/to/checkout       reuse this checkout instead of cloning

The kernel stops at the first failed step, so a broken run never looks like a
successful one.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REPO_URL = "https://github.com/Abhishek-Singh-2008/Pneumonia-detection.git"
GIT_REF = os.environ.get("STAGE4_REF", "stage4-masking-report")
SMOKE = os.environ.get("STAGE4_SMOKE", "0") == "1"

WORK = Path("/kaggle/working")
# STAGE4_REPO lets a notebook reuse a checkout it cloned itself instead of letting
# this script clone (and replace) the repository.
REPO = Path(os.environ.get("STAGE4_REPO", str(WORK / "Pneumonia-detection")))
MASK_DIR = WORK / "lung_masks"
MASKED_ROOT = WORK / "masked_chest_xray"
CKPT_DIR = Path("/kaggle/working/checkpoints")

# Stage 3 lung masks shared on the team Drive (5,856 PNGs + areas.csv, verified
# against results/stage3/lung_mask_summary.csv). Downloading them is faster and
# more reproducible than re-running the segmentation model.
DRIVE_MASK_ZIP_ID = "1ijzEW5sEwMmEbyUQfpTLJ7Oehl5IKqpr"
DRIVE_MASK_ZIP_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_MASK_ZIP_ID}"
MASK_ZIP_MIN_BYTES = 1_000_000

# Optional attached inputs. First existing path wins; none is required.
CODE_OVERLAY = Path("/kaggle/input/stage4-code")
MASK_CANDIDATES = (
    Path("/kaggle/input/lung-masks/lung_masks"),
    Path("/kaggle/input/lung-masks"),
    Path("/kaggle/input/stage3-lung-masks/lung_masks"),
)
EXPECTED_IMAGES = 5856


def log(message: str) -> None:
    """Print a timestamped step marker so the Kaggle log is easy to follow."""
    print(f"\n[{time.strftime('%H:%M:%S')}] === {message} ===", flush=True)


def run(cmd, cwd=None) -> None:
    """Run a command, echoing it, and raise on a non-zero exit code."""
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def print_commit() -> None:
    """Print the checked-out commit and any local changes, for the run log."""
    run(["git", "log", "--oneline", "-1"], cwd=REPO)
    run(["git", "status", "--short"], cwd=REPO)


def clone_repo() -> None:
    """Clone the project at GIT_REF, or reuse the checkout named by STAGE4_REPO."""
    if os.environ.get("STAGE4_REPO") and (REPO / ".git").is_dir():
        log(f"use existing checkout {REPO} (STAGE4_REPO is set)")
        print_commit()
        return
    if REPO.exists():
        shutil.rmtree(REPO)
    log(f"clone {REPO_URL} @ {GIT_REF}")
    try:
        run(["git", "clone", "--depth", "1", "--branch", GIT_REF, REPO_URL, REPO])
    except subprocess.CalledProcessError:
        # A commit hash is not a branch; fall back to a full clone plus checkout.
        print("branch clone failed, trying a full clone for a commit hash", flush=True)
        run(["git", "clone", REPO_URL, REPO])
        run(["git", "checkout", GIT_REF], cwd=REPO)
    print_commit()


def overlay_code() -> None:
    """Copy a newer src/ tree from an attached dataset over the clone, if present."""
    if not CODE_OVERLAY.is_dir():
        print("no code overlay dataset; running the cloned branch as-is", flush=True)
        return
    log(f"overlay code from {CODE_OVERLAY}")
    copied = 0
    for source in sorted(CODE_OVERLAY.rglob("*.py")):
        relative = source.relative_to(CODE_OVERLAY)
        target = REPO / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    print(f"copied {copied} .py files", flush=True)


def ensure_torchxrayvision() -> None:
    """Install torchxrayvision only if the kernel image does not already have it."""
    try:
        import torchxrayvision  # noqa: F401
        print("torchxrayvision already installed", flush=True)
        return
    except ImportError:
        pass
    log("install torchxrayvision")
    run([sys.executable, "-m", "pip", "install", "-q", "torchxrayvision==1.5.7"])


def prepare_masks() -> None:
    """Get the 5,856 Stage 3 masks: attached dataset, then Drive zip, then generate."""
    for candidate in MASK_CANDIDATES:
        if candidate.is_dir() and len(list(candidate.glob("*.png"))) >= EXPECTED_IMAGES:
            log(f"reuse masks from attached dataset {candidate}")
            if candidate.resolve() != MASK_DIR.resolve():
                if MASK_DIR.exists():
                    shutil.rmtree(MASK_DIR)
                MASK_DIR.mkdir(parents=True, exist_ok=True)
                for file in candidate.iterdir():
                    if file.is_file():
                        shutil.copy2(file, MASK_DIR / file.name)
            check_mask_count()
            return

    if MASK_DIR.exists() and len(list(MASK_DIR.glob("*.png"))) >= EXPECTED_IMAGES:
        log(f"reuse masks already in {MASK_DIR}")
        check_mask_count()
        return

    if download_masks_from_drive():
        return

    log("generate lung masks (Stage 3.2)")
    ensure_torchxrayvision()
    run([sys.executable, "-m", "src.masking.lung_masks", "--out-dir", MASK_DIR], cwd=REPO)
    check_mask_count()


def check_mask_count() -> None:
    """Stop if the mask folder is not the complete Stage 3 set."""
    found = len(list(MASK_DIR.glob("*.png")))
    if found < EXPECTED_IMAGES:
        raise RuntimeError(f"only {found} masks in {MASK_DIR}, expected {EXPECTED_IMAGES}")
    print(f"masks: {found} PNG files in {MASK_DIR}", flush=True)


def flatten_masks() -> None:
    """Move masks up if the zip wrapped them in a single sub-folder."""
    if list(MASK_DIR.glob("*.png")):
        return
    subdirs = [path for path in MASK_DIR.iterdir() if path.is_dir()]
    if len(subdirs) != 1:
        return
    for file in subdirs[0].iterdir():
        if file.is_file():
            shutil.move(str(file), str(MASK_DIR / file.name))


def download_masks_from_drive() -> bool:
    """Download the team Drive mask zip and unpack it; return False if unavailable."""
    log(f"download lung masks from Drive ({DRIVE_MASK_ZIP_ID})")
    WORK.mkdir(parents=True, exist_ok=True)
    zip_path = WORK / "lung_masks.zip"
    request = urllib.request.Request(DRIVE_MASK_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, open(zip_path, "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as error:  # noqa: BLE001 - any network error falls back to generating masks
        print(f"download failed ({error}); falling back to mask generation", flush=True)
        return False

    size = zip_path.stat().st_size
    print(f"downloaded {size / 1e6:.1f} MB", flush=True)
    with open(zip_path, "rb") as handle:
        magic = handle.read(2)
    if size < MASK_ZIP_MIN_BYTES or magic != b"PK":
        print("downloaded file is not a zip (Drive may have returned an HTML page); falling back", flush=True)
        zip_path.unlink(missing_ok=True)
        return False

    if MASK_DIR.exists():
        shutil.rmtree(MASK_DIR)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(MASK_DIR)
    print(f"zip entries: {len(names)} (PNG {sum(n.lower().endswith('.png') for n in names)}, "
          f"other {[n for n in names if not n.lower().endswith('.png')][:5]})", flush=True)
    flatten_masks()
    check_mask_count()

    areas = MASK_DIR / "areas.csv"
    if areas.is_file():
        table = pd.read_csv(areas)
        print(f"areas.csv: {len(table)} rows, columns {list(table.columns)}", flush=True)
        if len(table) != EXPECTED_IMAGES:
            raise RuntimeError(f"areas.csv has {len(table)} rows, expected {EXPECTED_IMAGES}")
    else:
        print("WARNING: areas.csv is missing, so flagged-mask counts will not be recorded", flush=True)
    return True


def build_masked_copies() -> None:
    """Create the masked image copies from the committed splits, or reuse a complete set."""
    log("build masked image copies (Stage 4.1)")
    expected = {"train": 4578, "val": 654, "test": 624}
    summary_path = MASKED_ROOT / "masked_dataset_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        same_masks = summary.get("mask_dir") == str(MASK_DIR.resolve())
        if summary.get("split_counts") == expected and same_masks:
            print("reusing the masked copies already in", MASKED_ROOT, flush=True)
            print(json.dumps(summary, indent=2), flush=True)
            return
    if MASKED_ROOT.exists():
        shutil.rmtree(MASKED_ROOT)
    run([
        sys.executable, "-m", "src.masking.masked_dataset",
        "--mask-dir", MASK_DIR,
        "--output-root", MASKED_ROOT,
    ], cwd=REPO)
    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary, indent=2), flush=True)
    if summary.get("split_counts") != expected:
        raise RuntimeError(f"masked split counts {summary.get('split_counts')} != {expected}")


def train_masked_model() -> None:
    """Run Phase A and Phase B; one epoch each when STAGE4_SMOKE is set."""
    log("train masked DenseNet-121 (Stage 4.1)")
    cmd = [sys.executable, "-m", "src.train_masked_densenet",
           "--masked-data-root", MASKED_ROOT, "--mask-dir", MASK_DIR]
    if SMOKE:
        print("SMOKE MODE: one epoch per phase", flush=True)
        cmd += ["--phase-a-epochs", "1", "--phase-b-epochs", "1", "--no-cache"]
    run(cmd, cwd=REPO)


def bundle_outputs() -> Path:
    """Zip checkpoints and Stage 4 results so they can be downloaded in one file."""
    log("bundle outputs")
    bundle = WORK / "stage4_bundle.zip"
    wanted = [CKPT_DIR / "densenet121_masked_phaseA_best.pth",
              CKPT_DIR / "densenet121_masked_phaseB_best.pth"]
    wanted += sorted((REPO / "results").glob("stage4*"))
    entries, seen = [], set()
    for path in wanted:
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(file for file in path.rglob("*") if file.is_file())
        else:
            continue
        for file in files:
            if file in seen:
                continue
            seen.add(file)
            # Checkpoints keep their file name; results keep their repo-relative path.
            arcname = file.name if file.parent == CKPT_DIR else str(file.relative_to(REPO))
            entries.append((file, arcname))
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file, arcname in entries:
            zf.write(file, arcname)
    print(f"bundle: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB, {len(entries)} files)", flush=True)
    for file in sorted(WORK.glob("*.zip")):
        print(f"  {file.name}: {file.stat().st_size / 1e6:.1f} MB", flush=True)
    return bundle


def main() -> None:
    """Run every step in order and print a short summary at the end."""
    log("environment")
    print(f"repo={REPO} ref={GIT_REF} smoke={SMOKE}", flush=True)
    run([sys.executable, "-c", "import torch; print('torch', torch.__version__, torch.cuda.is_available())"])

    clone_repo()
    overlay_code()
    prepare_masks()
    build_masked_copies()
    train_masked_model()
    bundle_outputs()

    log("done")
    for folder in (CKPT_DIR, REPO / "results"):
        if folder.exists():
            print(folder, flush=True)
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    print(f"  {file.relative_to(folder)}  {file.stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
