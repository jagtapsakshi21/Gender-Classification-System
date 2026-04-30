"""
dataset_setup.py — Download & prepare the Gender Face Dataset from Kaggle
or split an existing flat folder into train/val/test sets.

Usage (Kaggle):
    kaggle datasets download -d ashishjangra27/gender-recognitoon-200k-images-celeba
    python dataset_setup.py --source downloads/archive --split

Usage (custom folder):
    python dataset_setup.py --source my_images_folder --split
    # my_images_folder must contain:  female/  and  male/  subfolders
"""
import argparse
import shutil
import random
from pathlib import Path

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "dataset"
EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset(src: Path, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Takes a source directory with class subfolders and copies files into:
      dataset/female/  dataset/male/   (train)
      dataset/test/female/  dataset/test/male/  (test)
    """
    random.seed(seed)
    classes = [d for d in src.iterdir() if d.is_dir()]
    print(f"  Found classes: {[c.name for c in classes]}")

    for cls_dir in classes:
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in EXTS]
        random.shuffle(files)
        n_test = int(len(files) * test_ratio)
        n_val  = int(len(files) * val_ratio)
        test_files  = files[:n_test]
        train_files = files[n_test:]

        # Train/val go directly under dataset/{class}/
        train_dst = DATA_DIR / cls_dir.name.lower()
        train_dst.mkdir(parents=True, exist_ok=True)
        for f in train_files:
            shutil.copy2(f, train_dst / f.name)

        # Test set under dataset/test/{class}/
        test_dst = DATA_DIR / "test" / cls_dir.name.lower()
        test_dst.mkdir(parents=True, exist_ok=True)
        for f in test_files:
            shutil.copy2(f, test_dst / f.name)

        print(f"  {cls_dir.name}: {len(train_files)} train | {n_test} test")
    print(f"\n✅  Dataset ready in {DATA_DIR}")


def show_stats():
    print("\n── Dataset Statistics ────────────────────────────")
    for split in ["", "test"]:
        base = DATA_DIR / split if split else DATA_DIR
        if not base.exists(): continue
        label = "Test" if split else "Train+Val"
        print(f"\n  [{label}]")
        for cls_dir in sorted(base.iterdir()):
            if cls_dir.is_dir() and cls_dir.name != "test":
                count = sum(1 for f in cls_dir.rglob("*") if f.suffix.lower() in EXTS)
                print(f"    {cls_dir.name:<12}: {count:>5} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None, help="Source directory")
    parser.add_argument("--split",  action="store_true", help="Split into train/test")
    parser.add_argument("--stats",  action="store_true", help="Show dataset stats")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.source and args.split:
        split_dataset(Path(args.source))
        show_stats()
    else:
        print("Usage: python dataset_setup.py --source <dir> --split")
        print("       python dataset_setup.py --stats")
