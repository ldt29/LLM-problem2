from datasets import load_dataset
import argparse
import random
import re
import os
from pathlib import Path
# Import fasttext after package installation
import fasttext


def normalize(text: str) -> str:
    """Very light text cleanup suitable for FastText.
    Removes line breaks and excessive whitespace.
    """
    # Collapse new lines and tabs
    text = re.sub(r"\s+", " ", text)  # replace any whitespace by single space
    text = text.strip()
    return text


def sample_indices(dataset_len: int, n_samples: int, seed: int = 42):
    random.seed(seed)
    if n_samples >= dataset_len:
        return list(range(dataset_len))
    return random.sample(range(dataset_len), n_samples)


def write_fasttext_file(ds, indices, label: str, fh):
    """Write selected indices to open file-handle in FastText format."""
    lbl_prefix = f"__label__{label} "
    for idx in indices:
        txt = ds[idx].get("text") or ds[idx].get("content") or ""
        txt = normalize(txt)
        if not txt:
            continue
        fh.write(lbl_prefix + txt + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastText math-domain classifier trainer")
    parser.add_argument("--n_pos", type=int, default=100000, help="#positive samples from open-web-math")
    parser.add_argument("--n_neg", type=int, default=100000, help="#negative samples from fineweb")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="validation split ratio from the merged dataset")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="test split ratio from the merged dataset")
    parser.add_argument("--out_dir", type=str, default="data_fasttext", help="output directory for txt files and model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos_slice", type=str, default=None, help="subset slice for positive dataset, e.g. 'train[:5%]'")
    parser.add_argument("--neg_slice", type=str, default=None, help="subset slice for negative dataset")
    parser.add_argument("--streaming", action="store_true", help="enable streaming mode to iterate dataset without full download")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load datasets
    print("Loading datasets…")

    load_kwargs = {"streaming": args.streaming} if args.streaming else {}

    if args.streaming:
        pos_split = "train"
        neg_split = "train"
    else:
        pos_split = args.pos_slice if args.pos_slice else "train"
        neg_split = args.neg_slice if args.neg_slice else "train"

    ds_pos = load_dataset("open-web-math/open-web-math", split=pos_split, **load_kwargs)
    # choose builder config for fineweb if provided; else default.
    ds_neg = load_dataset("HuggingFaceFW/fineweb", split=neg_split, **load_kwargs)

    def take_n(dataset_iterable, n):
        """Return list of (index,text) tuples from iterable dataset without loading all."""
        examples = []
        for idx, item in enumerate(dataset_iterable):
            if len(examples) >= n:
                break
            examples.append((idx, item))
        return examples

    if args.streaming:
        if args.pos_slice or args.neg_slice:
            print("[INFO] slice syntax is ignored in --streaming mode; full stream will be read until n samples are collected.")
        pos_samples = take_n(ds_pos, args.n_pos)
        neg_samples = take_n(ds_neg, args.n_neg)
        lines_path = out_dir / "all.txt"
        with lines_path.open("w", encoding="utf-8") as fh:
            for _, item in pos_samples:
                txt = item.get("text") or item.get("content") or ""
                txt = normalize(txt)
                if not txt:
                    continue
                fh.write("__label__math " + txt + "\n")
            for _, item in neg_samples:
                txt = item.get("text") or item.get("content") or ""
                txt = normalize(txt)
                if not txt:
                    continue
                fh.write("__label__other " + txt + "\n")
    else:
        print(f"Positive dataset size: {len(ds_pos):,} | Negative dataset size: {len(ds_neg):,}")

        # 2. Sample indices within loaded datasets
        idx_pos = sample_indices(len(ds_pos), args.n_pos, seed=args.seed)
        idx_neg = sample_indices(len(ds_neg), args.n_neg, seed=args.seed)

        # 3. Prepare combined data
        lines_path = out_dir / "all.txt"
        with lines_path.open("w", encoding="utf-8") as fh:
            write_fasttext_file(ds_pos, idx_pos, "math", fh)
            write_fasttext_file(ds_neg, idx_neg, "other", fh)

    # 4. Shuffle and split
    print("Shuffling & splitting…")
    with lines_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    random.shuffle(lines)

    n_total = len(lines)
    n_test = int(n_total * args.test_ratio)
    n_valid = int(n_total * args.valid_ratio)
    n_train = n_total - n_test - n_valid

    train_lines = lines[:n_train]
    valid_lines = lines[n_train : n_train + n_valid]
    test_lines = lines[n_train + n_valid :]

    (out_dir / "train.txt").write_text("".join(train_lines), encoding="utf-8")
    (out_dir / "valid.txt").write_text("".join(valid_lines), encoding="utf-8")
    (out_dir / "test.txt").write_text("".join(test_lines), encoding="utf-8")

    print(f"Split sizes: train {n_train:,}, valid {n_valid:,}, test {n_test:,}")

    # 5. Train FastText model
    model_prefix = out_dir / "math_cls"
    model = fasttext.train_supervised(
        input=str(out_dir / "train.txt"),
        lr=0.5,
        epoch=10,
        wordNgrams=2,
        dim=200,
        thread=os.cpu_count() or 2,
    )

    # save model and vectors
    model.save_model(str(model_prefix) + ".bin")

    # 6. Evaluate on valid & test sets
    def fasttext_test(txt_file):
        return model.test(str(txt_file))

    print("Validation set evaluation:")
    valid_result = fasttext_test(out_dir / "valid.txt")
    print(f"P@1: {valid_result[1]:.3f}, R@1: {valid_result[2]:.3f}, Number of examples: {valid_result[0]}")

    print("Test set evaluation:")
    test_result = fasttext_test(out_dir / "test.txt")
    print(f"P@1: {test_result[1]:.3f}, R@1: {test_result[2]:.3f}, Number of examples: {test_result[0]}")

    print(f"Model saved to {model_prefix}.bin")