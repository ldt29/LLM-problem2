import argparse
import json
from pathlib import Path

import fasttext
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select 5000 '__label__other' lines from test.txt, relabel with FastText, and export JSONL."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data_fasttext/math_cls.bin",
        help="Path to trained fastText .bin model.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data_fasttext/test.txt",
        help="Input fastText-formatted text file (one line per record).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_fasttext/relabelled_5000.jsonl",
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5000,
        help="Number of '__label__other' examples to process.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single input text to classify; if provided, batch relabeling is skipped.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    assert model_path.is_file(), f"Model not found: {model_path}"
    assert input_path.is_file(), f"Input file not found: {input_path}"

    print(f"Loading model from {model_path}…")
    ft_model = fasttext.load_model(str(model_path))

    # ----- Single sentence inference -----
    if args.text is not None:
        text_single = args.text.strip()
        labels, probs = ft_model.predict(text_single)
        label = labels[0].replace("__label__", "")
        prob = float(probs[0])
        print(f"[SINGLE] label={label}, prob={prob:.4f}\ntext={text_single}")
        return

    print("Scanning for '__label__other' lines…")
    selected_lines = []
    with input_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            if line.startswith("__label__other "):
                selected_lines.append(line[len("__label__other ") :].strip())
                if len(selected_lines) >= args.n:
                    break

    print(f"Collected {len(selected_lines)} lines. Running inference…")
    with output_path.open("w", encoding="utf-8") as f_out:
        for text in tqdm(selected_lines):
            labels, probs = ft_model.predict(text)
            label = labels[0].replace("__label__", "")
            prob = float(probs[0])
            json.dump({"text": text, "label": label, "prob": prob}, f_out, ensure_ascii=False)
            f_out.write("\n")

    print(f"Saved relabelled data to {output_path}")


if __name__ == "__main__":
    main() 