import argparse
import math

from datasets import load_dataset
from transformers import AutoTokenizer


SQUAD_DATASET_REPO = "rajpurkar/squad"
DEFAULT_TOKENIZER = "bert-base-uncased"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile SQuADv1.1 BERT token lengths for TPDCIM sparse-block experiments."
    )
    parser.add_argument("--dataset-repo", default=SQUAD_DATASET_REPO)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-lengths", nargs="+", type=int, default=[384, 512])
    parser.add_argument("--tile-ns", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def percentile(sorted_values, pct):
    if not sorted_values:
        return 0
    rank = math.ceil((pct / 100.0) * len(sorted_values)) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def fmt_cell(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def tokenize_lengths(dataset, tokenizer, max_length, batch_size):
    lengths = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        encoded = tokenizer(
            batch["question"],
            batch["context"],
            truncation="only_second",
            padding="max_length",
            max_length=max_length,
        )
        lengths.extend(int(sum(mask)) for mask in encoded["attention_mask"])
    return lengths


def print_table(rows):
    headers = [
        "dataset_repo",
        "split",
        "tokenizer",
        "max_length",
        "tile_n",
        "num_blocks",
        "num_examples",
        "min_nonpad_tokens",
        "mean_nonpad_tokens",
        "median_nonpad_tokens",
        "p90_nonpad_tokens",
        "p95_nonpad_tokens",
        "max_nonpad_tokens",
        "mean_real_blocks",
        "p90_real_blocks",
        "p95_real_blocks",
        "max_real_blocks",
    ]
    print(" | ".join(headers))
    print(" | ".join("-" * len(h) for h in headers))
    for row in rows:
        print(" | ".join(fmt_cell(row[header]) for header in headers))


def main():
    args = parse_args()
    print("track: squad_length_profile")
    print("Note: this profiles token lengths only; it does not evaluate QA accuracy.")
    print("dataset_repo:", args.dataset_repo)
    print("split:", args.split)
    print("tokenizer:", args.tokenizer)

    dataset = load_dataset(args.dataset_repo, split=args.split)
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = []

    for max_length in args.max_lengths:
        lengths = tokenize_lengths(dataset, tokenizer, max_length, args.batch_size)
        sorted_lengths = sorted(lengths)
        base_stats = {
            "dataset_repo": args.dataset_repo,
            "split": args.split,
            "tokenizer": args.tokenizer,
            "max_length": max_length,
            "num_examples": len(lengths),
            "min_nonpad_tokens": min(lengths) if lengths else 0,
            "mean_nonpad_tokens": mean(lengths),
            "median_nonpad_tokens": percentile(sorted_lengths, 50),
            "p90_nonpad_tokens": percentile(sorted_lengths, 90),
            "p95_nonpad_tokens": percentile(sorted_lengths, 95),
            "max_nonpad_tokens": max(lengths) if lengths else 0,
        }

        for tile_n in args.tile_ns:
            num_blocks = ceil_div(max_length, tile_n)
            real_blocks = [min(ceil_div(length, tile_n), num_blocks) for length in lengths]
            sorted_blocks = sorted(real_blocks)
            rows.append(
                {
                    **base_stats,
                    "tile_n": tile_n,
                    "num_blocks": num_blocks,
                    "mean_real_blocks": mean(real_blocks),
                    "p90_real_blocks": percentile(sorted_blocks, 90),
                    "p95_real_blocks": percentile(sorted_blocks, 95),
                    "max_real_blocks": max(real_blocks) if real_blocks else 0,
                }
            )

    print_table(rows)


if __name__ == "__main__":
    main()
