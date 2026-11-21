"""CLI for training and generating product descriptions."""
import argparse
import json
from pathlib import Path

from app.core import milestone_1, milestone_2, milestone_4, milestone_5, milestone_6


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-product-describer", description="Train a tiny GPT locally and generate product descriptions.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the project is installed")
    sub.add_parser("prepare", help="create or inspect the product-description corpus")
    train = sub.add_parser("train", help="train GPT from scratch")
    train.add_argument("--epochs", type=int, default=400)
    train.add_argument("--merges", type=int, default=100)
    gen = sub.add_parser("generate", help="generate text from a free-form prompt")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--tokens", type=int, default=180)
    gen.add_argument("--temperature", type=float, default=0.8)
    describe = sub.add_parser("describe", help="describe a product from its details")
    describe.add_argument("--name", required=True)
    describe.add_argument("--features", required=True)
    describe.add_argument("--tone", default="friendly")
    describe.add_argument("--length", choices=["short", "medium", "long"], default="medium")
    describe.add_argument("--output", help="optional path to save the description")
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("AI Product Describer is ready.")
    elif args.command == "prepare":
        texts = milestone_1()
        print(f"Prepared {len(texts)} product descriptions in data/product_descriptions.txt")
        print("Category: reusable fabric bags and small organisers")
    elif args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs must be at least 1")
        report = milestone_4(epochs=args.epochs, num_merges=args.merges)
        print(json.dumps(report, indent=2))
        print("Saved model: models/product_gpt.pt")
    elif args.command == "generate":
        print(milestone_5(args.prompt, args.tokens, args.temperature))
    elif args.command == "describe":
        result = milestone_6(args.name, args.features, args.tone, args.length)
        print(result)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result + "\n", encoding="utf-8")
            print(f"Saved to {path}")


if __name__ == "__main__":
    main()
