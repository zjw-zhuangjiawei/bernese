# CLI subpackage
import sys


def main():
    """Bernese CLI entry point."""
    from bernese.cli import train as train_module
    from bernese.cli import data as data_module

    if len(sys.argv) < 2:
        print("Bernese CLI")
        print("Usage: bernese <command> [options]")
        print("")
        print("Commands:")
        print("  data    Prepare genomic data for training")
        print("  train   Train a SeqNN model")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        sys.argv = sys.argv[1:]  # Remove 'train' from args
        train_module.main()
    elif command == "data":
        sys.argv = sys.argv[1:]  # Remove 'data' from args
        data_module.main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: data, train")
        sys.exit(1)


__all__ = ["main"]
