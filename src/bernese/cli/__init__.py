# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Bernese CLI - Command-line interface for regulatory genomics predictions."""

import typer

from bernese.cli import data
from bernese.cli import inspect
from bernese.cli import summary
from bernese.cli import train

app = typer.Typer(
    help="Bernese - Sequential Neural Network for regulatory genomics predictions.",
    add_completion=False,
)

# Register subcommands - both prepare and train are flattened to top level
app.add_typer(data.prepare_app, name="prepare")
app.command(name="inspect")(inspect.inspect)
app.command(name="summary")(summary.summary)
app.command(name="train")(train.train)


@app.callback()
def main() -> None:
    """Bernese CLI.

    A PyTorch-based library for regulatory genomics predictions using
    Sequential Neural Networks (SeqNN).

    Commands:
        inspect  Inspect a prepared dataset
        prepare  Prepare genomic data for training
        train    Train a SeqNN model

    Example:
        bernese inspect data_dir/
        bernese prepare genome.fa targets.tsv -o data_out
        bernese train params.json data_dir/
    """
    pass


if __name__ == "__main__":
    app()
