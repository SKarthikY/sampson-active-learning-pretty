# src/sampson_active_learning/run.py

import argparse
from pathlib import Path

from .config import load_config, AppConfig
from .active_learning import (
    run_propose_round,
    run_preprocess_round,
    run_training_round,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Active learning pipeline for the Sedona neural network emulator."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["propose", "preprocess", "train"],
        required=True,
        help=(
            "Which stage to run:\n"
            "  propose   - select worst samples, propose new points, submit Sedona jobs\n"
            "  preprocess - submit job to recompute normalization and preprocessed spectra\n"
            "  train     - submit emulator training job (after preprocessing is done)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: AppConfig = load_config(Path(args.config))

    if args.mode == "propose":
        run_propose_round(cfg)
    elif args.mode == "preprocess":
        run_preprocess_round(cfg)
    elif args.mode == "train":
        run_training_round(cfg)
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")


if __name__ == "__main__":
    main()