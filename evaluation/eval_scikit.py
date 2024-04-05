import argparse
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split

from utils import seed_everthing, set_logger
from utils import train_sklearn_models


def main(args):
    # Logging configuration
    target_dir = Path(args.output_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    time = datetime.now().strftime("%m%d-%H:%M:%S")
    set_logger(f=target_dir / f"scikit_{time}.log")

    # Print configuration
    for key, val in vars(args).items():
        logging.info(f"{key}: {val}")
    seed_everthing(args.seed)

    # Load Data
    synthesis = np.load(args.synthesis_path)

    images = synthesis["data"]
    labels = synthesis["labels"]

    # Data preparation
    assert type(images) == np.ndarray
    assert len(images.shape) == 4

    if images.dtype == np.float32:
        images = images * 255
        images = images.astype(np.uint8)
    if images.shape[3] == 3 or images.shape[3] == 1:
        images = np.transpose(images, [0, 3, 1, 2])

    images = images.astype(np.float32)
    images /= 255.

    # Train models
    logging.info(f"Training sklearn models...")
    mean_acc = train_sklearn_models(images, labels, args.dataset)
    logging.info(f"Mean Acc: {mean_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthesis_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="_results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    main(args)
