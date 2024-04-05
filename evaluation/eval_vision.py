import argparse
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from utils import seed_everthing, set_logger
from utils import compute_activation_and_logits
from utils import compute_fid_score
from utils import compute_inception_score_from_logits
from utils import compute_activation_stat


def main(args):
    # Logging configuration
    target_dir = Path(args.output_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    time = datetime.now().strftime("%m%d-%H:%M:%S")
    set_logger(f=target_dir / f"vision_{time}.log")

    # Print configuaration
    for key, val in vars(args).items():
        logging.info(f"{key}: {val}")
    seed_everthing(args.seed)

    # Load dataset stat
    assert args.dataset in ["mnist", "fmnist", "celeba", "cifar10", "celeba64"]
    real_stat = np.load(f"_assets/{args.dataset}/stats.npz")
    real_mu = real_stat["mu"]
    real_sigma = real_stat["sigma"]

    # Load synthetic data
    synthesis = np.load(args.synthesis_path)
    images = synthesis["data"]

    assert type(images) == np.ndarray
    assert len(images.shape) == 4

    if images.shape[1] == 3 or images.shape[1] == 1:
        images = np.transpose(images, [0, 2, 3, 1])
    if images.shape[3] == 1:
        images = images.repeat(3, 3)
    if images.dtype == np.uint8:
        images = images.astype(np.float32)
        images /= 255.

    # Compute synthetic data stat
    logging.info(f"Compute activations...")
    acts, logits = compute_activation_and_logits(images)
    gen_mu, gen_sigma = compute_activation_stat(acts)

    # Compute FID and IS
    fid = compute_fid_score(real_mu, real_sigma, gen_mu, gen_sigma)
    is_mean, _ = compute_inception_score_from_logits(logits)

    logging.info(f"IS: {is_mean}")
    logging.info(f"FID: {fid}")


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
        "--batch_size",
        type=int,
        default=128,
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
