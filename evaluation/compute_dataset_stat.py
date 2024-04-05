import numpy as np
from pathlib import Path
from PIL import Image

import argparse

from utils import compute_activation_and_logits
from utils import compute_inception_score_from_logits
from utils import compute_activation_stat
from keras.datasets import mnist, fashion_mnist, cifar10


def main(args):
    if args.dataset == "mnist":
        (images, _), (_, _) = mnist.load_data()
        images = images.reshape(60000, 28, 28, 1)
        images = images.repeat(3, -1)
    elif args.dataset == "fmnist":
        (images, _), (_, _) = fashion_mnist.load_data()
        images = images.reshape(60000, 28, 28, 1)
        images = images.repeat(3, -1)
    elif args.dataset == "cifar10":
        (images, _), (_, _) = cifar10.load_data()
    elif args.dataset == "celeba":
        path = "../_data/CelebA/processed"
        images = []
        from tqdm import tqdm
        for f in tqdm(Path(path).glob("*.png")):
            image = Image.open(f)
            images.append(np.array(image))
        images = np.stack(images, axis=0)
    elif args.dataset == "celeba64":
        path = "../_data/CelebA/processed64"
        images = []
        from tqdm import tqdm
        for f in tqdm(Path(path).glob("*.png")):
            image = Image.open(f)
            images.append(np.array(image))
        images = np.stack(images, axis=0)
    else:
        raise NotImplementedError()

    assert (type(images) == np.ndarray)
    assert (images.dtype == np.uint8)
    assert (len(images.shape) == 4)
    assert (images.shape[3] == 3)

    images = images.astype('float32')
    images /= 255.

    acts, logits = compute_activation_and_logits(images)
    is_mean = compute_inception_score_from_logits(logits)
    print(f"Inception Score: {is_mean}")

    mu, sigma = compute_activation_stat(acts)

    savedir = Path(args.savedir) / f"{args.dataset}"
    savedir.mkdir(parents=True, exist_ok=True)

    np.savez(
        savedir / f"stats.npz",
        mu=mu,
        sigma=sigma,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--path",
        type=str,
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="_assets/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    main(args)
