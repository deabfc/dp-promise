import argparse
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import ToTensor

from models import MLP, CNN
from utils import seed_everthing, set_logger
from utils import train_classifier, test_classifier


def main(args):
    # Logging configuration
    target_dir = Path(args.output_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    time = datetime.now().strftime("%m%d-%H:%M:%S")
    set_logger(f=target_dir / f"downstream_{time}.log")

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

    if images.dtype == np.uint8:
        images = images.astype(np.float32)
        images /= 255.
    if images.shape[3] == 3 or images.shape[3] == 1:
        images = np.transpose(images, [0, 3, 1, 2])

    assert images.shape[1] == 1 or images.shape[1] == 3

    gen_dataset = TensorDataset(
        torch.from_numpy(images), torch.from_numpy(labels)
    )

    # Load original data
    if args.dataset == "mnist":
        from torchvision.datasets import MNIST
        real_dataset = MNIST(root="../_data", train=False,
                             download=True, transform=ToTensor(),)
        img_ch = 1
        num_classes = 10
    elif args.dataset == "fmnist":
        from torchvision.datasets import FashionMNIST
        real_dataset = FashionMNIST(root="../_data", train=False,
                                    download=True, transform=ToTensor(),)
        img_ch = 1
        num_classes = 10
    else:
        raise NotImplementedError

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset not in ["mnist", "fmnist"]:
        raise NotImplementedError
    
    # Prepare train, val, test dataset
    train_dataset, val_dataset = train_test_split(
        gen_dataset, test_size=0.1)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    test_dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    models = {
        "MLP": MLP(num_classes=num_classes),
        "CNN": CNN(num_classes=num_classes),
    }

    # Training classifier
    for name, model in models.items():
        logging.info(f"Training {name} classifier...")
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        train_classifier(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            epochs=args.epochs,
        )
        acc = test_classifier(model, test_dataloader)
        logging.info(f"{name} classifier Accuracy: {acc:.4f}")


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
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    args = parser.parse_args()

    main(args)
