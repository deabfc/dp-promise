import logging
from tqdm import tqdm
import numpy as np
import random
import sys

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, Dataset

from torchvision import transforms
from torchvision.utils import make_grid

from src.models import UNetModel


def set_logger(path, level="INFO"):
    logger = logging.getLogger()

    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')

    stdout_handler = logging.StreamHandler(open(path / "log.txt", "w"))
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(open(path / "err.txt", "w"))
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    sys.stderr = stderr_handler.stream

    logger.setLevel(level)


def seed_everthing(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_unet_model(config, checkpoint_path=None):
    model = UNetModel(
        in_channels=config.data.img_ch,
        model_channels=config.model.ch,
        out_channels=config.data.img_ch,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attn,
        dropout=config.model.dropout,
        channel_mult=config.model.ch_mult,
        num_classes=config.data.num_classes,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        for name, param in model.named_parameters():
            if "time_emb" in name:
                param.requires_grad = False
    return model


def get_num_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    return total


def load_dataset(name, path=None, transform=None, train=True):
    if name == "mnist":
        from torchvision.datasets import MNIST
        dataset = MNIST(
            root="_data",
            train=train,
            download=True,
            transform=transform,
        )
    elif name == "fmnist":
        from torchvision.datasets import FashionMNIST
        dataset = FashionMNIST(
            root="_data",
            train=train,
            download=True,
            transform=transform,
        )
    elif name == "cifar10":
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root="_data", train=train,
                          download=True, transform=transform)
    elif name == "celeba":
        from src.data import CelebA
        if path is None:
            path = "_data/CelebA/processed"
        dataset = CelebA(path=path, transform=transform)
    elif name == "celeba64":
        if path is None:
            path = "_data/CelebA/processed64"
        from src.data import CelebA
        dataset = CelebA(path=path, transform=transform)
    elif name == "imagenet":
        if path is None:
            path = "_data/ImageNet/processed"
        from src.data import ImageNet
        dataset = ImageNet(path=path, transform=transform)
    elif name == "imagenet64":
        if path is None:
            path = "_data/ImageNet/processed64"
        from src.data import ImageNet
        dataset = ImageNet(path=path, transform=transform)
    else:
        raise NotImplementedError()
    return dataset


def load_dataset_from_config(config, transform=None, train=True):
    if config.data.name in ["cifar10"]:
        if config.data.img_ch != 3:
            transform = transforms.Compose([
                transforms.Grayscale(config.data.img_ch),
                transform,
            ])
        if config.data.img_size != 32:
            transform = transforms.Compose([
                transforms.Resize(
                    (config.data.img_size, config.data.img_size)),
                transform,
            ])
    if config.data.name in ["mnist", "cifar10", "fmnist"]:
        path = None
    else:
        path = config.data.path
    dataset = load_dataset(config.data.name, path=path, transform=transform)
    return dataset


def get_sampler(config):
    if config.sample.type == "ddim":
        from src.samplers import DDIMSampler
        sampler = DDIMSampler(config)
    else:
        raise NotImplementedError
    return sampler


@torch.no_grad()
def sample_images(model, config, num_samples):
    device = next(model.parameters()).device
    sampler = get_sampler(config)

    if config.data.class_condition:
        labels = torch.randint(config.data.num_classes, size=(num_samples, ))
    else:
        labels = torch.zeros(size=(num_samples, ), dtype=torch.long)

    dataloader = DataLoader(
        TensorDataset(labels),
        batch_size=config.sample.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    images = []
    for (y, ) in tqdm(dataloader):
        shape = (
            y.size(0),
            config.data.img_ch,
            config.data.img_size,
            config.data.img_size,
        )
        y = y.to(device) + 1
        if not config.data.class_condition:
            y = torch.zeros_like(y, dtype=torch.long).to(device)
        X = sampler.sample(model, shape, y)
        X = (X + 1.) / 2.
        X = X.clip(min=0., max=1.)
        X = X.detach().cpu()
        images.append(X)
    images = torch.cat(images)
    return images, labels


@torch.no_grad()
def sample_example_image(model, config):
    model.eval()
    device = next(model.parameters()).device

    sampler = get_sampler(config)

    if config.data.class_condition:
        labels = list(range(config.data.num_classes)) * 4
    else:
        labels = [0] * 40

    class LabelDataset(Dataset):
        def __init__(self, lables) -> None:
            super().__init__()
            self.labels = lables

        def __getitem__(self, index):
            return self.labels[index]

        def __len__(self):
            return len(self.labels)

    dataloader = DataLoader(
        LabelDataset(labels),
        batch_size=config.sample.batch_size,
        shuffle=False,
    )
    images = []
    for y in dataloader:
        y = y.to(device) + 1
        if not config.data.class_condition:
            y = torch.zeros_like(y).to(device)
        shape = (
            y.size(0),
            config.data.img_ch,
            config.data.img_size,
            config.data.img_size,
        )
        X = sampler.sample(model, shape, y)
        X = (X + 1.) / 2.
        X = X.clip(min=0., max=1.)
        images.append(X)
    images = torch.cat(images)
    image = make_grid(images, nrow=config.data.num_classes)
    return image
