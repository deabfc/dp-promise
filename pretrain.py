import argparse
from pathlib import Path
from datetime import datetime
import logging

from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

from src.utils import get_unet_model
from src.utils import load_dataset_from_config
from src.utils import get_num_parameters
from src.utils import set_logger
from src.utils import seed_everthing
from src.runners import train_ddpm


def main(config, opt):
    time = datetime.now().strftime("%m%d-%H:%M:%S")
    running_id = f"vanilla_{opt.job_id}_{config.data.name}_{time}"
    workdir = Path(opt.workdir) / running_id
    workdir.mkdir(parents=True)

    # Logging configuration
    (workdir / "logs").mkdir()
    set_logger(path=workdir / "logs")

    # Print Configuration
    logging.info(f"Configuration: {config}")

    # Save configuration
    OmegaConf.save(config, workdir / "config.yaml")

    # Setting seed
    seed_everthing(config.train.seed)
    logging.info(f"Using seed: {config.sample.seed}")

    # Load Datasets
    logging.info(f"Load dataset {config.data.name}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x * 2. - 1.,
    ])
    dataset = load_dataset_from_config(config, transform)

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.data.num_workers,
    )

    # Load model
    model = get_unet_model(config)
    logging.info(f"Num parameters: {get_num_parameters(model)}")

    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    # Training
    train_ddpm(
        model,
        optimizer,
        dataloader,
        workdir=workdir,
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="_workdir/pretrain",
        required=False,
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="<job-id>",
    )
    opt, _ = parser.parse_known_args()

    config = OmegaConf.load(opt.config)

    main(config, opt)
