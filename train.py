import argparse
from pathlib import Path
from datetime import datetime
import logging

from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from torchvision import transforms

from src.utils import get_unet_model
from src.utils import load_dataset_from_config
from src.utils import set_logger
from src.utils import get_num_parameters
from src.utils import seed_everthing
from src.runners import train_dp_promise
from src.mechanisms import eps_from_config


def main(config, opt):
    # Prepare
    time = datetime.now().strftime("%m%d-%H:%M:%S")
    running_id = f"dp-promise_{opt.job_id}_{config.data.name}_eps{config.dp.epsilon}_{time}"
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
    dataset = load_dataset_from_config(config, transform, train=True)

    dataloader1 = DataLoader(
        dataset,
        batch_size=config.train.batch_size1,
        shuffle=True,
        pin_memory=True,
        num_workers=config.data.num_workers,
    )

    dataloader2 = DataLoader(
        dataset,
        batch_size=config.train.batch_size2,
        shuffle=True,
        pin_memory=True,
        num_workers=config.data.num_workers,
    )

    # Calculate privacy budget
    if config.dp.epsilon == "inf":
        logging.info("Epsilon: inf")
    else:
        eps = eps_from_config(config)
        logging.info(f"Satisfy ({eps}, {config.dp.delta})-DP")

    # Load model
    model = get_unet_model(
        config, checkpoint_path=config.train.get("ckpt_path"))
    logging.info(f"Num parameters: {get_num_parameters(model)}")

    logging.info(f"Start training...")

    # Training
    train_dp_promise(
        model,
        dataloader1,
        dataloader2,
        config=config,
        workdir=workdir,
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
        default="_workdir/dp_promise",
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
