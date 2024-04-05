# Differentially Private Diffusion Probabilistic Models for Image Synthesis (dp-promise)

This is the implementation of "dp-promise: Differentially Private Diffusion Probabilistic Models for Image Synthesis". The architecture of model is based on the improved DDPM repository (https://github.com/openai/improved-diffusion).

## Requirements

The code is adapted for Python 3.9, PyTorch 2.0.0 and torchvison 0.15.1 with CUDA 11.7. Run the following command to install requirements

```shell
$ pip install -r requirements.txt
```

## Preparation

Create the data folder

```shell
$ mkdir _data
```

For MNIST, Fashion MNIST and CIFAR-10, dataset will be downloaded automatically.

- CelebA

    Download dataset from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, then

    ```shell
    $ python prepare_celeba.py --source _data/CelebA/img_align_celeba --dest _data/CelebA/processed
    # For 64 x 64 resolution
    $ python prepare_celeba.py --source _data/CelebA/img_align_celeba --dest _data/CelebA/processed --width 64 --height 64
    ```

- ImageNet

    Download dataset from https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data, then

    ```shell
    $ python prepare_imagenet.py --source _data/ImageNet --dest _data/ImageNet/processed
    # For 64 x 64 resolution
    $ python prepare_imagenet.py --source _data/ImageNet --dest _data/ImageNet/processed --width 64 --height 64
    ```

## Pre-training

```shell
$ python pretrain.py --config configs/vanilla/<dataset>/config.yaml
```

## Training

```shell
$ python train.py --config configs/dp_promise/<dataset>/config.yaml
```

## Evaluation

Note that the FID and IS evaluation is adapted for TensorFlow 2.12.0 and tensorflow-gan 2.1.0

```shell
$ pip install requirements_eval.txt
```

First, compute dataset statistics

```shell
$ cd evaluation
# For MNIST
$ python compute_dataset_stat.py --dataset {mnist,fmnist,cifar10}
$ python compute_dataset_stat.py --dataset celeba --path ../_data/CelebA/processed
```

Then, run the following command to conduct evaluation

```shell
# downstream classifier
$ python eval_downstream.py --dataset <dataset> --synthesis_path <path/to/synthesis.npz> --output_path <output_path>
# scikit-learn classifier
$ python eval_scikit.py --dataset <dataset> --synthesis_path <path/to/synthesis.npz> --output_path <output_path>
# FID score and Inception Score
$ python eval_vision.py --dataset <dataset> --synthesis_path <path/to/synthesis.npz> --output_path <output_path>
```
