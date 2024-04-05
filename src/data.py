from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import Grayscale


class LabelDataset(Dataset):

    def __init__(self, lables) -> None:
        super().__init__()
        self.labels = lables

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)


class ImageDataset(Dataset):

    def __init__(self, images, transform=None) -> None:
        super().__init__()
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


class LabelImageDataset(Dataset):

    def __init__(self, images, labels, transform=None) -> None:
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)


class CelebA(Dataset):

    def __init__(self, path, transform=None) -> None:
        super().__init__()
        self.images = [f for f in Path(path).glob("*.png")]
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.images)


class ImageNet(Dataset):
    def __init__(self, path, transform=None) -> None:
        super().__init__()
        self.images = [f for f in Path(path).glob("*.png")]
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image)
        if image.mode == "L":
            image = Grayscale(3)(image)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.images)
