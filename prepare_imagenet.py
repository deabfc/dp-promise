import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm


def center_crop(img, width, height):
    img = np.array(img)
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2:(img.shape[0] + crop) // 2,
              (img.shape[1] - crop) // 2:(img.shape[1] + crop) // 2]
    channels = img.shape[2] if img.ndim == 3 else 1
    
    img = Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
    img = img.resize((width, height), Image.LANCZOS)
    return img


def main(args):
    source = Path(args.source)
    image_fnames = [f for f in source.rglob("*.JPEG")]
    transform = center_crop
    width, height = args.width, args.height
    dest = Path(args.dest)
    dest.mkdir(parents=True)
    for idx, fname in tqdm(enumerate(image_fnames)):
        image = Image.open(fname)
        if image.mode not in ["RGB", "L"]:
            continue
        image = transform(image, width, height)
        image.save(f"{dest}/{idx:08d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    main(args)
