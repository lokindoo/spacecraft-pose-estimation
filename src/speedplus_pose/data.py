"""
Datasets, dataloaders, and the offline image preparation.

Preparation is offline by necessity: 60k images at 1920x1200 will not stream
from Drive fast enough to keep a Colab GPU busy, so images are downscaled once
to 320x512, zipped, and unpacked to local disk at the start of a session.
``materialise_images`` is that unpack step, and is a no-op when the images are
already on local disk.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ImageConfig, TrainConfig

IMAGE_GLOB = "**/*.jpg"


class SpeedPlusDataset(Dataset):
    """
    One SPEED+ split: a directory of images plus a label frame.

    The label frame must be in ``labels.COLUMN_ORDER`` (filename first, then
    the seven pose values), because both are read positionally.
    """

    def __init__(self, images_path: str | Path, labels: pd.DataFrame, transform=None):
        self.images_path = Path(images_path)
        self.df = labels.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.images_path / row.iloc[0])
        label = np.array(row.iloc[1:], dtype=np.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def filenames(self) -> np.ndarray:
        return self.df.iloc[:, 0].to_numpy()


def build_dataloader(
    dataset: Dataset, train: TrainConfig, shuffle: bool, batch_size: int | None = None
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size or train.batch_size,
        shuffle=shuffle,
        num_workers=train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# --------------------------------------------------------------------------
# offline preparation
# --------------------------------------------------------------------------


def downscale_image(image: Image.Image, config: ImageConfig) -> Image.Image:
    """
    Downscale to the study resolution.

    PIL's LANCZOS rather than skimage's ``resize``, which the original
    preparation notebooks used: the training transform loads with PIL and
    resizes with torchvision, so matching the library avoids a second, subtly
    different resampling path. The published subsets came from the skimage
    route, and regenerating them here gives visually identical 320x512 crops;
    use the released subset archives if you need byte-identical inputs.
    """
    return image.resize((config.width, config.height), Image.LANCZOS)


def downscale_directory(
    source: str | Path,
    destination: str | Path,
    filenames: set[str] | None = None,
    config: ImageConfig | None = None,
    skip_existing: bool = True,
) -> int:
    """
    Downscale every image in ``source`` (optionally restricted to ``filenames``)
    into a flat ``destination`` directory. Returns the number written.
    """
    config = config or ImageConfig()
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in source.glob(IMAGE_GLOB)]
    if filenames is not None:
        candidates = [p for p in candidates if p.name in filenames]
    if skip_existing:
        candidates = [p for p in candidates if not (destination / p.name).exists()]

    for path in tqdm(candidates, desc=f"downscaling -> {destination.name}"):
        with Image.open(path) as image:
            downscale_image(image, config).save(destination / path.name, quality=95)
    return len(candidates)


def archive_images(images_path: str | Path, archive_path: str | Path) -> int:
    """
    Zip a prepared image directory, stored rather than deflated: the JPEGs are
    already compressed, so deflate costs time and saves nothing. Returns the
    file count.
    """
    images_path, archive_path = Path(images_path), Path(archive_path)
    paths = sorted(images_path.glob(IMAGE_GLOB))
    if archive_path.exists():
        raise FileExistsError(f"{archive_path} already exists; remove it to rebuild")

    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in tqdm(paths, desc=f"archiving -> {archive_path.name}"):
            archive.write(path, arcname=path.name)
    return len(paths)


def materialise_images(
    subset_path: str | Path,
    local_path: str | Path | None = None,
    expected: int | None = None,
) -> Path:
    """
    Return a local directory holding this subset's images.

    If ``subset_path/images`` already exists, as it does when running locally,
    it is used as is. Otherwise ``subset_path/images.zip`` is copied to fast
    local storage and unpacked, which is the Colab pattern: the subset lives on
    a mounted Drive, and reading 30k small files across the mount is unusably
    slow.
    """
    subset_path = Path(subset_path)

    direct = subset_path / "images"
    if direct.is_dir() and any(direct.glob(IMAGE_GLOB)):
        images_path = direct
    else:
        archive = subset_path / "images.zip"
        if not archive.exists():
            raise FileNotFoundError(
                f"{subset_path} has neither an images/ directory nor images.zip - "
                "run scripts/prepare_data.py first"
            )
        images_path = Path(local_path or Path(os.environ.get("TMPDIR", "/tmp")) / subset_path.name)
        if not (images_path.is_dir() and any(images_path.glob(IMAGE_GLOB))):
            staged = images_path.with_suffix(".zip")
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, staged)
            shutil.unpack_archive(staged, images_path)

    count = len(list(images_path.glob(IMAGE_GLOB)))
    if expected is not None and count != expected:
        raise AssertionError(
            f"{images_path}: expected {expected} images, found {count}. The label "
            "files and the image archive have drifted apart."
        )
    return images_path
