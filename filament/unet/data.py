"""Choose one annotator deterministically per observation and epoch."""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from filament.disk import detect_disk
from filament.masks import union_mask


def prepare_input(image, size, inset):
    disk = detect_disk(image, inset).mask
    masked = image.astype(np.float32) / 255
    masked[~disk] = 0
    small = cv2.resize(masked, (size, size), interpolation=cv2.INTER_AREA)
    return small, disk


class TrainingDataset(TorchDataset):
    def __init__(self, dataset, ids, size=512, seed=543, inset=0.03, cache=None):
        self.dataset, self.ids = dataset, list(ids)
        self.size, self.seed, self.inset = size, seed, inset
        self.epoch = 0
        self.cache = (
            None if cache is None else Path(cache) / f"{dataset.sha256[:16]}_{size}_{inset}_v1"
        )
        if self.cache is not None:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.ids)

    def load_arrays(self, index):
        observation = self.dataset.observations[self.ids[index]]
        path = None if self.cache is None else self.cache / f"{observation.id}.npz"
        if path is not None and path.exists():
            with np.load(path) as saved:
                return saved["image"], saved["targets"]
        image, disk = prepare_input(self.dataset.read_image(observation.id), self.size, self.inset)
        targets = []
        for annotation in observation.annotations:
            target = union_mask(annotation.rles(observation.shape), observation.shape) & disk
            targets.append(
                cv2.resize(
                    target.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST
                )
            )
        targets = np.stack(targets)
        if path is not None:
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, image=image, targets=targets)
            temporary.replace(path)
        return image, targets

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        image, targets = self.load_arrays(index)
        target = targets[int(rng.integers(len(targets)))]
        if rng.random() < 0.5:
            image, target = np.fliplr(image), np.fliplr(target)
        if rng.random() < 0.5:
            image, target = np.flipud(image), np.flipud(target)
        rotation = int(rng.integers(4))
        image, target = np.rot90(image, rotation), np.rot90(target, rotation)
        foreground = image > 0
        image = np.clip(image * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
        image[~foreground] = 0
        return (
            torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(target[None], dtype=np.float32)),
        )
