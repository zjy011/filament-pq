"""E7a: train on train IDs, early-stop using native-resolution validation PQ."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from filament.data import Dataset
from filament.evaluate import Counts, evaluate_pair
from filament.experiment import provenance, write_json
from filament.pipeline import load_config
from filament.splits import load_split, split_fingerprint
from filament.unet.data import TrainingDataset
from filament.unet.inference import predict_image
from filament.unet.model import build_model, dice_bce, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--split", default="splits/main.json")
    parser.add_argument("--postprocess", default="configs/baseline.yaml")
    parser.add_argument("--out", type=Path, default=Path("results/unet"))
    parser.add_argument("--encoder", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train", type=int, help="Smoke test only; recorded in provenance")
    parser.add_argument("--limit-val", type=int, help="Smoke test only; recorded in provenance")
    args = parser.parse_args()
    if args.size < 32 or args.size % 32 or min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("Use size divisible by 32 and positive epochs/patience/batch size")
    if not 0 < args.threshold < 1 or args.lr <= 0:
        parser.error("Invalid learning rate or probability threshold")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset = Dataset(args.annotations, args.images)
    split = load_split(dataset, args.split)
    train_ids = split["train"][: args.limit_train]
    val_ids = split["val"][: args.limit_val]
    if not train_ids or not val_ids:
        parser.error("Training and validation partitions must be nonempty")
    config = load_config(args.postprocess)
    training = TrainingDataset(dataset, train_ids, args.size, args.seed, config.disk_inset)
    loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    device = select_device(args.device)
    model = build_model(args.encoder, weights=None if args.no_pretrained else "imagenet").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "provenance.json",
        {
            **provenance(dataset, args.split, config),
            "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "train_ids": train_ids,
            "val_ids": val_ids,
            "device": str(device),
            "torch": torch.__version__,
            "loss": "Dice + BCE",
        },
    )
    best, stale, history = -1.0, 0, []
    for epoch in range(args.epochs):
        training.epoch = epoch
        model.train()
        loss_total = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = dice_bce(model(inputs.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(inputs)
        model.eval()
        validation = Counts()
        for key in val_ids:
            observation = dataset.observations[key]
            pred = predict_image(
                model, dataset.read_image(key), args.size, config, args.threshold, device
            )
            for annotation in observation.annotations:
                validation += Counts(
                    **evaluate_pair(annotation.rles(observation.shape), pred)["counts"]
                )
        metrics = validation.metrics()
        history.append(
            {"epoch": epoch + 1, "loss": loss_total / len(training), "validation": metrics}
        )
        write_json(args.out / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        if metrics["pq"] > best:
            best, stale = metrics["pq"], 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder": args.encoder,
                    "size": args.size,
                    "epoch": epoch + 1,
                    "validation_pq": best,
                    "seed": args.seed,
                    "dataset_sha256": dataset.sha256,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "split_sha256": split_fingerprint(args.split),
                    "postprocess": asdict(config),
                    "threshold": args.threshold,
                },
                args.out / "best.pt",
            )
        else:
            stale += 1
        if stale >= args.patience:
            break


if __name__ == "__main__":
    main()
