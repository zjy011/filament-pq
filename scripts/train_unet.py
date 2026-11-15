"""E7a: train on train IDs, early-stop using native-resolution validation PQ."""

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
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
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--loss", choices=["L1", "L2"], default="L1")
    parser.add_argument("--cldice-iterations", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cache", default="cache/training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train", type=int, help="Smoke test only; recorded in provenance")
    parser.add_argument("--limit-val", type=int, help="Smoke test only; recorded in provenance")
    args = parser.parse_args()
    if args.size < 32 or args.size % 32 or min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("Use size divisible by 32 and positive epochs/patience/batch size")
    if (
        not 0 < args.threshold < 1
        or args.lr <= 0
        or min(args.accumulate, args.cldice_iterations) < 1
    ):
        parser.error("Invalid learning rate or probability threshold")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        parser.error("Choose a new output directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset = Dataset(args.annotations, args.images)
    split = load_split(dataset, args.split)
    train_ids = split["train"][: args.limit_train]
    val_ids = split["val"][: args.limit_val]
    if not train_ids or not val_ids:
        parser.error("Training and validation partitions must be nonempty")
    config = load_config(args.postprocess)
    training = TrainingDataset(
        dataset, train_ids, args.size, args.seed, config.disk_inset, args.cache
    )
    device = select_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    model = build_model(
        args.encoder, weights=None if args.no_pretrained or args.resume else "imagenet"
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        **provenance(dataset, args.split, config),
        "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_ids": train_ids,
        "val_ids": val_ids,
        "device": str(device),
        "torch": torch.__version__,
        "loss": "Dice + BCE" if args.loss == "L1" else "Dice + BCE + 0.5 soft-clDice",
        "amp": amp,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    manifest = json.loads(json.dumps(manifest))
    best, stale, history, start_epoch = -1.0, 0, [], 0
    if args.resume:
        previous = json.loads((args.out / "provenance.json").read_text(encoding="utf-8"))
        for key in ("dataset_sha256", "split_sha256", "config", "train_ids", "val_ids", "amp"):
            if previous[key] != manifest[key]:
                parser.error(f"Resume differs in {key}")
        for key in (
            "encoder",
            "size",
            "lr",
            "batch_size",
            "seed",
            "threshold",
            "no_pretrained",
            "patience",
            "loss",
            "accumulate",
            "cldice_iterations",
        ):
            defaults = {"loss": "L1", "accumulate": 1, "cldice_iterations": 10}
            if previous["training"].get(key, defaults.get(key)) != manifest["training"][key]:
                parser.error(f"Resume differs in {key}")
        checkpoint = torch.load(args.out / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        best, stale, history = checkpoint["best"], checkpoint["stale"], checkpoint["history"]
        start_epoch = checkpoint["epoch"]
        torch.set_rng_state(checkpoint["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        write_json(args.out / "resume.json", manifest)
    else:
        write_json(args.out / "provenance.json", manifest)

    def save_checkpoint(name, value):
        temporary = args.out / (name + ".tmp")
        torch.save(value, temporary)
        temporary.replace(args.out / name)

    for epoch in range(start_epoch, args.epochs):
        if stale >= args.patience:
            break
        started = perf_counter()
        training.epoch = epoch
        loader = DataLoader(
            training,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            generator=torch.Generator().manual_seed(args.seed + epoch),
            pin_memory=amp,
        )
        model.train()
        loss_total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch, (inputs, targets) in enumerate(loader, 1):
            targets = targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(inputs.to(device))
                loss = dice_bce(logits, targets)
                if args.loss == "L2":
                    from filament.unet.cldice import soft_cldice

                    loss = loss + 0.5 * soft_cldice(
                        logits.float().sigmoid(), targets, args.cldice_iterations
                    )
            # Weight the final, possibly incomplete accumulation group by images.
            group_start = ((batch - 1) // args.accumulate) * args.accumulate * args.batch_size
            group_images = min(args.accumulate * args.batch_size, len(training) - group_start)
            scaler.scale(loss * len(inputs) / group_images).backward()
            if batch % args.accumulate == 0 or batch == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(loss.detach()) * len(inputs)
            if batch % 20 == 0 or batch == len(loader):
                progress = {
                    "status": "training",
                    "epoch": epoch + 1,
                    "batch": batch,
                    "batches": len(loader),
                    "seconds": perf_counter() - started,
                    "best_pq": best,
                }
                write_json(args.out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        model.eval()
        write_json(args.out / "progress.json", {"status": "validation", "epoch": epoch + 1})
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
            {
                "epoch": epoch + 1,
                "loss": loss_total / len(training),
                "validation": metrics,
                "seconds": perf_counter() - started,
            }
        )
        write_json(args.out / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        if metrics["pq"] > best:
            best, stale = metrics["pq"], 0
            save_checkpoint(
                "best.pt",
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
                    "loss": args.loss,
                    "cldice_iterations": args.cldice_iterations,
                },
            )
        else:
            stale += 1
        save_checkpoint(
            "last.pt",
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "stale": stale,
                "history": history,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        )
    write_json(
        args.out / "progress.json",
        {
            "status": "complete",
            "epochs": len(history),
            "best_pq": best,
            "early_stopped": stale >= args.patience,
        },
    )


if __name__ == "__main__":
    main()
