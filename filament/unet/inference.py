"""Native-resolution instance conversion shares classical M4-M6."""

from dataclasses import asdict
from time import perf_counter

import cv2
import torch

from filament.evaluate import aggregate_counts, evaluate_pair
from filament.masks import label_rles
from filament.pipeline import postprocess
from filament.unet.data import prepare_input
from filament.unet.model import build_model, select_device


@torch.inference_mode()
def predict_image(model, image, size, config, threshold, device):
    small, disk = prepare_input(image, size, config.disk_inset)
    tensor = torch.from_numpy(small[None, None]).to(device)
    probability = model(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(
        probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return label_rles(postprocess(probability >= threshold, disk, config))


def predict_dataset(dataset, ids, checkpoint_path, config, threshold=0.5, device_name="auto"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["dataset_sha256"] != dataset.sha256:
        raise ValueError("Checkpoint was trained on a different annotation file")
    device = select_device(device_name)
    torch.set_num_threads(4)
    model = build_model(checkpoint["encoder"], weights=None)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    results = []
    for key in ids:
        observation = dataset.observations[key]
        image = dataset.read_image(key)
        start = perf_counter()
        prediction = predict_image(model, image, checkpoint["size"], config, threshold, device)
        seconds = perf_counter() - start
        comparisons = [
            {"annotation_id": a.id, **evaluate_pair(a.rles(observation.shape), prediction)}
            for a in observation.annotations
        ]
        results.append(
            {
                "id": key,
                "site": observation.site,
                "seconds": seconds,
                "predictions": prediction,
                "comparisons": comparisons,
                "counts": asdict(aggregate_counts(comparisons)),
            }
        )
    return results
