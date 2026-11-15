"""MAGFiLO's COCO image entries represent annotation sets, not unique photos."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from filament.masks import polygon_rle


@dataclass
class AnnotationSet:
    id: str
    instances: list[dict] = field(default_factory=list)

    def rles(self, shape: tuple[int, int]) -> list[dict]:
        return [polygon_rle(a["segmentation"], *shape) for a in self.instances]


@dataclass
class Observation:
    id: str
    file_name: str
    site: str
    year: int
    height: int
    width: int
    url: str
    annotations: list[AnnotationSet] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


class Dataset:
    def __init__(self, annotations: str | Path, images: str | Path = "data/images"):
        self.source = Path(annotations)
        self.images = Path(images)
        raw = self.source.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        content = json.loads(raw)
        instances = defaultdict(list)
        instance_ids = set()
        for a in content["annotations"]:
            if str(a["id"]) in instance_ids:
                raise ValueError(f"Duplicate instance ID: {a['id']}")
            instance_ids.add(str(a["id"]))
            instances[str(a["image_id"])].append(a)
        self.observations: dict[str, Observation] = {}
        annotation_ids = set()
        for entry in content["images"]:
            annotation_id = str(entry["id"])
            if annotation_id in annotation_ids:
                raise ValueError(f"Duplicate annotation-set ID: {annotation_id}")
            annotation_ids.add(annotation_id)
            filename = entry["file_name"]
            if Path(filename).name != filename or "\\" in filename:
                raise ValueError(f"Expected a plain image filename: {filename}")
            observation_id = Path(filename).stem
            match = re.fullmatch(r"(\d{4})\d{10}(Bh|Ch|Lh|Mh|Th|Uh)", observation_id)
            if match is None:
                raise ValueError(f"Unrecognized GONG observation name: {filename}")
            observation = self.observations.setdefault(
                observation_id,
                Observation(
                    observation_id,
                    filename,
                    match[2],
                    int(match[1]),
                    int(entry["height"]),
                    int(entry["width"]),
                    entry.get("url", ""),
                ),
            )
            if observation.shape != (entry["height"], entry["width"]):
                raise ValueError(f"Inconsistent dimensions for {observation_id}")
            observation.annotations.append(AnnotationSet(annotation_id, instances[annotation_id]))
        orphans = set(instances) - annotation_ids
        if orphans:
            raise ValueError(f"Annotations refer to missing image entries: {sorted(orphans)[:3]}")
        self.observations = dict(sorted(self.observations.items()))
        for observation in self.observations.values():
            observation.annotations.sort(key=lambda a: a.id)

    def read_image(self, observation_id: str) -> np.ndarray:
        observation = self.observations[observation_id]
        path = self.images / observation.file_name
        if not path.exists():
            path = path.with_suffix(".jpg")
        if not path.exists():
            raise FileNotFoundError(f"Missing {observation.file_name} under {self.images}")
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != observation.shape:
            raise ValueError(f"Unreadable image or unexpected dimensions: {path}")
        return image

    def summary(self) -> dict:
        observations = list(self.observations.values())
        return {
            "sha256": self.sha256,
            "observations": len(observations),
            "annotation_sets": sum(len(o.annotations) for o in observations),
            "instances": sum(len(a.instances) for o in observations for a in o.annotations),
            "empty_annotation_sets": sum(
                not a.instances for o in observations for a in o.annotations
            ),
            "sites": dict(sorted(Counter(o.site for o in observations).items())),
            "annotator_counts": dict(
                sorted(Counter(len(o.annotations) for o in observations).items())
            ),
        }
