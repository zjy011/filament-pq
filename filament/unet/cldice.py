"""2D soft-clDice adapted from jocpae/clDice (MIT; see THIRD_PARTY_NOTICES.md).

The official foreground loss uses batch-pooled sums, smoothing 1 and ten
skeleton iterations. Here the single channel already represents foreground.
"""

import torch
from torch.nn import functional as F


def soft_erode(image):
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_skeleton(image, iterations=10):
    def opening(value):
        return F.max_pool2d(soft_erode(value), 3, stride=1, padding=1)

    skeleton = F.relu(image - opening(image))
    for _ in range(iterations):
        image = soft_erode(image)
        delta = F.relu(image - opening(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice(probability, target, iterations=10):
    probability, target = probability.float(), target.float()
    prediction_skeleton = soft_skeleton(probability, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    precision = ((prediction_skeleton * target).sum() + 1) / (prediction_skeleton.sum() + 1)
    sensitivity = ((target_skeleton * probability).sum() + 1) / (target_skeleton.sum() + 1)
    return 1 - 2 * precision * sensitivity / (precision + sensitivity)
