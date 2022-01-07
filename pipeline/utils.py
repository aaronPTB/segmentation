from pathlib import Path
from typing import Union, Dict, List, Tuple
import torch

EPSILON = 1e-15

def get_id2_file_paths(path):
    return {x.stem: x for x in Path(path).glob("*.*")}


def get_samples(image_path, mask_path):
    """Couple masks and images.
    Args:
        image_path:
        mask_path:
    """

    image2path = get_id2_file_paths(image_path)
    mask2path = get_id2_file_paths(mask_path)

    return [(image_file_path, mask2path[file_id]) for file_id, image_file_path in image2path.items()]

def binary_mean_iou(logits, targets):
    output = (logits > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result