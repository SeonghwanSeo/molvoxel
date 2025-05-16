import torch
from torch import FloatTensor

from molvoxel.voxelizer.base.transform import BaseRandomTransform, BaseT

from ._quaternion import Q, apply_quaternion, random_quaternion


class T(BaseT):
    def __init__(self, translation: FloatTensor | None, quaternion: Q | None):
        self.translation = translation
        self.quaternion = quaternion

    def __call__(self, coords, center):
        return do_transform(coords, center, self.translation, self.quaternion)

    @classmethod
    def create(
        cls,
        random_translation: float = 0.0,
        random_rotation: bool = False,
    ):
        if random_translation > 0.0:
            translation = (torch.rand((1, 3)) - 0.5) * (2 * random_translation)
        else:
            translation = None
        if random_rotation:
            quaternion = random_quaternion()
        else:
            quaternion = None
        return cls(translation, quaternion)


class RandomTransform(BaseRandomTransform):
    class_T = T

    def forward(self, coords: FloatTensor, center: FloatTensor | None) -> FloatTensor:
        return do_random_transform(coords, center, self.random_translation, self.random_rotation)


def do_transform(
    coords: FloatTensor,
    center: FloatTensor | None = None,
    translation: FloatTensor | None = None,
    quaternion: Q | None = None,
) -> FloatTensor:
    device = coords.device
    if quaternion is not None:
        if center is not None:
            center = center.view(1, 3)
            coords = apply_quaternion(coords - center, quaternion)
            coords.add_(center)
        else:
            coords = apply_quaternion(coords, quaternion)

        if translation is not None:
            return coords.add_(translation.to(device))
    else:
        if translation is not None:
            return coords + translation.to(device)
    return coords


def do_random_transform(
    coords: FloatTensor,
    center: FloatTensor | None = None,
    random_translation: float | None = 0.0,
    random_rotation: bool = False,
) -> FloatTensor:
    device = coords.device

    if random_rotation:
        quaternion = random_quaternion()
    else:
        quaternion = None

    if (random_translation is not None) and (random_translation > 0.0):
        translation = (torch.rand((1, 3), device=device) - 0.5) * (2 * random_translation)
    else:
        translation = None

    return do_transform(coords, center, translation, quaternion)
