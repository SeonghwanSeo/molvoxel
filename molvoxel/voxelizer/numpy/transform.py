import numpy as np
from numpy.typing import NDArray

from molvoxel.voxelizer.base.transform import BaseRandomTransform, BaseT

from ._quaternion import Q, apply_quaternion, random_quaternion

NDArrayFloat = NDArray[np.float64]


class T(BaseT):
    def __init__(self, translation: NDArrayFloat | None, quaternion: Q | None):
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
            translation = np.random.uniform(-random_translation, random_translation, size=(1, 3)).astype(np.float32)
        else:
            translation = None
        if random_rotation:
            quaternion = random_quaternion()
        else:
            quaternion = None
        return cls(translation, quaternion)


class RandomTransform(BaseRandomTransform):
    class_T = T

    def forward(self, coords: NDArrayFloat, center: NDArrayFloat | None) -> NDArrayFloat:
        return do_random_transform(coords, center, self.random_translation, self.random_rotation)


def do_transform(
    coords: NDArrayFloat,
    center: NDArrayFloat | None = None,
    translation: NDArrayFloat | None = None,
    quaternion: Q | None = None,
) -> NDArrayFloat:
    if quaternion is not None:
        if center is not None:
            center = center.reshape(1, 3)
            coords = apply_quaternion(coords - center, quaternion)
            coords += center
        else:
            coords = apply_quaternion(coords, quaternion)
        if translation is not None:
            coords += translation
    if translation is not None:
        coords = coords + translation
    return coords


def do_random_transform(
    coords: NDArrayFloat,
    center: NDArrayFloat | None = None,
    random_translation: float | None = 0.0,
    random_rotation: bool = False,
) -> NDArray:

    if random_rotation:
        quaternion = random_quaternion()
    else:
        quaternion = None

    if (random_translation is not None) and (random_translation > 0.0):
        translation = np.random.uniform(-random_translation, random_translation, size=(1, 3)).astype(np.float32)
    else:
        translation = None

    return do_transform(coords, center, translation, quaternion)
