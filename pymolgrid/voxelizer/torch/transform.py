import torch
from torch import Tensor, FloatTensor

from typing import Tuple, Union, Optional
from ._quaternion import random_quaternion, apply_quaternion, Q

class T() :
    def __init__(self, translation: Optional[FloatTensor], quaternion: Optional[Q]) :
        self.translation = translation
        self.quaternion = quaternion

    def __call__(self, coords, center) :
        return do_transform(coords, center, self.translation, self.quaternion)

    @classmethod
    def create(
        cls,
        random_translation: float = 0.0,
        random_rotation: bool = False,
    ) :
        if random_translation > 0.0:
            translation = (torch.rand((1,3)) - 0.5) * (2 * random_translation)
        else :
            translation = None
        if random_rotation :
            quaternion = random_quaternion()
        else :
            quaternion = None
        return cls(translation, quaternion)

class RandomTransform() :
    def __init__(
        self,
        random_translation: float = 0.0,
        random_rotation: bool = False,
    ) :
        self.random_translation = random_translation
        self.random_rotation = random_rotation

    def forward(self, coords: FloatTensor, center: Optional[FloatTensor]) -> FloatTensor:
        return do_random_transform(coords, center, self.random_translation, self.random_rotation)

    def get_transform(self) -> T :
        return T.create(self.random_translation, self.random_rotation)

def do_transform(
    coords: FloatTensor,
    center: Optional[FloatTensor] = None,
    translation: Optional[FloatTensor] = None,
    quaternion: Optional[Q] = None,
) -> FloatTensor:
    device = coords.device
    if quaternion is not None :
        if center is not None :
            center = center.view(1,3)
            coords = apply_quaternion(coords - center, quaternion)
            coords.add_(center)
        else :
            coords = apply_quaternion(coords, quaternion)

        if translation is not None :
            return coords.add_(translation.to(device))
    else :
        if translation is not None :
            return coords + translation.to(device)
    return coords

def do_random_transform(
    coords: FloatTensor,
    center: Optional[FloatTensor] = None,
    random_translation: Optional[float] = 0.0,
    random_rotation: bool = False,
) -> FloatTensor:
    device = coords.device

    if random_rotation :
        quaternion = random_quaternion()
    else :
        quaternion = None

    if (random_translation is not None) and (random_translation > 0.0) :
        translation = (torch.rand((1,3), device=device) - 0.5) * (2 * random_translation)
    else :
        translation = None
    
    return do_transform(coords, center, translation, quaternion)

