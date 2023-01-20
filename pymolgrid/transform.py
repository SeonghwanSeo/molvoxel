import torch
from torch import Tensor, FloatTensor

from typing import Tuple, Union
from .quaternion import random_quaternion, apply_quaternion

class Transform() :
    def __init__(
        self,
        random_translation: float = 0.0,
        random_rotation: bool = False,
    ) :
        self.center = center
        self.random_translation = random_translation
        self.random_rotation = random_rotation

    def forward(self, coords: FloatTensor, center: Tuple[float, float, float]) :
        return self.do_transform(coords, center, self.random_translation, self.random_rotation)

    @staticmethod
    def do_transform(
        coords: FloatTensor,
        center: Union[Tuple[float, float, float], FloatTensor, None] = None,
        random_translation: float = 0.0,
        random_rotation: bool = False,
    ) -> FloatTensor:
        device = coords.device

        if random_rotation :
            quaternion = random_quaternion()
            if center is not None :
                if not isinstance(center, Tensor) :
                    center = torch.FloatTensor([center], dtype=device)
                if center.size() == (3,) :
                    center = center.unsqueeze(0)
                coords = apply_quaternion(coords - center, quaternion)
                coords.add_(center)
            else :
                coords = apply_quaternion(coords, quaternion)

        if random_translation > 0.0:
            # translate ~ Uniform(-d, d)
            translate = (torch.rand((1,3), device=device).sub_(-0.5)).mul_(2 * random_translation)
            if random_rotation :
                coords.add_(translate)
            else :
                coords = coords + translate
        return coords
