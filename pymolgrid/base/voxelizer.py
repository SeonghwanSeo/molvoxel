import math

from typing import Tuple

class BaseVoxelizer() :
    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 48,
        atom_scale: float = 1.5,
        channel_wise_radii: bool = False,
    ) :
        self._resolution = resolution
        self._dimension = dimension
        self._width = width = resolution * (dimension - 1)

        self.atom_scale = atom_scale
        self.channel_wise_radii = channel_wise_radii

        self.upper_bound: float = width / 2.
        self.lower_bound = -1 * self.upper_bound
    
    def grid_dimension(self, num_channels: int) -> Tuple[int, int, int, int]:
        return (num_channels, self._dimension, self._dimension, self._dimension)

    @property
    def spatial_dimension(self) -> Tuple[int, int, int]:
        return (self._dimension, self._dimension, self._dimension)
    
    @property
    def resolution(self) -> float :
        return self._resolution

    @property
    def dimension(self) -> int :
        return self._dimension

    @property
    def width(self) -> float :
        return self._width
