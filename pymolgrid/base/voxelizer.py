import math
import types
import functools

from typing import Tuple, Optional, Union
from numpy.typing import ArrayLike

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
        self.spatial_dimension = (self._dimension, self._dimension, self._dimension)
    
    def grid_dimension(self, num_channels: int) -> Tuple[int, int, int, int]:
        return (num_channels, self._dimension, self._dimension, self._dimension)

    @property
    def resolution(self) -> float :
        return self._resolution

    @property
    def dimension(self) -> int :
        return self._dimension

    @property
    def width(self) -> float :
        return self._width

    def get_empty_grid(self, num_channels: int, batch_size: Optional[int] = None, init_zero: bool = False) -> ArrayLike:
        raise NotImplemented

    def asarray(self, array: ArrayLike, obj: str) :
        raise NotImplemented

    def forward(
        self,
        coords: ArrayLike,
        center: Optional[ArrayLike],
        channels: ArrayLike,
        radii: Union[float, ArrayLike],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[ArrayLike] = None
    ) -> ArrayLike :
        """
        coords: (V, 3)
        center: (3,)
        channels: (V, C) or (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out: (C,D,H,W)
        """
        raise NotImplemented

    __call__ = forward
