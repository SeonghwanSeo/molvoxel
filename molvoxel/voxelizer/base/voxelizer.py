import numpy as np

from typing import Tuple, Optional, Union
from numpy.typing import ArrayLike

class BaseVoxelizer() :
    SCALAR = 0
    CHANNEL_WISE = 1
    ATOM_WISE = 2
    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 48,
        atom_scale: float = 1.5,
        radii_type: str = 'scalar',
    ) :
        self._resolution = resolution
        self._dimension = dimension
        self._width = width = resolution * (dimension - 1)

        self.atom_scale = atom_scale
        self.radii_type = radii_type

        self.upper_bound: float = width / 2.
        self.lower_bound = -1 * self.upper_bound
        self._spatial_dimension = (self._dimension, self._dimension, self._dimension)
   
    @property
    def radii_type(self) -> str:
        if self._radii_type == self.SCALAR :
            return 'scalar'
        elif self._radii_type == self.CHANNEL_WISE :
            return 'channel-wise'
        else :
            return 'atom-wise'
    
    @radii_type.setter
    def radii_type(self, radii_type: str) :
        assert radii_type in ['scalar', 'channel-wise', 'atom-wise']
        if radii_type == 'scalar' :
            self._radii_type = self.SCALAR
        elif radii_type == 'channel-wise' :
            self._radii_type = self.CHANNEL_WISE
        else :
            self._radii_type = self.ATOM_WISE
    
    @property
    def is_radii_type_scalar(self) :
        return self._radii_type == self.SCALAR

    @property
    def is_radii_type_channel_wise(self) :
        return self._radii_type == self.CHANNEL_WISE

    @property
    def is_radii_type_atom_wise(self) :
        return self._radii_type == self.ATOM_WISE

    def grid_dimension(self, num_channels: int) -> Tuple[int, int, int, int]:
        return (num_channels, self._dimension, self._dimension, self._dimension)

    @property
    def spatial_dimension(self) -> Tuple[int, int, int, int]:
        return self._spatial_dimension

    @property
    def resolution(self) -> float :
        return self._resolution

    @property
    def dimension(self) -> int :
        return self._dimension

    @property
    def width(self) -> float :
        return self._width

    """ Forward """
    def forward(
        self,
        coords: ArrayLike,
        center: Optional[ArrayLike],
        channels: ArrayLike,
        radii: Union[float, ArrayLike],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None
    ) -> ArrayLike :
        """
        coords: (V, 3)
        center: (3,)
        types: (V, C) or (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out_grid: (C,D,H,W)
        """
        if np.ndim(channels) == 1 :
            types = channels
            return self.forward_types(coords, center, types, radii, random_translation, random_rotation, out_grid)
        else :
            features = channels
            return self.forward_features(coords, center, features, radii, random_translation, random_rotation, out_grid)

    __call__ = forward

    def forward_types(
        self,
        coords: ArrayLike,
        center: Optional[ArrayLike],
        types: ArrayLike,
        radii: Union[float, ArrayLike],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None
    ) -> ArrayLike :
        raise NotImplementedError

    def forward_features(
        self,
        coords: ArrayLike,
        center: Optional[ArrayLike],
        features: ArrayLike,
        radii: Union[float, ArrayLike],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None
    ) -> ArrayLike :
        raise NotImplementedError

    def get_empty_grid(self, num_channels: int, batch_size: Optional[int] = None, init_zero: bool = False) -> ArrayLike:
        raise NotImplementedError

    def asarray(self, array: ArrayLike, obj: str) -> ArrayLike:
        raise NotImplementedError

