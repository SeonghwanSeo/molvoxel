import math
import numba as nb
import numpy as np
import itertools

from typing import Tuple, Union, Optional, Dict, List
from numpy.typing import NDArray

from pymolgrid.base import BaseVoxelizer
from .transform import do_random_transform

from . import func_features, func_types

NDArrayInt = NDArray[np.int16]
NDArrayFloat = NDArray[np.float32]
NDArrayBool = NDArray[np.bool_]

"""
radii: input value of pymolgrid
atom_size: radii * atom_scale           # atom boundary
"""

class Voxelizer(BaseVoxelizer) :
    LIB='Numba'
    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 64,
        atom_scale: float = 1.5,
        density: str = 'gaussian',
        channel_wise_radii: bool = False,
        blockdim: Optional[int] = None
    ) :
        super(Voxelizer, self).__init__(resolution, dimension, atom_scale, channel_wise_radii)
        assert density in ['gaussian', 'binary']
        self._density = density.lower()
        self._gaussian = (self._density == 'gaussian')
        self._setup_block(blockdim)

    @property
    def density(self) -> str :
        return self._density
    
    @density.setter
    def density(self, density: str) :
        density = density.lower()
        assert density in ['gaussian', 'binary'], f'density ({density}) should be gaussian or binary'
        self._density = density
        self._gaussian = (density == 'gaussian')

    def _setup_block(self, blockdim) :
        blockdim = blockdim if blockdim is not None else 8
        self.blockdim = blockdim

        axis = np.arange(self.dimension, dtype=np.float32) * self.resolution - (self.width/2.)
        self.num_blocks = num_blocks = math.ceil(self.dimension / blockdim)
        if self.num_blocks > 1 :
            self.grid = None
            self.grid_block_dict = {}
            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3) :
                x_axis = axis[xidx*blockdim : (xidx+1)*blockdim]
                y_axis = axis[yidx*blockdim : (yidx+1)*blockdim]
                z_axis = axis[zidx*blockdim : (zidx+1)*blockdim]
                self.grid_block_dict[(xidx, yidx, zidx)] = (x_axis, y_axis, z_axis)

            self.bounds = [(axis[idx*blockdim] + (self.resolution / 2.))
                                                for idx in range(1, num_blocks)]
        else :
            self.grid_block_dict = None
            self.grid = (axis, axis, axis)

    def get_empty_grid(self, num_channels: int, batch_size: Optional[int] = None, init_zero: bool = True) -> NDArrayFloat:
        if init_zero :
            if batch_size is None :
                return np.zeros(self.grid_dimension(num_channels), dtype=np.float32)
            else :
                return np.zeros((batch_size,) + self.grid_dimension(num_channels), dtype=np.float32)
        else :
            if batch_size is None :
                return np.empty(self.grid_dimension(num_channels), dtype=np.float32)
            else :
                return np.empty((batch_size,) + self.grid_dimension(num_channels), dtype=np.float32)

    """ Forward """
    def forward(
        self,
        coords: NDArrayFloat,
        center: NDArrayFloat,
        channels: Union[NDArrayFloat, NDArrayInt],
        radii: Union[float, NDArrayFloat],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat :
        """
        coords: (V, 3)
        center: (3,)
        types: (V, C) or (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out: (C,D,H,W)
        """
        if channels.ndim == 1 :
            types = channels
            return self.forward_types(coords, center, types, radii, random_translation, random_rotation, out)
        else :
            features = channels
            return self.forward_features(coords, center, features, radii, random_translation, random_rotation, out)

    __call__ = forward

    """ VECTOR """
    def forward_features(
        self,
        coords: NDArrayFloat,
        center: Optional[NDArrayFloat],
        features: NDArrayFloat,
        radii: Union[float, NDArrayFloat],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat :
        """
        coords: (V, 3)
        center: (3,)
        features: (V, C)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out: (C,D,H,W)
        """
        self._check_args_features(coords, features, radii, out)

        # Set Coordinate
        if center is not None :
            coords = coords - center.reshape(1,3)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        # DataType
        coords = self._dtypechange(coords, np.float32)
        features = self._dtypechange(features, np.float32)
        if not np.isscalar(radii) :
            radii = self._dtypechange(radii, np.float32)

        # Set Out
        if out is None :
            C = features.shape[1]
            out = self.get_empty_grid(C, init_zero=True)
        else :
            out.fill(0.)

        # Set Atom Radii
        is_atom_wise_radii = (not np.isscalar(radii) and not self.channel_wise_radii)
        if self.channel_wise_radii :
            atom_size = radii.max() * self.atom_scale
        else :
            atom_size = radii * self.atom_scale

        # Clipping Overlapped Atoms
        box_overlap = self._get_overlap(coords, atom_size)
        coords, features = coords[box_overlap], features[box_overlap]
        if is_atom_wise_radii:
            radii = radii[box_overlap]

        # Run
        if self.num_blocks > 1 :
            blockdim = self.blockdim
            if is_atom_wise_radii:
                atom_size = radii * self.atom_scale
            block_overlap_dict = self._get_overlap_blocks(coords, atom_size)
            
            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3) :
                start_x, end_x = xidx*blockdim, (xidx+1)*blockdim
                start_y, end_y = yidx*blockdim, (yidx+1)*blockdim
                start_z, end_z = zidx*blockdim, (zidx+1)*blockdim

                out_block = out[:, start_x:end_x, start_y:end_y, start_z:end_z]
                
                overlap = block_overlap_dict[(xidx, yidx, zidx)]
                if overlap.shape[0] == 0 :
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block, features_block = coords[overlap], features[overlap]
                radii_block = radii[overlap] if is_atom_wise_radii else radii
                self._set_grid_features(coords_block, features_block, radii_block, grid_block, out_block)
        else :
            grid = self.grid
            self._set_grid_features(coords, features, radii, grid, out)
        
        return out

    def _check_args_features(self, coords: NDArrayFloat, features: NDArrayFloat, radii: Union[float,NDArrayFloat], 
                    out: Optional[NDArrayFloat] = None) :
        V = coords.shape[0]
        C = features.shape[1]
        D = H = W = self.dimension
        assert features.shape[0] == V, f'atom features does not match number of atoms: {features.shape[0]} vs {V}'
        assert features.ndim == 2, f"atom features does not match dimension: {features.shape} vs {(V,'*')}"
        if self.channel_wise_radii :
            assert not np.isscalar(radii)
            assert radii.shape == (C,), f'radii does not match dimension (number of channels,): {radii.shape} vs {(C,)}'
        else :
            if not np.isscalar(radii) :
                assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out is not None :
            assert out.shape == (C, D, H, W), f'Output grid dimension incorrect: {out.shape} vs {(C,D,H,W)}'

    def _set_grid_features(
        self,
        coords: NDArrayFloat,
        features: NDArrayFloat,
        radii: Union[float, NDArrayFloat],
        grid: NDArrayFloat,
        out: NDArrayFloat,
    ) -> NDArrayFloat :
        """
        coords: (V, 3)
        features: (V, C)
        radii: scalar or (V, ) or (C, )
        grid: (D,), (H,), (W,)

        out: (C, D, H, W)
        """
        if self._gaussian :
            if self.channel_wise_radii :
                func_features.gaussian_channel_wise_radii(coords, features, *grid, radii, self.atom_scale, out)
            elif np.isscalar(radii) :
                func_features.gaussian_scalar_radii(coords, features, *grid, radii, self.atom_scale, out)
            else :
                func_features.gaussian_atom_wise_radii(coords, features, *grid, radii, self.atom_scale, out)
        else :
            if self.channel_wise_radii :
                func_features.binary_channel_wise_radii(coords, features, *grid, radii, self.atom_scale, out)
            elif np.isscalar(radii) :
                func_features.binary_scalar_radii(coords, features, *grid, radii, self.atom_scale, out)
            else :
                func_features.binary_atom_wise_radii(coords, features, *grid, radii, self.atom_scale, out)
        return out

    """ INDEX """
    def forward_types(
        self,
        coords: NDArrayFloat,
        center: Optional[NDArrayFloat],
        types: NDArrayInt,
        radii: Union[float, NDArrayFloat],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat :
        """
        coords: (V, 3)
        center: (3,)
        types: (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out: (C,D,H,W)
        """
        self._check_args_types(coords, types, radii, out)

        # Set Coordinate
        if center is not None :
            coords = coords - center.reshape(1, 3)
        coords = self.do_random_transform(coords, None, random_translation, random_rotation)

        # DataType
        coords = self._dtypechange(coords, np.float32)
        types = self._dtypechange(types, np.int16)
        if not np.isscalar(radii) :
            radii = self._dtypechange(radii, np.float32)

        # Set Out
        if out is None :
            if self.channel_wise_radii :
                C = radii.shape[0]
            else :
                C = np.max(types) + 1
            out = self.get_empty_grid(C, init_zero=True)
        else :
            out.fill(0.)

        # Set Atom Radii
        is_atom_wise_radii = (not np.isscalar(radii) and not self.channel_wise_radii)
        if self.channel_wise_radii :
            if isinstance(radii, float) :
                assert not np.isscalar(radii), \
                        'Radii type indexed requires type indexed radii (np.ndarray(shape=(C,)))' 
            atom_size = radii[types] * self.atom_scale          # (C, ) -> (V, )
        else :
            atom_size = radii * self.atom_scale

        # Clipping Overlapped Atoms
        box_overlap = self._get_overlap(coords, atom_size)
        coords, types = coords[box_overlap], types[box_overlap]
        radii = radii[box_overlap] if is_atom_wise_radii else radii

        # Run
        if self.num_blocks > 1 :
            blockdim = self.blockdim
            if self.channel_wise_radii :
                atom_size = radii[types] * self.atom_scale      # (C, ) -> (V, )
            else :
                atom_size = radii * self.atom_scale
            block_overlap_dict = self._get_overlap_blocks(coords, atom_size)

            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3) :
                start_x, end_x = xidx*blockdim, (xidx+1)*blockdim
                start_y, end_y = yidx*blockdim, (yidx+1)*blockdim
                start_z, end_z = zidx*blockdim, (zidx+1)*blockdim

                out_block = out[:, start_x:end_x, start_y:end_y, start_z:end_z]

                overlap = block_overlap_dict[(xidx, yidx, zidx)]
                if overlap.shape[0] == 0 :
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block, types_block = coords[overlap], types[overlap]
                radii_block = radii[overlap] if is_atom_wise_radii else radii
                self._set_grid_types(coords_block, types_block, radii_block, grid_block, out_block)
        else :
            grid = self.grid
            self._set_grid_types(coords, types, radii, grid, out)

        return out

    def _check_args_types(self, coords: NDArrayFloat, types: NDArrayInt, radii: Union[float,NDArrayFloat], 
                    out: Optional[NDArrayFloat] = None) :
        V = coords.shape[0]
        C = np.max(types) + 1
        D = H = W = self.dimension
        assert types.shape == (V,), f"types does not match dimension: {types.shape} vs {(V,)}"
        if not np.isscalar(radii) :
            if self.channel_wise_radii :
                assert radii.ndim == 1, f"radii does not match dimension: {radii.shape} vs {('Channel',)}"
                assert radii.shape == (C,), f'radii does not match dimension (number of types,): {radii.shape} vs {(C,)}'
            else :
                assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out is not None :
            assert out.shape[0] >= C, f'Output channel is less than number of types: {out.shape[0]} < {C}'
            assert out.shape[1:] == (D, H, W), f'Output grid dimension incorrect: {out.shape} vs {("*",D,H,W)}'

    def _set_grid_types(
        self,
        coords: NDArrayFloat,
        types: NDArrayInt,
        radii: Union[float, NDArrayFloat],
        grid: NDArrayFloat,
        out: NDArrayFloat,
    ) -> NDArrayFloat :
        """
        coords: (V, 3)
        types: (V,)
        radii: scalar or (V, )
        grid: (D,), (H,), (W,)

        out: (C, D, H, W)
        """
        if self._gaussian :
            if self.channel_wise_radii :
                func_types.gaussian_channel_wise_radii(coords, types, *grid, radii, self.atom_scale, out)
            elif np.isscalar(radii) :
                func_types.gaussian_scalar_radii(coords, types, *grid, radii, self.atom_scale, out)
            else :
                func_types.gaussian_atom_wise_radii(coords, types, *grid, radii, self.atom_scale, out)
        else :
            if self.channel_wise_radii :
                func_types.binary_channel_wise_radii(coords, types, *grid, radii, self.atom_scale, out)
            elif np.isscalar(radii) :
                func_types.binary_scalar_radii(coords, types, *grid, radii, self.atom_scale, out)
            else :
                func_types.binary_atom_wise_radii(coords, types, *grid, radii, self.atom_scale, out)
        return out

    """ COMMON BLOCK DIVISION """
    def _get_overlap(
        self,
        coords: NDArrayFloat,
        atom_size: Union[NDArrayFloat, float],
    ) -> NDArrayInt :
        if np.isscalar(atom_size):
            lb_overlap = np.greater(coords, self.lower_bound - atom_size).all(axis=-1)  # (V,)
            ub_overlap = np.less(coords, self.upper_bound + atom_size).all(axis=-1)     # (V,)
        else :
            atom_size = np.expand_dims(atom_size, 1)
            lb_overlap = np.greater(coords + atom_size, self.lower_bound).all(axis=-1)  # (V,)
            ub_overlap = np.less(coords - atom_size, self.upper_bound).all(axis=-1)     # (V,)
        overlap = np.logical_and(lb_overlap, ub_overlap)                                # (V,)
        return np.where(overlap)

    def _get_overlap_blocks(
        self,
        coords: NDArrayFloat,
        atom_size: Union[NDArray, float]
    ) -> Dict[Tuple[int, int, int], NDArrayInt] :

        def get_axis_overlap_list(coord_1d, atom_size) -> List[NDArrayBool]:
            overlaps = [None] * self.num_blocks
            for i in range(self.num_blocks) :
                if i == 0 :
                    upper = np.less(coord_1d, self.bounds[i] + atom_size)        # (V,)
                    overlaps[i] = upper
                elif i == self.num_blocks - 1 :
                    lower = np.greater(coord_1d, self.bounds[i-1] - atom_size)   # (V,)
                    overlaps[i] = lower
                else :
                    lower = np.greater(coord_1d, self.bounds[i-1] - atom_size)   # (V,)
                    upper = np.less(coord_1d, self.bounds[i] + atom_size)        # (V,)
                    overlaps[i] = np.logical_and(lower, upper)
            return overlaps
        
        overlap_dict = {key: None for key in self.grid_block_dict.keys()}
        x, y, z = np.split(coords, 3, axis=1)
        if not np.isscalar(atom_size) :
            atom_size = np.expand_dims(atom_size, 1)
        x_overlap_list = get_axis_overlap_list(x, atom_size)
        y_overlap_list = get_axis_overlap_list(y, atom_size)
        z_overlap_list = get_axis_overlap_list(z, atom_size)
        for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3) :
            x_overlap = x_overlap_list[xidx]
            y_overlap = y_overlap_list[yidx]
            z_overlap = z_overlap_list[zidx]
            overlap_dict[(xidx,yidx,zidx)] = np.where(x_overlap & y_overlap & z_overlap)[0]
        return overlap_dict

    @staticmethod
    def _dtypechange(array, dtype) :
        if array.dtype != dtype :
            return array.astype(dtype)
        else : 
            return array

    def asarray(self, array, obj) :
        if isinstance(array, np.ndarray) :
            if obj in ['coords', 'center', 'features', 'radii'] :
                return self._dtypechange(array, np.float32)
            elif obj == 'types' :
                return self._dtypechange(array, np.int16)
        else :
            if obj in ['coords', 'center', 'features', 'radii'] :
                return np.array(array, dtype=np.float32)
            elif obj == 'types' :
                return np.array(array, dtype=np.int16)
        raise ValueError("obj should be ['coords', center', 'types', 'features', 'radii']")

    @staticmethod
    def do_random_transform(coords, center, random_translation, random_rotation) :
        return do_random_transform(coords, center, random_translation, random_rotation)
