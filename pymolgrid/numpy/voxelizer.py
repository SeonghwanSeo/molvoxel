import math
import numpy as np
import itertools

from typing import Tuple, Union, Optional, Dict, List
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from pymolgrid.base import BaseVoxelizer
from .transform import do_random_transform

NDArrayInt = NDArray[np.int_]
NDArrayFloat = NDArray[np.float_]
NDArrayBool = NDArray[np.bool_]

"""
radii: input value of pymolgrid
atom_size: radii * atom_scale           # atom boundary
"""

class Voxelizer(BaseVoxelizer) :
    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 48,
        atom_scale: float = 1.5,
        density: str = 'gaussian',
        channel_wise_radii: bool = False,
        precision: int = 32,
        blockdim: Optional[int] = None
    ) :
        super(Voxelizer, self).__init__(resolution, dimension, atom_scale, channel_wise_radii)
        assert density in ['gaussian', 'binary']
        self._density = density.lower()
        self._gaussian = (self._density == 'gaussian')
        assert precision in [32, 64]
        self.fp = np.float32 if precision == 32 else np.float64
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

        axis = np.arange(self.dimension, dtype=np.float64) * self.resolution - (self.width/2.)  # cdist only support float64
        self.num_blocks = num_blocks = math.ceil(self.dimension / blockdim)
        if self.num_blocks > 1 :
            self.grid = None
            self.grid_block_dict = {}
            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3) :
                x_axis = axis[xidx*blockdim : (xidx+1)*blockdim]
                y_axis = axis[yidx*blockdim : (yidx+1)*blockdim]
                z_axis = axis[zidx*blockdim : (zidx+1)*blockdim]
                grid_block = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
                self.grid_block_dict[(xidx, yidx, zidx)] = np.stack(grid_block, axis=-1)

            self.bounds = [(axis[idx*blockdim] + (self.resolution / 2.))
                                                for idx in range(1, num_blocks)]
        else :
            self.grid_block_dict = None
            self.grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)

    def get_empty_grid(self, num_channels: int, batch_size: Optional[int] = None, init_zero: bool = False) -> NDArrayFloat:
        if init_zero :
            if batch_size is None :
                return np.zeros(self.grid_dimension(num_channels), dtype=self.fp)
            else :
                return np.zeros((batch_size,) + self.grid_dimension(num_channels), dtype=self.fp)
        else :
            if batch_size is None :
                return np.empty(self.grid_dimension(num_channels), dtype=self.fp)
            else :
                return np.empty((batch_size,) + self.grid_dimension(num_channels), dtype=self.fp)

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
            return self.forward_type(coords, center, types, radii, random_translation, random_rotation, out)
        else :
            features = channels
            return self.forward_feature(coords, center, features, radii, random_translation, random_rotation, out)

    __call__ = forward

    """ VECTOR """
    def forward_feature(
        self,
        coords: NDArrayFloat,
        center: NDArrayFloat,
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
        self._check_args_feature(coords, features, radii, out)

        # Set Coordinate
        coords = coords - center.reshape(1,3)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        # DataType
        if coords.dtype != np.float64 :     # cdist support only float64
            coords = coords.astype(np.float64)
        if features.dtype != self.fp :
            features = features.astype(self.fp)
        if not np.isscalar(radii) and radii.dtype != self.fp :
            radii = radii.astype(self.fp)

        # Set Out
        if out is None :
            C = features.shape[1]
            out = self.get_empty_grid(C)

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
                    out_block.fill(0.)
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block, features_block = coords[overlap], features[overlap]
                radii_block = radii[overlap] if is_atom_wise_radii else radii
                self._set_grid_feature(coords_block, features_block, radii_block, grid_block, out_block)
        else :
            grid = self.grid
            self._set_grid_feature(coords, features, radii, grid, out)
        
        return out

    def _check_args_feature(self, coords: NDArrayFloat, features: NDArrayFloat, radii: Union[float,NDArrayFloat], 
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

    def _set_grid_feature(
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
        grid: (D, H, W, 3)

        out: (C, D, H, W)
        """
        features = features.T                                             # (V, C) -> (C, V)
        D, H, W, _ = grid.shape
        grid = grid.reshape(-1, 3)                                                 # (D*H*W, 3)
        if out.data.contiguous :
            _out = out.reshape(-1, D*H*W)
            if self.channel_wise_radii :
                for type_idx in range(features.shape[0]) :
                    typ = features[type_idx]                                 # (V,)
                    res = self._calc_grid(coords, radii[type_idx], grid)       # (V, D*H*W)
                    np.matmul(typ, res, out=_out[type_idx])
            else :
                res = self._calc_grid(coords, radii, grid)                     # (V, D*H*W)
                np.matmul(features, res, out=_out)                           # (C, D*H*W)
        else :
            if self.channel_wise_radii :
                for type_idx in range(features.shape[0]) :
                    typ = features[type_idx]                                 # (V,)
                    res = self._calc_grid(coords, radii[type_idx], grid)  # (V, D*H*W)
                    out[type_idx] = np.matmul(typ, res).reshape(D, H, W)        # (D, H, W)
            else :
                res = self._calc_grid(coords, radii, grid)                # (V, D*H*W)
                out[:] = np.matmul(features, res).reshape(-1, D, H, W)       # (C, D, H, W)
        return out

    """ INDEX """
    def forward_type(
        self,
        coords: NDArrayFloat,
        center: NDArrayFloat,
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
        self._check_args_type(coords, types, radii, out)

        # Set Coordinate
        coords = coords - center.reshape(1, 3)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        # DataType
        if coords.dtype != np.float64 :     # cdist support only float64
            coords = coords.astype(np.float64)
        if not np.isscalar(radii) and radii.dtype != self.fp :
            radii = radii.astype(self.fp)

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
        if self.channel_wise_radii :
            if isinstance(radii, float) :
                assert not np.isscalar(radii), \
                        'Radii type indexed requires type indexed radii (np.ndarray(shape=(C,)))' 
            radii = radii[types]           # (C, ) -> (V, )
        atom_size = radii * self.atom_scale

        # Clipping Overlapped Atoms
        box_overlap = self._get_overlap(coords, atom_size)
        coords, types = coords[box_overlap], types[box_overlap]
        radii = radii[box_overlap] if not np.isscalar(radii) else radii

        # Run
        if self.num_blocks > 1 :
            blockdim = self.blockdim
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
                radii_block = radii[overlap] if not np.isscalar(radii) else radii
                self._set_grid_type(coords_block, types_block, radii_block, grid_block, out_block)
        else :
            grid = self.grid
            self._set_grid_type(coords, types, radii, grid, out)

        return out

    def _check_args_type(self, coords: NDArrayFloat, types: NDArrayFloat, radii: Union[float,NDArrayFloat], 
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

    def _set_grid_type(
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
        grid: (D, H, W, 3)

        out: (C, D, H, W)
        """
        D, H, W, _ = grid.shape
        grid = grid.reshape(-1, 3)
        res = self._calc_grid(coords, radii, grid)         # (V, D*H*W)
        res = res.reshape(-1, D, H, W)                      # (V, D, H, W)
        for vidx, typ in enumerate(types) :
            out[typ] += res[vidx] 
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

    """ COMMON - GRID CALCULATION """
    def _calc_grid(
        self,
        coords: NDArrayFloat,
        radii: Union[float, NDArrayFloat],
        grid: NDArrayFloat,
    ) -> NDArrayFloat :
        """
        coords: (V, 3)
        radii: scalar or (V, )
        grid: (D*H*W, 3)

        out: (V, D*H*W)
        """
        dist = cdist(coords, grid)              # (V, D, H, W)
        if self.fp is np.float32 :
            dist = dist.astype(self.fp)         # np.float64 -> np.float32
        if not np.isscalar(radii) :
            radii = np.expand_dims(radii, -1)
        dr = np.divide(dist, radii)
        if self._gaussian :
            return self.__calc_grid_density_gaussian(dr)
        else :
            return self.__calc_grid_density_binary(dr)

    def __calc_grid_density_binary(self, dr: NDArrayFloat) -> NDArrayFloat :
        return np.less(dr, self.atom_scale, dr)

    def __calc_grid_density_gaussian(self, dr: NDArrayFloat) -> NDArrayFloat :
        out = np.exp(np.multiply(np.power(dr, 2, dr), -2, dr), dr)
        out[dr > self.atom_scale] = 0
        return out
