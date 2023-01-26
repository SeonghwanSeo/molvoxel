import math
import numpy as np
import itertools

from typing import Tuple, Union, Optional, Dict, List
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from .transform import do_random_transform

"""
radii: input value of molgrid
atom_radii: radii * radius_scale                            # scaled radii
atom_size: radii * radius_scale * final_radius_multiple     # atom point cloud boundary
"""

class GridMaker() :
    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 48,
        radius_scale: float = 1.0,
        gaussian_radius_multiple: float = 1.0,
        binary: bool = False,
        radii_type_indexed: bool = False,
        device: str = 'cpu',
        blockdim: int = None
    ) :
        self.resolution = resolution
        self.dimension = dimension
        self.width = width = resolution * (dimension - 1)
        self.radius_scale = radius_scale
        self.binary = binary
        self.radii_type_indexed = radii_type_indexed

        grm = gaussian_radius_multiple
        if grm < 0 :    # recommend 1.5 (libmolgrid)
            grm *= -1
            final_radius_multiple = grm
            self.mix_density = False
        else :
            final_radius_multiple = (1 + 2 * grm**2) / (2 * grm)
            self.mix_density = True # Mix Gaussian + Quadratic
        self.gaussian_radius_multiple = grm
        self.final_radius_multiple = final_radius_multiple

        self.A = math.exp(-2 * grm**2) * (4 * grm**2)                       # d^2/r^2
        self.B = -1 * math.exp(-2 * grm**2) * (4 * grm + 8 * grm**3)        # d/r
        self.C = math.exp(-2*grm**2)*(4*grm**4 + 4*grm**2 + 1)              # constant

        #self.D = 8 * grm**2 * math.exp(-2 * grm**2)                         # d/r^2
        #self.E = - (4 * grm + 8 * grm**3) * math.exp(-2 * grm**2)           # 1/r
        self.D = 2 * self.A
        self.E = self.B

        self.upper_bound: float = width / 2.
        self.lower_bound = -1 * self.upper_bound

        self._setup_block(blockdim)

    def cuda(self, update_blockdim = True, blockdim = None) :
        return self.to('cuda', update_blockdim, blockdim)

    def cpu(self, update_blockdim = True, blockdim = None) :
        return self.to('cpu', update_blockdim, blockdim)
            
    """ Attribute """
    def spatial_grid_dimension(self) -> Tuple[int, int, int] :
        return (self.dimension, self.dimension, self.dimension)
    def grid_dimension(self, ntypes: int) -> Tuple[int, int, int, int] :
        return (ntypes, self.dimension, self.dimension, self.dimension)
    def grid_width(self) -> int :
        return self.width

    def set_resolution(self, res: float) :
        self.resolution = res
    def get_resolution(self) -> float :
        return self.resolution

    def set_dimension(self, d: int) :
        self.dimension = d
    def get_dimension(self) -> int :
        return self.dimension

    def set_radii_type_indexed(self, b: bool) :
        self.radii_type_indexed = b
    def get_radii_type_indexed(self) -> bool:
        return self.radii_type_indexed

    def set_binary(self, b: bool) :
        self.binary = b
    def get_binary(self) -> bool:
        return self.binary

    def forward(
        self,
        coords: NDArray[np.float_],
        center: NDArray[np.float_],
        types: Union[NDArray[np.float_], NDArray[np.int_]],
        radii: Union[float, NDArray[np.float_]],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[NDArray[np.float_]] = None
    ) -> NDArray[np.float_] :
        """
        coords: (V, 3)
        center: (3,)
        types: (V, C) or (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out: (C,D,H,W)
        """
        if types.ndim == 1 :
            type_index = types
            return self.forward_index(coords, center, type_index, radii, random_translation, random_rotation, out)
        else :
            type_vector = types
            return self.forward_vector(coords, center, type_vector, radii, random_translation, random_rotation, out)

    __call__ = forward

    """ VECTOR """
    def forward_vector(
        self,
        coords: NDArray[np.float_],
        center: NDArray[np.float_],
        type_vector: NDArray[np.float_],
        radii: Union[float, NDArray[np.float_]],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[NDArray[np.float_]] = None
    ) -> NDArray[np.float_] :
        """
        coords: (V, 3)
        center: (3,)
        type_vector: (V, C)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out: (C,D,H,W)
        """
        self._check_vector_args(coords, type_vector, radii, out)
        C = type_vector.shape[1]
        D = H = W = self.dimension

        if out is None :
            out = np.empty((C, D, H, W))

        coords = coords - center.reshape(1,3)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        atom_radii = radii * self.radius_scale
        if self.radii_type_indexed :
            atom_size = atom_radii.max() * self.final_radius_multiple
        else :
            atom_size = atom_radii * self.final_radius_multiple

        # Clipping Overlapped Atoms
        box_overlap = self._get_overlap(coords, atom_size)
        coords, type_vector = coords[box_overlap], type_vector[box_overlap]
        if not np.isscalar(atom_radii) and not self.radii_type_indexed :
            atom_radii = atom_radii[box_overlap]
            atom_size = atom_radii * self.final_radius_multiple

        if self.num_blocks > 1 :
            # Run
            blockdim = self.blockdim
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
                coords_block, type_vector_block = coords[overlap], type_vector[overlap]
                atom_radii_block = atom_radii[overlap] if not self.radii_type_indexed and not np.isscalar(atom_radii) \
                                                    else atom_radii
                self._set_atoms_vector(coords_block, type_vector_block, atom_radii_block, grid_block, out_block)
        else :
            grid = self.grid
            self._set_atoms_vector(coords, type_vector, atom_radii, grid, out)
             
        if self.binary :
            np.clip(out, a_min=None, a_max=1., out=out)
        
        return out

    def _check_vector_args(self, coords: NDArray[np.float_], type_vector: NDArray[np.float_], radii: Union[float,NDArray[np.float_]], 
                    out: Optional[NDArray[np.float_]] = None) :
        V = coords.shape[0]
        C = type_vector.shape[1]
        D = H = W = self.dimension
        assert type_vector.shape[0] == V, f'type vector does not match number of atoms: {type_vector.shape[0]} vs {V}'
        assert type_vector.ndim == 2, f"type vector does not match dimension: {type_vector.shape} vs {(V,'*')}"
        if self.radii_type_indexed :
            assert not np.isscalar(radii)
            assert radii.shape == (C,), f'radii does not match dimension (number of types,): {radii.shape} vs {(C,)}'
        else :
            if not np.isscalar(radii) :
                assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out is not None :
            assert out.shape == (C, D, H, W), f'Output grid dimension incorrect: {out.shape} vs {(C,D,H,W)}'

    def _set_atoms_vector(
        self,
        coords: NDArray[np.float_],
        type_vector: NDArray[np.float_],
        atom_radii: Union[float, NDArray[np.float_]],
        grid: NDArray[np.float_],
        out: NDArray[np.float_],
    ) -> NDArray[np.float_] :
        """
        coords: (V, 3)
        type_vector: (V, C)
        atom_radii: scalar or (V, ) or (C, )
        grid: (D, H, W, 3)

        out: (C, D, H, W)
        """
        type_vector = type_vector.T                                             # (V, C) -> (C, V)
        D, H, W, _ = grid.shape
        grid = grid.reshape(-1, 3)                                              # (D*H*W, 3)
        if out.data.contiguous :
            _out = out.reshape(-1, D*H*W)
            if self.radii_type_indexed :
                for type_idx in range(type_vector.shape[0]) :
                    typ = type_vector[type_idx]                                 # (V,)
                    res = self._calc_point(coords, atom_radii[type_idx], grid)  # (V, D*H*W)
                    np.matmul(typ, res, out=_out[type_idx])
            else :
                res = self._calc_point(coords, atom_radii, grid)                # (V, D*H*W)
                np.matmul(type_vector, res, out=_out)                           # (C, D*H*W)
        else :
            if self.radii_type_indexed :
                for type_idx in range(type_vector.shape[0]) :
                    typ = type_vector[type_idx]                                 # (V,)
                    res = self._calc_point(coords, atom_radii[type_idx], grid)  # (V, D*H*W)
                    out[type_idx] = np.matmul(typ, res).reshape(D, H, W)        # (D, H, W)
            else :
                res = self._calc_point(coords, atom_radii, grid)                # (V, D*H*W)
                out[:] = np.matmul(type_vector, res).reshape(-1, D, H, W)       # (C, D, H, W)
        return out

    """ INDEX """
    def forward_index(
        self,
        coords: NDArray[np.float_],
        center: NDArray[np.float_],
        type_index: NDArray[np.float_],
        radii: Union[float, NDArray[np.float_]],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[NDArray[np.float_]] = None
    ) -> NDArray[np.float_] :
        """
        coords: (V, 3)
        center: (3,)
        type_index: (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out: (C,D,H,W)
        """
        self._check_index_args(coords, type_index, radii, out)
        if self.radii_type_indexed :
            C = radii.shape[0]
        else :
            C = np.max(type_index) + 1
        D = H = W = self.dimension
        
        if out is None :
            out = np.zeros((C, D, H, W))
        else :
            out.fill(0.)

        coords = coords - center.reshape(1, 3)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        if self.radii_type_indexed :
            if isinstance(radii, float) :
                assert not np.isscalar(radii), \
                        'Radii type indexed requires type indexed radii (np.ndarray(shape=(C,)))' 
            radii = radii[type_index]           # (C, ) -> (V, )

        atom_radii = radii * self.radius_scale
        atom_size = atom_radii * self.final_radius_multiple

        # Clipping Overlapped Atoms
        box_overlap = self._get_overlap(coords, atom_size)
        coords, type_index = coords[box_overlap], type_index[box_overlap]
        atom_radii = atom_radii[box_overlap] if not np.isscalar(atom_radii) else atom_radii
        atom_size = atom_radii * self.final_radius_multiple
        block_overlap_dict = self._get_overlap_blocks(coords, atom_size)

        # Run
        blockdim = self.blockdim
        for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3) :
            start_x, end_x = xidx*blockdim, (xidx+1)*blockdim
            start_y, end_y = yidx*blockdim, (yidx+1)*blockdim
            start_z, end_z = zidx*blockdim, (zidx+1)*blockdim

            overlap = block_overlap_dict[(xidx, yidx, zidx)]
            if overlap.shape[0] == 0 :
                continue

            grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
            coords_block, type_index_block = coords[overlap], type_index[overlap]
            atom_radii_block = atom_radii[overlap] if not np.isscalar(atom_radii) else atom_radii
            out_block = out[:, start_x:end_x, start_y:end_y, start_z:end_z]
            self._set_atoms_index(coords_block, type_index_block, atom_radii_block, grid_block, out_block)

        if self.binary :
            np.clip(out, a_min=None, a_max=1., out=out)

        return out

    def _check_index_args(self, coords: NDArray[np.float_], type_index: NDArray[np.float_], radii: Union[float,NDArray[np.float_]], 
                    out: Optional[NDArray[np.float_]] = None) :
        V = coords.shape[0]
        C = np.max(type_index) + 1
        D = H = W = self.dimension
        assert type_index.ndim == 1, f"type index does not match dimension: {type_vector.shape[0]} vs {(V,)}"
        assert type_index.shape[0] == V, f'type index does not match number of atoms: {type_index.shape[0]} vs {V}'
        if not np.isscalar(radii) :
            if self.radii_type_indexed :
                assert radii.ndim == 1, f"radii does not match dimension: {radii.shape} vs {('Channel',)}"
                assert radii.shape == (C,), f'radii does not match dimension (number of types,): {radii.shape} vs {(C,)}'
            else :
                assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out is not None :
            assert out.shape[0] >= C, f'Output channel is less than number of types: {out.shape[0]} < {C}'
            assert out.shape[1:] == (D, H, W), f'Output grid dimension incorrect: {out.shape} vs {("*",D,H,W)}'

    def _set_atoms_index(
        self,
        coords: NDArray[np.float_],
        type_index: NDArray[np.float_],
        atom_radii: Union[float, NDArray[np.float_]],
        grid: NDArray[np.float_],
        out: NDArray[np.float_],
    ) -> NDArray[np.float_] :
        """
        coords: (V, 3)
        type_index: (V,)
        atom_radii: scalar or (V, )
        grid: (D, H, W, 3)

        out: (C, D, H, W)
        """
        D, H, W, _ = grid.shape
        grid = grid.reshape(-1, 3)
        res = self._calc_point(coords, atom_radii, grid)    # (V, D*H*W)
        res = res.reshape(-1, D, H, W)                      # (V, D, H, W)
        for vidx, typ in enumerate(type_index) :
            out[typ] += res[vidx] 
        return out

    """ COMMON - BLOCK """
    def _setup_block(self, blockdim) :
        blockdim = blockdim if blockdim is not None else 8
        self.blockdim = blockdim

        axis = np.arange(self.dimension, dtype=np.float_) * self.resolution - (self.width/2.)
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

    def _get_overlap(
        self,
        coords: NDArray[np.float_],
        atom_size: Union[NDArray, float],
    ) -> NDArray[np.int_] :
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
        coords: NDArray[np.float_],
        atom_size: Union[NDArray, float]
    ) -> Dict[Tuple[int, int, int], NDArray[np.int64]] :

        def get_axis_overlap_list(coord_1d, atom_size) -> List[NDArray[np.bool_]]:
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
    def _calc_point(
        self,
        coords: NDArray[np.float_],
        atom_radii: Union[float, NDArray[np.float_]],
        grid: NDArray[np.float_],
    ) -> NDArray[np.float_] :
        """
        coords: (V, 3)
        atom_radii: scalar or (V, )
        grid: (D*H*W, 3)

        out: (V, D*H*W)
        """
        dist = cdist(coords, grid)                    # (V, D, H, W)
        if not np.isscalar(atom_radii) :
            atom_radii = np.expand_dims(atom_radii, -1)
        dr = np.divide(dist, atom_radii)
        if self.binary :
            return self.__calc_grid_density_binary(dr)
        elif self.mix_density :
            return self.__calc_grid_density_mix(dr)
        else :
            return self.__calc_grid_density_gaussian(dr)

    def __calc_grid_density_binary(self, dr: NDArray) -> NDArray :
        return np.less(dr, 1.0, dr)

    def __calc_grid_density_gaussian(self, dr: NDArray) -> NDArray :
        mask = np.greater(dr, self.final_radius_multiple)
        out = np.exp(np.multiply(np.power(dr, 2, dr), -2, dr), dr)
        out[mask] = 0
        return out

    def __calc_grid_density_mix(self, dr: NDArray) -> NDArray :
        mask_gaus = np.greater(dr, self.gaussian_radius_multiple)
        mask_final = np.greater(dr, self.final_radius_multiple)
        drsquare = np.power(dr, 2)
        
        gaus = np.exp(-2.0 * drsquare)
        quad = self.A * drsquare + self.B * dr + self.C
        out = np.where(mask_gaus, gaus, quad)
        out[mask_final] = 0.
        return out
    
