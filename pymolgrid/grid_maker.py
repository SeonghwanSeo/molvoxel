import math
import torch
import torch.nn.functional as F
import itertools

from torch import Tensor, FloatTensor, LongTensor
from typing import Tuple, Union, Optional

from .transform import Transform

BLOCKDIM=8

"""
radii: input value of molgrid
atom_radii: radii * radius_scale                            # scaled radii
atom_size: radii * radius_scale * final_radius_multiple     # atom point cloud boundary
"""

def _get_overlap(
    coords: FloatTensor,                        # (V, 3)
    atom_size: Union[float, FloatTensor],       # (V,) or scalar
    lower_bound: Union[float, FloatTensor],     # (1, 3) or scalar
    upper_bound: Union[float, FloatTensor],     # (1, 3) or scalar
) -> LongTensor:
    device = coords.device
    if isinstance(lower_bound, Tensor) :
        lower_bound.to(device)
    if isinstance(upper_bound, Tensor) :
        upper_bound.to(device)
    if isinstance(atom_size, Tensor) and atom_size.dim() == 1:
        atom_size = atom_size.unsqueeze(1)
    lb_mask = torch.less(coords + atom_size, lower_bound).sum(dim=-1)       # (V,)
    ub_mask = torch.greater(coords - atom_size, upper_bound).sum(dim=-1)    # (V,)
    mask = lb_mask.add_(ub_mask)                                            # (V,)
    return torch.where(mask==0)

class _GridBox() :
    def __init__(self, x_axis: FloatTensor, y_axis: FloatTensor, z_axis: FloatTensor) :
        self.grid = grid = torch.stack(torch.meshgrid([x_axis, y_axis, z_axis], indexing='ij'), dim=-1)
                                                                        # (BLOCKDIM, BLOCKDIM, BLOCKDIM, 3)
        self.lower_bound = torch.clone(grid[0,0,0]).unsqueeze(0)        # (1, 3)
        self.upper_bound = torch.clone(grid[-1,-1,-1]).unsqueeze(0)     # (1, 3)

    def get_overlap(
        self,
        coords: FloatTensor,
        atom_size: Union[float, FloatTensor],
    ) -> LongTensor :
        device = coords.device
        return _get_overlap(coords, atom_size, self.lower_bound, self.upper_bound)

    def spatial_grid_dimension(self) -> Tuple[int, int, int] :
        return tuple(self.box.size())

class GridMaker() :
    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 48,
        radius_scale: float = 1.0,
        gaussian_radius_multiple: float = 1.0,
        binary: bool = False,
        radii_type_indexed: bool = False,
    ) :
        self.resolution = resolution
        self.dimension = dimension
        self.width = width = resolution * (dimension - 1)
        self.radius_scale = radius_scale

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

        self.A = math.exp(-2 * grm**2) * (4 * grm**2)                   # d^2/r^2
        self.B = -1 * math.exp(-2 * grm**2) * (4 * grm + 8 * grm**3)    # d/r
        self.C = math.exp(-2*grm**2)*(4*grm**4 + 4*grm**2 + 1)          # constant

        #self.D = 8 * grm**2 * math.exp(-2 * grm**2)                     # d/r^2
        #self.E = - (4 * grm + 8 * grm**3) * math.exp(-2 * grm**2)       # 1/r
        self.D = 2 * self.A
        self.E = self.B

        self.radii_type_indexed = radii_type_indexed
        self.binary = binary

        self.upper_bound: float = width / 2.
        self.lower_bound = -1 * self.upper_bound

        self.num_blocks = num_blocks = math.ceil(dimension / BLOCKDIM)
        self.grid_box_dict = {}
        axis = torch.arange(dimension, dtype=torch.float) * resolution - (width / 2.)
        for xidx in range(num_blocks) :
            x_axis = axis[xidx*BLOCKDIM : (xidx+1)*BLOCKDIM]
            for yidx in range(num_blocks) :
                y_axis = axis[yidx*BLOCKDIM : (yidx+1)*BLOCKDIM]
                for zidx in range(num_blocks) :
                    z_axis = axis[zidx*BLOCKDIM : (zidx+1)*BLOCKDIM]
                    self.grid_box_dict[(xidx, yidx, zidx)] = _GridBox(x_axis, y_axis, z_axis)

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

    # Vector
    def forward_vector(
        self,
        coords: FloatTensor,
        center: FloatTensor,
        type_vector: FloatTensor,
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[FloatTensor] = None
    ) -> FloatTensor :
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
        C = type_vector.size(1)
        D = H = W = self.dimension
        
        if out is None :
            out = torch.empty((C, D, H, W), device=coords.device)

        coords = coords - center.unsqueeze(0)
        coords = self.do_transform(coords, None, random_translation, random_rotation)

        atom_radii = radii * self.radius_scale
        if self.radii_type_indexed :
            atom_size = atom_radii.max().item() * self.final_radius_multiple
        else :
            atom_size = atom_radii * self.final_radius_multiple

        # Clipping Overlapped Atoms
        mask = _get_overlap(coords, atom_size, self.lower_bound, self.upper_bound)
        coords, type_vector = coords[mask], type_vector[mask]
        if isinstance(atom_radii, Tensor) and not self.radii_type_indexed:
            atom_radii = atom_radii[mask]
            atom_size = atom_size[mask]
        
        # Run
        for xidx in range(self.num_blocks) :
            start_x = xidx * BLOCKDIM
            end_x = start_x + BLOCKDIM
            for yidx in range(self.num_blocks) :
                start_y = yidx * BLOCKDIM
                end_y = start_y + BLOCKDIM
                for zidx in range(self.num_blocks) :
                    start_z = zidx * BLOCKDIM
                    end_z = start_z + BLOCKDIM
                    out_box = out[:, start_x:end_x, start_y:end_y, start_z:end_z]
                    grid_box = self.grid_box_dict[(xidx, yidx, zidx)]
                    self._set_atoms_vector(coords, type_vector, atom_radii, atom_size, grid_box, out_box)

        return out

    def _check_vector_args(self, coords: FloatTensor, type_vector: FloatTensor, radii: Union[float,FloatTensor], 
                    out: Optional[FloatTensor] = None) :
        N = coords.size(0)
        C = type_vector.size(1)
        D = H = W = self.dimension
        assert isinstance(type_vector, FloatTensor), f'type vector should be FloatTensor, dtype: {type_vector.dtype}'
        assert type_vector.size(0) == N, f'type vector does not match number of atoms: {type_vector.size(0)} vs {N}'
        assert type_vector.dim() == 2, f"type vector does not match dimension: {tuple(type_vector.size())} vs {(N,'*')}"
        if self.radii_type_indexed :
            assert isinstance(radii, Tensor)
            assert radii.size() == (C,), f'radii does not match dimension (number of types,): {tuple(radii.size())} vs {(C,)}'
        else :
            if isinstance(radii, Tensor) :
                assert radii.size() == (N,), f'radii does not match dimension (number of atoms,): {tuple(radii.size())} vs {(N,)}'
        if out is not None :
            assert out.size() == (C, D, H, W), f'Output grid dimension incorrect: {tuple(out.size())} vs {(C,D,H,W)}'

    def _set_atoms_vector(
        self,
        coords: FloatTensor,
        type_vector: FloatTensor,
        atom_radii: Union[float, FloatTensor],
        atom_size: Union[float, FloatTensor],
        grid_box: _GridBox,
        out_box: FloatTensor,
    ) -> FloatTensor :
        """
        coords: (V, 3)
        type_vector: (V, C)
        atom_radii: scalar or (V, ) or (C, )
        atom_size: scalar or (V, )
        grid_box:
            grid: (D, H, W, 3)
            lower_bound: (1, 3)
            upper_bound: (1, 3)

        out_box: (C, D, H, W)
        """
        D, H, W = grid_box.grid.size()[:-1]
        device = coords.device

        mask = grid_box.get_overlap(coords, atom_size)
        coords, type_vector = coords[mask], type_vector[mask]
        if not self.radii_type_indexed :
            atom_radii = atom_radii[mask] if isinstance(atom_radii, Tensor) else atom_radii

        if self.radii_type_indexed :
            for type_idx in range(type_vector.size(1)) :
                _type = type_vector[:,type_idx]                         # (V,)
                _out = self._calc_point(coords, atom_radii[type_idx], grid_box.grid)  # (D, H, W, V)
                torch.matmul(_out, _type, out = out_box[type_idx])      # (D, H, W, V) * (V,) -> (D, H, W)
        else :
            _out = self._calc_point(coords, atom_radii, grid_box.grid)  # (D, H, W, V)
            _out = torch.matmul(_out, type_vector)                      # (D, H, W, C)
            out_box[:] = _out.permute(3, 0, 1, 2)                       # (C, D, H, W)
        if self.binary :
            out_box.clip_(max=1.)
        return out_box

    def forward_index(
        self,
        coords: FloatTensor,
        center: FloatTensor,
        type_index: FloatTensor,
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[FloatTensor] = None
    ) -> FloatTensor :
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
            C = radii.size(0)
        else :
            C = torch.max(type_index).item() + 1
        D = H = W = self.dimension
        
        if out is None :
            out = torch.zeros((C, D, H, W), device=coords.device)
        else :
            out.fill_(0.)

        coords = coords - center.unsqueeze(0)
        coords = self.do_transform(coords, None, random_translation, random_rotation)

        if self.radii_type_indexed :
            if isinstance(radii, float) :
                assert isinstance(radii, Tensor), \
                        'Radii type indexed requires type indexed radii (torch.FloatTensor(size=(C,)))' 
            radii = radii[type_index]           # (C, ) -> (V, )

        atom_radii = radii * self.radius_scale
        atom_size = atom_radii * self.final_radius_multiple

        # Clipping Overlapped Atoms
        mask = _get_overlap(coords, atom_size, self.lower_bound, self.upper_bound)
        coords, type_index = coords[mask], type_index[mask]
        if isinstance(atom_radii, Tensor) :
            atom_radii, atom_size = atom_radii[mask], atom_size[mask]

        # Run
        for xidx in range(self.num_blocks) :
            start_x = xidx*BLOCKDIM
            end_x = start_x + BLOCKDIM
            for yidx in range(self.num_blocks) :
                start_y = yidx*BLOCKDIM
                end_y = start_y + BLOCKDIM
                for zidx in range(self.num_blocks) :
                    start_z = zidx*BLOCKDIM
                    end_z = start_z + BLOCKDIM
                    out_box = out[:, start_x:end_x, start_y:end_y, start_z:end_z]
                    grid_box = self.grid_box_dict[(xidx, yidx, zidx)]
                    self._set_atoms_index(coords, type_index, atom_radii, atom_size, grid_box, out_box)

        return out

    def _check_index_args(self, coords: FloatTensor, type_index: FloatTensor, radii: Union[float,FloatTensor], 
                    out: Optional[FloatTensor] = None) :
        N = coords.size(0)
        C = torch.max(type_index).item() + 1
        D = H = W = self.dimension
        assert isinstance(type_index, LongTensor), f'type index should be LongTensor, dtype: {type_index.dtype}'
        assert type_index.dim() == 1, f"type index does not match dimension: {tuple(type_vector.size())} vs {(N,)}"
        assert type_index.size(0) == N, f'type index does not match number of atoms: {type_index.size(0)} vs {N}'
        if isinstance(radii, Tensor) :
            if self.radii_type_indexed :
                assert radii.dim() == 1, f"radii does not match dimension: {tuple(radii.size())} vs {('Channel',)}"
                assert radii.size() == (C,), f'radii does not match dimension (number of types,): {tuple(radii.size())} vs {(C,)}'
            else :
                assert radii.size() == (N,), f'radii does not match dimension (number of atoms,): {tuple(radii.size())} vs {(N,)}'
        if out is not None :
            assert out.size(0) >= C, f'Output channel is less than number of types: {out.size(0)} < {C}'
            assert out.size()[1:] == (D, H, W), f'Output grid dimension incorrect: {tuple(out.size())} vs {("*",D,H,W)}'

    def _set_atoms_index(
        self,
        coords: FloatTensor,
        type_index: FloatTensor,
        atom_radii: Union[float, FloatTensor],
        atom_size: Union[float, FloatTensor],
        grid_box: _GridBox,
        out_box: FloatTensor,
    ) -> FloatTensor :
        """
        coords: (V, 3)
        type_index: (V,)
        atom_radii: scalar or (V, )
        atom_size: scalar or (V, )
        grid_box:
            grid: (D, H, W, 3)
            lower_bound: (1, 3)
            upper_bound: (1, 3)

        out_box: (C, D, H, W)
        """
        D, H, W = grid_box.grid.size()[:-1]
        device = coords.device

        mask = grid_box.get_overlap(coords, atom_size)
        coords, type_index = coords[mask], type_index[mask]
        atom_radii = atom_radii[mask] if isinstance(atom_radii, Tensor) else atom_radii
        _out = self._calc_point(coords, atom_radii, grid_box.grid)   # (D, H, W, V)
        for idx, typ in enumerate(type_index) :
            out_box[typ].add_(_out[:,:,:,idx])                           # (C, D, H, W)
        if self.binary :
            out_box.clip_(max=1.)
        return out_box


    @staticmethod
    def do_transform(coords, center, random_translation, random_rotation) -> FloatTensor:
        return Transform.do_transform(coords, center, random_translation, random_rotation)
    
    def _calc_point(
        self,
        coords: FloatTensor,
        atom_radii: Union[float, FloatTensor],
        grid: FloatTensor,
    ) -> FloatTensor :
        """
        coords: (V, 3)
        radii: scalar or (V, )
        grid: (D, H, W, 3)
        """
        if self.binary :
            return self.__calc_grid_density_binary(coords, atom_radii, grid)
        elif self.mix_density :
            return self.__calc_grid_density_mix(coords, atom_radii, grid)
        else :
            return self.__calc_grid_density_gaussian(coords, atom_radii, grid)

    def __calc_grid_density_binary(
        self,
        coords: FloatTensor,
        atom_radii: Union[float, FloatTensor],
        grid: FloatTensor,
    ) -> FloatTensor :
        dist = torch.cdist(grid, coords)   # (D*H*W, V)
        return torch.less(dist, atom_radii).float()

    def __calc_grid_density_mix(
        self,
        coords: FloatTensor,
        atom_radii: Union[float, FloatTensor],
        grid: FloatTensor,
    ) -> FloatTensor :
        dist = torch.cdist(grid, coords)   # (D*H*W, V)
        if isinstance(atom_radii, float) :
            dr = dist / atom_radii                                  # (D*H*W, V)
        else :
            dr = dist / atom_radii.unsqueeze(0)                     # (D*H*W, V)
        mask1 = torch.greater(dr, self.final_radius_multiple)       # (D*H*W, V)
        mask2 = torch.greater(dr, self.gaussian_radius_multiple)    # (D*H*W, V)
        drsquare = torch.pow(dr, 2)
        
        gaus = torch.exp(-2.0 * drsquare)
        quad = self.A * drsquare + self.B * dr + self.C
        out = torch.where(mask2, gaus, quad)
        out = torch.where(mask1, 0, out)
        return out

    def __calc_grid_density_gaussian(
        self,
        coords: FloatTensor,
        atom_radii: Union[float, FloatTensor],
        grid: FloatTensor,
    ) -> FloatTensor :
        dist = torch.cdist(grid, coords)                            # (D*H*W, V)
        if isinstance(atom_radii, float) :
            dr = dist / atom_radii                                  # (D*H*W, V)
        else :
            dr = dist / atom_radii.unsqueeze(0)                     # (D*H*W, V)
        mask = torch.greater(dr, self.final_radius_multiple)        # (D*H*W, V)
        out = torch.exp(-2.0*torch.pow(dr, 2))
        out.masked_fill_(mask, 0)
        return out

    def __calc_scalar_density(self, distance: float, atom_radius: float) -> float:
        if self.binary :
            return 1.0 if distance < radius else 0.0
        else :
            dr = distance / radius
            if dr > self.final_radius_multiple :
                return 0.0
            elif dr <= gaussian_radius_multiple :
                return math.exp(-2.0 * dr ** 2)
            else :  # quadratic
                q = (self.A * dr + self.B) * dr + self.C
                return max(q, 0)
