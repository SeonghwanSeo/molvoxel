import math
import torch
import numpy as np
import itertools

from torch import Tensor, FloatTensor, LongTensor, BoolTensor
from typing import Tuple, Union, Optional, Dict, List, Callable

from pymolgrid.base import BaseVoxelizer
from .transform import do_random_transform

"""
radii: input value of pymolgrid
atom_size: radii * atom_scale           # atom boundary
"""

class Voxelizer(BaseVoxelizer) :
    LIB='PyTorch'
    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 64,
        atom_scale: float = 1.5,
        density: str = 'gaussian',
        channel_wise_radii: bool = False,
        device: str = 'cpu',
        blockdim: Optional[int] = None
    ) :
        super(Voxelizer, self).__init__(resolution, dimension, atom_scale, channel_wise_radii)
        assert density in ['gaussian', 'binary']
        self._density = density.lower()
        self._gaussian = (self._density == 'gaussian')

        self.device = device = torch.device(device)
        self.gpu = (device != torch.device('cpu'))
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
        if blockdim is None :
            if self.gpu :
                blockdim = math.ceil(self.dimension / 2)
            else :
                blockdim = 8
        self.blockdim = blockdim

        axis = torch.arange(self.dimension, dtype=torch.float, device=self.device) * self.resolution - (self.width / 2.)
        self.num_blocks = num_blocks = math.ceil(self.dimension / blockdim)
        if self.num_blocks > 1 :
            self.grid = None
            self.grid_block_dict: Dict[Tuple[int,int,int], FloatTensor] = {}
            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3) :
                x_axis = axis[xidx*blockdim : (xidx+1)*blockdim]
                y_axis = axis[yidx*blockdim : (yidx+1)*blockdim]
                z_axis = axis[zidx*blockdim : (zidx+1)*blockdim]
                grid_block = torch.stack(torch.meshgrid([x_axis, y_axis, z_axis], indexing='ij'), dim=-1)
                self.grid_block_dict[(xidx, yidx, zidx)] = grid_block

            self.bounds = [(axis[idx*blockdim].item() + (self.resolution / 2.))
                                                for idx in range(1, num_blocks)]
        else :
            self.grid_block_dict = None
            self.grid = torch.stack(torch.meshgrid([axis, axis, axis], indexing='ij'), dim=-1)

    def get_empty_grid(self, num_channels: int, batch_size: Optional[int] = None, init_zero: bool = False) -> FloatTensor:
        if init_zero :
            if batch_size is None :
                return torch.zeros(self.grid_dimension(num_channels), device=self.device)
            else :
                return torch.zeros((batch_size,) + self.grid_dimension(num_channels), device=self.device)
        else :
            if batch_size is None :
                return torch.empty(self.grid_dimension(num_channels), device=self.device)
            else :
                return torch.empty((batch_size,) + self.grid_dimension(num_channels), device=self.device)

    """ DEVICE """
    def to(self, device, update_blockdim: bool = True, blockdim: Optional[int] = None) :
        device = torch.device(device)
        if device == self.device :
            return
        self.device = device
        self.gpu = (device != torch.device('cpu'))

        if update_blockdim :
            self._setup_block(blockdim)
        return self

    def cuda(self, update_blockdim: bool = True, blockdim: Optional[int] = None) :
        return self.to('cuda', update_blockdim, blockdim)

    def cpu(self, update_blockdim: bool = True, blockdim: Optional[int] = None) :
        return self.to('cpu', update_blockdim, blockdim)
            
    """ Forward """
    def forward(
        self,
        coords: FloatTensor,
        center: FloatTensor,
        channels: Union[FloatTensor, LongTensor],
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[FloatTensor] = None
    ) -> FloatTensor :
        """
        coords: (V, 3)
        center: (3,)
        channels: (V, C) or (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out: (C,D,H,W)
        """
        if channels.dim() == 1 :
            types = channels
            return self.forward_type(coords, center, types, radii, random_translation, random_rotation, out)
        else :
            features = channels
            return self.forward_feature(coords, center, features, radii, random_translation, random_rotation, out)

    def forward_mol(
        self,
        rdmol,
        center: FloatTensor,
        channels: Union[FloatTensor, LongTensor],
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[FloatTensor] = None
    ) -> FloatTensor :
        """
        coords: (V, 3)
        center: (3,)
        channels: (V, C) or (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out: (C,D,H,W)
        """
        if channels.dim() == 1 :
            types = channels
            return self.forward_type(coords, center, types, radii, random_translation, random_rotation, out)
        else :
            features = channels
            return self.forward_feature(coords, center, features, radii, random_translation, random_rotation, out)

    __call__ = forward

    """ VECTOR """
    @torch.no_grad()
    def forward_feature(
        self,
        coords: FloatTensor,
        center: FloatTensor,
        features: FloatTensor,
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[FloatTensor] = None
    ) -> FloatTensor :
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
        coords = coords - center.unsqueeze(0)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        # Set Out
        if out is None :
            C = features.size(1)
            out = self.get_empty_grid(C)

        # Set Atom Radii
        is_atom_wise_radii = (isinstance(radii, Tensor) and (not self.channel_wise_radii))
        if self.channel_wise_radii :
            atom_size = radii.max().item() * self.atom_scale
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
                if overlap.size(0) == 0 :
                    out_block.fill_(0.)
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block, features_block = coords[overlap], features[overlap]
                radii_block = radii[overlap] if is_atom_wise_radii else radii
                self._set_grid_feature(coords_block, features_block, radii_block, grid_block, out_block)
        else :
            grid = self.grid
            self._set_grid_feature(coords, features, radii, grid, out)
        
        return out

    def _check_args_feature(self, coords: FloatTensor, features: FloatTensor, radii: Union[float,FloatTensor], 
                    out: Optional[FloatTensor] = None) :
        V = coords.size(0)
        C = features.size(1)
        D = H = W = self.dimension
        assert features.size(0) == V, f'atom features does not match number of atoms: {features.size(0)} vs {V}'
        assert features.dim() == 2, f"atom features does not match dimension: {tuple(features.size())} vs {(V,'*')}"
        if self.channel_wise_radii :
            assert isinstance(radii, Tensor)
            assert radii.size() == (C,), f'radii does not match dimension (number of channels,): {tuple(radii.size())} vs {(C,)}'
        else :
            if isinstance(radii, Tensor) :
                assert radii.size() == (V,), f'radii does not match dimension (number of atoms,): {tuple(radii.size())} vs {(V,)}'
        if out is not None :
            assert out.size() == (C, D, H, W), f'Output grid dimension incorrect: {tuple(out.size())} vs {(C,D,H,W)}'

    def _set_grid_feature(
        self,
        coords: FloatTensor,
        features: FloatTensor,
        radii: Union[float, FloatTensor],
        grid: FloatTensor,
        out: FloatTensor,
    ) -> FloatTensor :
        """
        coords: (V, 3)
        features: (V, C)
        radii: scalar or (V, ) or (C, )
        grid: (D, H, W, 3)

        out: (C, D, H, W)
        """
        features = features.T                                               # (V, C) -> (C, V)
        D, H, W, _ = grid.size()
        grid = grid.view(-1, 3)                                             # (DHW, 3)
        if self.channel_wise_radii :
            if out.is_contiguous() :
                _out = out.view(-1, D*H*W)
                for type_idx in range(features.size(0)) :
                    typ = features[type_idx]                                # (V,)
                    res = self._calc_grid(coords, radii[type_idx], grid)    # (V, DHW)
                    torch.matmul(typ, res, out=_out[type_idx])              # (V,) @ (V, DHW) -> (DHW) = (D, H, W)
            else :
                for type_idx in range(features.size(0)) :
                    typ = features[type_idx]                                # (V,)
                    res = self._calc_grid(coords, radii[type_idx], grid)    # (V, DHW)
                    out[type_idx] = torch.matmul(typ, res).view(D, H, W)    # (D, H, W)
        else :
            if out.is_contiguous() :
                res = self._calc_grid(coords, radii, grid)                  # (V, DHW)
                torch.mm(features, res, out=out.view(-1, D*H*W))            # (V,C) @ (V, DHW) -> (C, DHW)
            else :
                res = self._calc_grid(coords, radii, grid)                  # (V, DHW)
                out[:] = torch.mm(features, res).view(-1, D, H, W)          # (C, D, H, W)
        return out

    """ INDEX """
    @torch.no_grad()
    def forward_type(
        self,
        coords: FloatTensor,
        center: FloatTensor,
        types: FloatTensor,
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out: Optional[FloatTensor] = None
    ) -> FloatTensor :
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
        coords = coords - center.unsqueeze(0)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        # Set Out
        if out is None :
            if self.channel_wise_radii :
                C = radii.size(0)
            else :
                C = torch.max(types).item() + 1
            out = self.get_empty_grid(C, init_zero=True)
        else :
            out.fill_(0.)

        # Set Atom Radii
        if self.channel_wise_radii :
            assert isinstance(radii, Tensor), \
                    'Type-Radii requires type indexed radii (torch.FloatTensor(size=(C,)))' 
            radii = radii[types]            # (C, ) -> (V, )
        atom_size = radii * self.atom_scale

        # Clipping Overlapped Atoms
        box_overlap = self._get_overlap(coords, atom_size)
        coords, types = coords[box_overlap], types[box_overlap]
        radii = radii[box_overlap] if isinstance(radii, Tensor) else radii

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
                if overlap.size(0) == 0 :
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block, types_block = coords[overlap], types[overlap]
                radii_block = radii[overlap] if isinstance(radii, Tensor) else radii
                self._set_grid_types(coords_block, types_block, radii_block, grid_block, out_block)
        else :
            grid = self.grid
            self._set_grid_types(coords, types, radii, grid, out)

        return out

    def _check_args_types(self, coords: FloatTensor, types: FloatTensor, radii: Union[float,FloatTensor], 
                    out: Optional[FloatTensor] = None) :
        V = coords.size(0)
        C = torch.max(types).item() + 1
        D = H = W = self.dimension
        assert isinstance(types, Tensor), f'types should be LongTensor, dtype: {types.dtype}'
        assert types.dim() == 1, f"types does not match dimension: {tuple(types.size())} vs {(V,)}"
        assert types.size(0) == V, f'types does not match number of atoms: {types.size(0)} vs {V}'
        if isinstance(radii, Tensor) :
            if self.channel_wise_radii :
                assert radii.dim() == 1, f"radii does not match dimension: {tuple(radii.size())} vs {('Channel',)}"
                assert radii.size() == (C,), f'radii does not match dimension (number of types,): {tuple(radii.size())} vs {(C,)}'
            else :
                assert radii.size() == (V,), f'radii does not match dimension (number of atoms,): {tuple(radii.size())} vs {(V,)}'
        if out is not None :
            assert out.size(0) >= C, f'Output channel is less than number of types: {out.size(0)} < {C}'
            assert out.size()[1:] == (D, H, W), f'Output grid dimension incorrect: {tuple(out.size())} vs {("*",D,H,W)}'

    def _set_grid_types(
        self,
        coords: FloatTensor,
        types: FloatTensor,
        radii: Union[float, FloatTensor],
        grid: FloatTensor,
        out: FloatTensor,
    ) -> FloatTensor :
        """
        coords: (V, 3)
        types: (V,)
        radii: scalar or (V, )
        grid: (D, H, W, 3)

        out: (C, D, H, W)
        """
        D, H, W, _ = grid.size()
        grid = grid.view(-1, 3)
        res = self._calc_grid(coords, radii, grid)          # (V, D*H*W)
        res = res.view(-1, D, H, W)                         # (V, D, H, W)
        types = types.view(-1, 1, 1, 1).expand(res.size())  # (V, D, H, W)
        return out.scatter_add_(0, types, res)

    """ COMMON BLOCK DIVISION """
    def _get_overlap(
        self,
        coords: FloatTensor,
        atom_size: Union[FloatTensor, float],
    ) -> LongTensor :
        if isinstance(atom_size, Tensor) and atom_size.dim() == 1:
            atom_size = atom_size.unsqueeze(1)
            lb_overlap = torch.greater(coords + atom_size, self.lower_bound).all(dim=-1)    # (V,)
            ub_overlap = torch.less(coords - atom_size, self.upper_bound).all(dim=-1)       # (V,)
        else :
            lb_overlap = torch.greater(coords, self.lower_bound - atom_size).all(dim=-1)    # (V,)
            ub_overlap = torch.less(coords, self.upper_bound + atom_size).all(dim=-1)       # (V,)
        overlap = lb_overlap.logical_and_(ub_overlap)                                       # (V,)
        return torch.where(overlap)

    def _get_overlap_blocks(
        self,
        coords: FloatTensor,
        atom_size: Union[FloatTensor, float]
    ) -> Dict[Tuple[int, int, int], LongTensor] :

        def get_axis_overlap_list(coord_1d, atom_size) -> List[BoolTensor]:
            overlaps = [None] * self.num_blocks
            for i in range(self.num_blocks) :
                if i == 0 :
                    upper = torch.less(coord_1d, self.bounds[i] + atom_size)        # (V,)
                    overlaps[i] = upper
                elif i == self.num_blocks - 1 :
                    lower = torch.greater(coord_1d, self.bounds[i-1] - atom_size)   # (V,)
                    overlaps[i] = lower
                else :
                    lower = torch.greater(coord_1d, self.bounds[i-1] - atom_size)   # (V,)
                    upper = torch.less(coord_1d, self.bounds[i] + atom_size)        # (V,)
                    overlaps[i] = lower.logical_and_(upper)
            return overlaps
        
        overlap_dict = {key: None for key in self.grid_block_dict.keys()}
        x, y, z = torch.unbind(coords, dim=-1)

        x_overlap_list = get_axis_overlap_list(x, atom_size)
        y_overlap_list = get_axis_overlap_list(y, atom_size)
        z_overlap_list = get_axis_overlap_list(z, atom_size)
        for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3) :
            x_overlap = x_overlap_list[xidx]
            y_overlap = y_overlap_list[yidx]
            z_overlap = z_overlap_list[zidx]
            overlap_dict[(xidx,yidx,zidx)] = torch.where(x_overlap & y_overlap & z_overlap)[0]
        return overlap_dict

    """ COMMON - GRID CALCULATION """
    def _calc_grid(
        self,
        coords: FloatTensor,
        radii: Union[float, FloatTensor],
        grid: FloatTensor,
    ) -> FloatTensor :
        """
        coords: (V, 3)
        radii: scalar or (V, )
        grid: (D*H*W, 3)

        out: (V, D*H*W)
        """
        dist = torch.cdist(coords, grid)                    # (V, D, H, W)
        if isinstance(radii, Tensor) :
            radii = radii.unsqueeze(-1)
        dr = dist.div_(radii)                               # (V, D, H, W)
        if self._gaussian :
            return self.__calc_grid_density_gaussian(dr)
        else :
            return self.__calc_grid_density_binary(dr)

    def __calc_grid_density_binary(self, dr: FloatTensor) -> FloatTensor :
        return dr.less_(self.atom_scale)

    def __calc_grid_density_gaussian(self, dr: FloatTensor) -> FloatTensor :
        out = dr.pow_(2).mul_(-2.0).exp_()
        out.masked_fill_(dr > self.atom_scale, 0)
        return out

    def asarray(self, array, obj) :
        if isinstance(array, np.ndarray) :
            array = torch.from_numpy(array)
        if isinstance(array, torch.Tensor) :
            if obj in ['coords', 'center', 'feature', 'radii'] :
                return array.to(device = self.device, dtype=torch.float)
            elif obj == 'type' : 
                return array.to(device = self.device, dtype=torch.long)
        else :
            if obj in ['coords', 'center', 'feature', 'radii'] :
                return torch.tensor(array, dtype=torch.float, device=self.device)
            elif obj == 'type' : 
                return torch.tensor(array, dtype=torch.long, device=self.device)
        raise ValueError("obj should be ['coords', center', 'type', 'feature', 'radii']")

    @staticmethod
    def do_random_transform(coords, center, random_translation, random_rotation) :
        return do_random_transform(coords, center, random_translation, random_rotation)
