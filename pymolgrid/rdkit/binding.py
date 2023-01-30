import rdkit
from rdkit import Chem
from rdkit.Chem import BondType
import types
import numpy as np

from rdkit.Chem import Mol
from typing import Callable, Optional, Union, Dict, Tuple, List
from numpy.typing import ArrayLike, NDArray

#from .getter import AtomTypeGetter, AtomFeatureGetter, BondTypeGetter, BondFeatureGetter
from .getter import RDMolChannelGetter

NDArrayFloat64 = NDArray[np.float64]
NDArrayFloat32 = NDArray[np.float32]
NDArrayInt16 = NDArray[np.int16]

class RDMolWrapper() :
    def __init__(
        self,
        channel_getters: List[RDMolChannelGetter],
    ) :
        self.getter_list = channel_getters

        self.num_total_channels = 0
        start_index = 0
        self.start_index_list = []
        for getter in channel_getters :
            self.start_index_list.append(start_index)
            num_channels = getter.num_channels
            self.num_total_channels += num_channels
            start_index += num_channels

    @staticmethod
    def run(
        voxelizer,
        rdmol_list: List[Mol],
        center: Union[ArrayLike, int, None] = None,
        radii: Union[float, List[float], List[List[float]]] = 1.0,
        random_translation: float = 0.0,
        random_rotation: bool = False,
        return_inputs: bool = False,
    ) -> Dict[str, ArrayLike]:
        wrapper = voxelizer.wrapper

        coords_list, features_list = wrapper.enumerate(rdmol_list)
        coords = np.concatenate(coords_list, axis=0)                        # (V, 3)
        features = np.concatenate(features_list, axis=0, dtype=np.float32)  # (V, C)
        
        if center is None :
            center = coords.mean(axis=0)
        elif isinstance(center, int) :
            center = coords_list[center].mean(axis=0)

        if np.isscalar(radii) :
            pass
        elif isinstance(radii[0], float) :
            if not voxelizer.type_wise_radii :
                assert len(radii) == len(rdmol_list)
                radii_list = [[r] * c.shape[0] for r, c in zip(radii, coords_list)]
                radii = np.concatenate(radii_list, dtype=np.float32)
        else :
            radii = np.concatenate(radii, dtype=np.float32)

        coords = voxelizer.asarray(coords, 'coords')
        center = voxelizer.asarray(center, 'center')
        features = voxelizer.asarray(features, 'feature')
        radii = radii if np.isscalar(radii) else voxelizer.asarray(radii, 'radii')

        coords = voxelizer.do_random_transform(coords, center, random_translation, random_rotation)
        grids = voxelizer.forward(coords, center, features, radii)
        if return_inputs :
            return grids, {'coords': coords, 'center': center, 'features': features, 'radii': radii}
        else :
            return grids

    def enumerate(self, rdmol_list: List[Mol]) :
        getter_list, start_index_list = self.getter_list, self.start_index_list
        coords_list = [self._get_conf(rdmol, getter.use_bond) \
                                for rdmol, getter in zip(rdmol_list, getter_list)]
        features_list = [getter.get_feature(rdmol, start_index, self.num_total_channels) \
                                for rdmol, getter, start_index in zip(rdmol_list, getter_list, start_index_list)]
        return coords_list, features_list

    @staticmethod
    def _get_conf(rdmol: Mol, use_bond: bool) -> NDArrayFloat64:
        conf = rdmol.GetConformer()
        atom_coords = conf.GetPositions()
        if use_bond :
            bond_coords = [(atom_coords[bond.GetBeginAtomIdx()] + atom_coords[bond.GetEndAtomIdx()])/2 \
                                                                    for bond in rdmol.GetBonds()]
            coords = np.concatenate([atom_coords, bond_coords], axis=0)
        else :
            coords = atom_coords
        return coords
    
    def split_channel(self, grids) :
        end_index = 0
        grid_dist_list = []
        for getter in self.getter_list :
            start_index, end_index = end_index, end_index + getter.num_channels
            mol_grids = grids[start_index: end_index]
            grid_dist_list.append(getter.split_channel(mol_grids))
        return grid_dist_list
