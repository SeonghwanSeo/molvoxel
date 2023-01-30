from rdkit import Chem
from rdkit.Chem import BondType
import numpy as np

from typing import List, Optional, Callable, OrderedDict, Dict
from rdkit.Chem import Atom, Bond, Mol
from numpy.typing import ArrayLike

class RDAtomChannelGetter() :
    def __init__(self, function: Callable[[Atom], List[float]], channels: List[str]) :
        self.function = function
        self.channels = channels
        self.num_channels = len(channels)

    def get_feature(self, atom: Atom) -> List[float]:
        return self.function(atom)

BondChannelDict = {
        BondType.SINGLE: [1, 0, 0, 0],
        BondType.DOUBLE: [0, 1, 0, 0],
        BondType.TRIPLE: [0, 0, 1, 0],
        BondType.AROMATIC: [0, 0, 0, 1],
}
BondChannel = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
class RDBondChannelGetter() :
    def __init__(self, function: Callable[[Bond], List[float]], channels: List[str]) :
        self.function = function
        self.channels = channels
        self.num_channels = len(channels)

    def get_feature(self, bond: Bond) -> List[float]:
        return self.function(bond)
        
    @classmethod
    def default(cls) :
        function = lambda bond: BondChannelDict[bond.GetBondType()]
        return cls(function, BondChannel)

class RDMolChannelGetter() :
    def __init__(
            self,
            atom_getter: RDAtomChannelGetter,
            bond_getter: Optional[RDBondChannelGetter] = None,
            prefix: str = ''
        ) :
        self.atom_getter = atom_getter
        self.bond_getter = bond_getter
        if bond_getter is not None :
            self.use_bond = True
            self.num_atom_channels = atom_getter.num_channels
            self.num_bond_channels = bond_getter.num_channels
            self.num_channels = atom_getter.num_channels + bond_getter.num_channels
            self.channels = atom_getter.channels + bond_getter.channels
        else :
            self.use_bond = False
            self.num_channels = atom_getter.num_channels
            self.channels = atom_getter.channels
        self.prefix = prefix

    def get_feature(self, rdmol, start_index = 0, num_total_channels = None, **kwargs) -> List[List[float]] :
        return self._get_feature(rdmol, start_index, num_total_channels, **kwargs)

    def _get_feature(self, rdmol, start_index = 0, num_total_channels = None, **kwargs) -> List[List[float]]:
        atom_feature = [self._run_atom(atom, start_index, num_total_channels, **kwargs) for atom in rdmol.GetAtoms()]
        if self.use_bond :
            bond_feature = [self._run_bond(bond, start_index, num_total_channels, **kwargs) for bond in rdmol.GetBonds()]
            feature = atom_feature + bond_feature
        else :
            feature = atom_feature
        return feature
    
    def _run_atom(self, atom: Atom, start_index = 0, num_total_channels = None, **kwargs) -> List[float]:
        atom_feature = self.atom_getter.get_feature(atom, **kwargs)
        if num_total_channels is not None:
            end_index = start_index + self.num_channels
            if self.use_bond :
                left_margin = [0] * start_index
                right_margin = [0] * (num_total_channels - end_index + self.num_bond_channels)
            else :
                left_margin = [0] * start_index
                right_margin = [0] * (num_total_channels - end_index)
            return left_margin + atom_feature + right_margin
        else :
            if self.use_bond :
                right_margin = [0] * (self.num_bond_channels)
                return atom_feature + right_margin
            else :
                return atom_feature

    def _run_bond(self, bond: Bond, start_index = 0, num_total_channels = None, **kwargs) -> List[float]:
        bond_feature = self.bond_getter.get_feature(bond, **kwargs)
        if num_total_channels :
            end_index = start_index + self.num_channels
            left_margin = [0] * (start_index + self.num_atom_channels)
            right_margin = [0] * (num_total_channels - end_index)
            return left_margin + bond_feature + right_margin
        else :
            left_margin = [0] * self.num_atom_channels
            return left_margin + bond_feature
    
    def split_channel(self, grids: ArrayLike) -> Dict[str, ArrayLike] :
        assert np.shape(grids)[0] == self.num_channels
        return {(self.prefix + channel): grid for channel, grid in zip(self.channels, grids)}
