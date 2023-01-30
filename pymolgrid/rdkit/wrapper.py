import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import types
import numpy as np

from rdkit.Chem import Mol, Atom
from typing import Callable, Optional, Union, Dict, Tuple
from numpy.typing import ArrayLike

def rdkit_binding(voxelizer) :
    voxelizer.run_mol = types.MethodType(run_mol, voxelizer)
    voxelizer.run_complex = types.MethodType(run_complex, voxelizer)
    return voxelizer

etkdg = AllChem.srETKDGv3()
def run_mol(
    voxelizer,
    rdmol: Mol,
    center: Optional[ArrayLike] = None,
    radii: float = 1.0,
    random_translation: float = 0.0,
    random_rotation: bool = False,
    return_coords: bool = False,
) -> Dict[str, ArrayLike]:

    if rdmol.GetNumConformers() == 0 :
        Chem.AddHs(rdmol)
        rdmol = Chem.AddHs(rdmol)
        rdmol = AllChem.EmbedMolecule(mol, etkdg)
        rdmol = Chem.RemoveHs(rdmol)
    assert rdmol.GetNumConformers() > 0, 'No Conformer'

    conf = rdmol.GetConformer()
    coords = conf.GetPositions()
    if center is None :
        center = coords.mean(axis=0)

    atom_symbols = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    channels = list(set(atom_symbols))
    channels_dict = {symbol: i for i, symbol in enumerate(channels)}
    atom_types = [channels_dict[symbol] for symbol in atom_symbols]

    coords = voxelizer.asarray(coords, typ=float)
    center = voxelizer.asarray(center, typ=float)
    coords = voxelizer.do_random_transform(coords, center, random_translation, random_rotation)
    atom_types = voxelizer.asarray(atom_types, typ=int)
    out = voxelizer.forward_type(coords, center, atom_types, radii, random_translation, random_rotation)

    out = {channel: channel_out for channel, channel_out in zip(channels, out)}

    if return_coords :
        return out, coords
    else :
        return out

def run_mol(
    voxelizer,
    rdmol: Mol,
    center: Optional[ArrayLike] = None,
    radii: float = 1.0,
    random_translation: float = 0.0,
    random_rotation: bool = False,
    return_coords: bool = False,
) -> Dict[str, ArrayLike]:

    if rdmol.GetNumConformers() == 0 :
        _rdmol = Chem.AddHs(rdmol)
        AllChem.EmbedMolecule(_rdmol, etkdg)
        _rdmol = Chem.RemoveHs(_rdmol)
        rdmol.AddConformer(_rdmol.GetConformer())
    assert rdmol.GetNumConformers() > 0, 'No Conformer'

    conf = rdmol.GetConformer()
    coords = conf.GetPositions()
    if center is None :
        center = coords.mean(axis=0)

    atom_symbols = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    channels = list(set(atom_symbols))
    channels_dict = {symbol: i for i, symbol in enumerate(channels)}
    atom_types = [channels_dict[symbol] for symbol in atom_symbols]

    coords = voxelizer.asarray(coords, typ=float)
    center = voxelizer.asarray(center, typ=float)
    coords = voxelizer.do_random_transform(coords, center, random_translation, random_rotation)
    atom_types = voxelizer.asarray(atom_types, typ=int)
    grids = voxelizer.forward_type(coords, center, atom_types, radii, random_translation, random_rotation)

    grid_dict = {channel: grid for channel, grid in zip(channels, grids)}

    if return_coords :
        return grid_dict, coords
    else :
        return grid_dict

def run_complex(
    voxelizer,
    ligand_rdmol: Mol,
    protein_rdmol: Mol,
    center: Optional[ArrayLike] = None,
    radii: float = 1.0,
    random_translation: float = 0.0,
    random_rotation: bool = False,
    return_coords: bool = False,
) -> Tuple[Dict[str, ArrayLike], Dict[str, ArrayLike]]:

    ligand_conf = ligand_rdmol.GetConformer()
    ligand_coords = ligand_conf.GetPositions()
    protein_conf = protein_rdmol.GetConformer()
    protein_coords = protein_conf.GetPositions()
    if center is None :
        center = ligand_coords.mean(axis=0)
    
    ligand_atom_symbols = [atom.GetSymbol() for atom in ligand_rdmol.GetAtoms()]
    ligand_channels = list(set(ligand_atom_symbols))
    ligand_channels_dict = {symbol: i for i, symbol in enumerate(ligand_channels)}
    ligand_atom_types = [ligand_channels_dict[symbol] for symbol in ligand_atom_symbols]
    num_ligand_channels = len(ligand_channels)

    protein_atom_symbols = [atom.GetSymbol() for atom in protein_rdmol.GetAtoms()]
    protein_channels = list(set(ligand_atom_symbols))
    protein_channels_dict = {symbol: (i + num_ligand_channels) for i, symbol in enumerate(protein_channels)}
    protein_atom_types = [protein_channels_dict[symbol] for symbol in protein_atom_symbols]

    coords = np.concatenate([ligand_coords, protein_coords], axis=0)
    atom_types = np.concatenate([ligand_atom_types, protein_atom_types], axis=-1)

    coords = voxelizer.asarray(coords, typ=float)
    center = voxelizer.asarray(center, typ=float)
    coords = voxelizer.do_random_transform(coords, center, random_translation, random_rotation)
    atom_types = voxelizer.asarray(atom_types, typ=int)
    grids = voxelizer.forward_type(coords, center, atom_types, radii, random_translation, random_rotation)

    ligand_grids, protein_grids = grids[:num_ligand_channels], grids[num_ligand_channels:]
    ligand_grid_dict = {channel: grid for channel, grid in zip(ligand_channels, ligand_grids)}
    protein_grid_dict = {channel: grid for channel, grid in zip(protein_channels, protein_grids)}

    if return_coords :
        num_ligand_atoms = ligand_rdmol.GetNumAtoms()
        return ligand_grid_dict, protein_grid_dict, coords[:num_ligand_atoms], coords[num_ligand_atoms:]
    else :
        return ligand_grid_dict, protein_grid_dict
