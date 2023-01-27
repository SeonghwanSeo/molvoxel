from pymolgrid.pymol import visualize_complex

from rdkit import Chem
from rdkit.Chem import Mol
from typing import List, Dict
from numpy.typing import ArrayLike

def apply_coord(rdmol: Mol, coords: ArrayLike) -> Mol :
    rdmol = Chem.Mol(rdmol)
    conf = rdmol.GetConformer()
    for i in range(rdmol.GetNumAtoms()) :
        conf.SetAtomPosition(i, coords[i].tolist())
    return rdmol

def split_channels(grid: ArrayLike, channel_names: List[str]) -> Dict[str, ArrayLike]:
    return {name: grid[i] for i, name in enumerate(channel_names)}

def draw_pse(pse_path, ligand_rdmol, pocket_rdmol, ligand_grids, pocket_grids, \
        ligand_channels, pocket_channels, center, resolution, new_coords = None) :
    if new_coords is not None :
        num_ligand_atoms = ligand_rdmol.GetNumAtoms()
        ligand_coords, pocket_coords = new_coords[:num_ligand_atoms], new_coords[num_ligand_atoms:]
        ligand_rdmol = apply_coord(ligand_rdmol, ligand_coords)
        pocket_rdmol = apply_coord(pocket_rdmol, pocket_coords)
    else :
        ligand_rdmol, pocket_rdmol = ligand_rdmol, pocket_rdmol

    ligand_grid_dict = split_channels(ligand_grids, ligand_channels) 
    pocket_grid_dict = split_channels(pocket_grids, pocket_channels) 
    
    visualize_complex(pse_path, ligand_rdmol, pocket_rdmol, \
            ligand_grid_dict, pocket_grid_dict, \
            center, resolution
    )
