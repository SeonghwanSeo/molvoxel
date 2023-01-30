import os
from pymolgrid import Voxelizer
from pymolgrid.rdkit import rdkit_binding
import numpy as np
from rdkit import Chem
try :
    from pymolgrid.pymol import visualize_mol, visualize_complex
    from utils import apply_coord
    pymol = True
    os.system('mkdir -p result_rdkit')
except :
    pymol = False

voxelizer = Voxelizer(resolution=0.5, dimension=48, atom_scale=1.5, density='gaussian', \
                    channel_wise_radii=False) # Default
voxelizer = rdkit_binding(voxelizer)

""" LOAD DATA """
ligand_path = './10gs/ligand.sdf'
pocket_path = './10gs/pocket.pdb'

ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
pocket_rdmol = Chem.MolFromPDBFile(pocket_path)

""" INDOLE TEST """
indole_rdmol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
grid_dict, coords = voxelizer.run_mol(indole_rdmol, return_coords = True)
center = coords.mean(0)
if pymol :
    visualize_mol('result_rdkit/indole.pse', indole_rdmol, grid_dict, center, 0.5)

""" SINGLE MOL TEST """
grid_dict, coords = voxelizer.run_mol(ligand_rdmol, return_coords = True)
center = coords.mean(0)
if pymol :
    visualize_mol('result_rdkit/ligand.pse', ligand_rdmol, grid_dict, center, 0.5)

""" COMPLEX TEST """
ligand_grid_dict, pocket_grid_dict, ligand_coords, pocket_coords = \
        voxelizer.run_complex(ligand_rdmol, pocket_rdmol, return_coords = True)
center = ligand_coords.mean(0)
if pymol :
    ligand_rdmol = apply_coord(ligand_rdmol, ligand_coords)
    pocket_rdmol = apply_coord(pocket_rdmol, pocket_coords)
    visualize_complex('result_rdkit/complex.pse', ligand_rdmol, pocket_rdmol, ligand_grid_dict, pocket_grid_dict, center, 0.5)
