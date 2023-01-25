import torch
import os

from pymolgrid import grid_maker
from rdkit import Chem

ligand_path = './10gs/ligand.sdf'
pocket_path = './10gs/pocket.pdb'
save_dir = 'result_vector'

try :
    from test_utils import draw_pse
    os.system(f'mkdir -p {save_dir}')
except :
    draw_pse = (lambda *x: None)

ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
pocket_rdmol = Chem.MolFromPDBFile(pocket_path)

ligand_channels = ['C', 'N', 'O', 'S', 'Aromatic']
pocket_channels = ['C', 'N', 'O', 'S', 'Aromatic']
num_ligand_types = len(ligand_channels)
num_pocket_types = len(pocket_channels)
num_types = num_pocket_types + num_ligand_types

ligand_coords = torch.FloatTensor(ligand_rdmol.GetConformer().GetPositions())
ligand_types = torch.LongTensor([atom.GetAtomicNum() for atom in ligand_rdmol.GetAtoms()])
ligand_center = ligand_coords.mean(dim=0)

pocket_coords = torch.FloatTensor(pocket_rdmol.GetConformer().GetPositions())
pocket_types = torch.LongTensor([atom.GetAtomicNum() for atom in pocket_rdmol.GetAtoms()])

coords = torch.cat([ligand_coords, pocket_coords], dim=0)
center = ligand_center

type_vector = torch.zeros(coords.size(0), num_types)
ligand_type_vector = type_vector[:ligand_rdmol.GetNumAtoms()]
pocket_type_vector = type_vector[ligand_rdmol.GetNumAtoms():]
ligand_type_vector[ligand_types==6, 0] = 1
ligand_type_vector[ligand_types==7, 1] = 1
ligand_type_vector[ligand_types==8, 2] = 1
ligand_type_vector[ligand_types==16, 3] = 1
ligand_type_vector[[atom.GetIsAromatic() for atom in ligand_rdmol.GetAtoms()], 4] = 1
pocket_type_vector[pocket_types==6, 5] = 1
pocket_type_vector[pocket_types==7, 6] = 1
pocket_type_vector[pocket_types==8, 7] = 1
pocket_type_vector[pocket_types==16, 8] = 1
pocket_type_vector[[atom.GetIsAromatic() for atom in pocket_rdmol.GetAtoms()], 9] = 1

type_radii = torch.ones((num_types,), dtype=torch.float)
type_radii[:4] = 2.0
type_radii[4] = 3.0
atom_radii = torch.ones((coords.size(0),))
atom_radii[100:] = 2.0

"""START"""
gmaker = grid_maker.GridMaker(device = 'cpu')
gmaker_hr = grid_maker.GridMaker(0.4, 64)
gmaker_gaus = grid_maker.GridMaker(gaussian_radius_multiple=-1.5)

grid = torch.zeros(gmaker.grid_dimension(num_types))
grid_hr = torch.zeros(gmaker_hr.grid_dimension(num_types))
grid_gaus = torch.zeros(gmaker_gaus.grid_dimension(num_types))

"""SET DEVICE"""
device = 'cpu'
coords, type_vector, center, type_radii, atom_radii = \
        coords.to(device), type_vector.to(device), center.to(device), type_radii.to(device), atom_radii.to(device)
#gmaker.to(device)
gmaker_hr.cpu()
gmaker_gaus.to(device)

#grid = grid.to(device)
grid_hr = grid_hr.cpu()
grid_gaus = grid_gaus.to(device)

"""SPLIT GRID"""
ligand_grid, pocket_grid = torch.split(grid, (num_ligand_types, num_pocket_types))
ligand_grid_hr, pocket_grid_hr = torch.split(grid_hr, (num_ligand_types, num_pocket_types))
ligand_grid_gaus, pocket_grid_gaus = torch.split(grid_gaus, (num_ligand_types, num_pocket_types))

print('Test 1: Binary: False, Raddi-Type-Index: False, Density: Mixed (Default)')
ref_grid = gmaker.forward_vector(coords, center, type_vector, radii=1.0)
out_grid = gmaker.forward_vector(coords, center, type_vector, radii=1.0, out=grid)
assert (grid - ref_grid).abs_().less_(1e-5).all().item(), 'REPRODUCTION FAIL'
assert grid is out_grid, 'INPLACE FAILE'
draw_pse(f'{save_dir}/ref.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 2: High Resolution')
gmaker_hr.forward_vector(coords, center, type_vector, radii=1.0, out=grid_hr)
draw_pse(f'{save_dir}/hr.pse', ligand_rdmol, pocket_rdmol, ligand_grid_hr, pocket_grid_hr, \
        ligand_channels, pocket_channels, center, gmaker_hr.get_resolution())

print('Test 3: With Atom-wise Radii')
gmaker.forward_vector(coords, center, type_vector, atom_radii, out=grid)
draw_pse(f'{save_dir}/atom-wise.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 4: Binary: True')
gmaker.set_binary(True)
gmaker.forward_vector(coords, center, type_vector, radii = 1.0, out=grid)
draw_pse(f'{save_dir}/binary.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 5: Raddi-Type-Index: True')
gmaker.set_binary(False)
gmaker.set_radii_type_indexed(True)
gmaker.forward_vector(coords, center, type_vector, type_radii, out=grid)
draw_pse(f'{save_dir}/type-wise.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 6: Density: Gaussian')
gmaker_gaus.forward_vector(coords, center, type_vector, radii = 1.0, out=grid_gaus)
draw_pse(f'{save_dir}/gaussian.pse', ligand_rdmol, pocket_rdmol, ligand_grid_gaus, pocket_grid_gaus, \
        ligand_channels, pocket_channels, center, gmaker_gaus.get_resolution())

print('Test 7: Random transform')
gmaker.set_radii_type_indexed(False)
new_coords = gmaker.do_transform(coords, center, random_translation=0.5, random_rotation=True)
gmaker.forward_vector(new_coords, center, type_vector, radii = 1.0, out=grid)
draw_pse(f'{save_dir}/transform.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution(), new_coords=new_coords)
