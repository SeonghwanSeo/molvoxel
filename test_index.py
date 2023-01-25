import torch
import os

from pymolgrid import grid_maker
from rdkit import Chem

ligand_path = './10gs/ligand.sdf'
pocket_path = './10gs/pocket.pdb'
save_dir = 'result_index'

try :
    from test_utils import draw_pse
    os.system(f'mkdir -p {save_dir}')
except :
    draw_pse = (lambda *x: None)

ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
pocket_rdmol = Chem.MolFromPDBFile(pocket_path)

ligand_channels = ['C', 'N', 'O', 'S']
pocket_channels = ['C', 'N', 'O', 'S']
num_ligand_types = len(ligand_channels)
num_pocket_types = len(pocket_channels)
num_types = num_ligand_types + num_pocket_types

ligand_type_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
pocket_type_dict = {'C': 4, 'N': 5, 'O': 6, 'S': 7}

ligand_coords = torch.FloatTensor(ligand_rdmol.GetConformer().GetPositions())
ligand_types = torch.LongTensor([ligand_type_dict[atom.GetSymbol()] for atom in ligand_rdmol.GetAtoms()])
ligand_center = ligand_coords.mean(dim=0)

pocket_coords = torch.FloatTensor(pocket_rdmol.GetConformer().GetPositions())
pocket_types = torch.LongTensor([pocket_type_dict[atom.GetSymbol()] for atom in pocket_rdmol.GetAtoms()])

coords = torch.cat([ligand_coords, pocket_coords], dim=0)
type_index = torch.cat([ligand_types, pocket_types], dim=0)
center = ligand_center

type_radii = torch.ones((num_types,), dtype=torch.float)
type_radii[:4] = 2.0
atom_radii = torch.ones((coords.size(0),))
atom_radii[100:] = 2.0

"""START"""
gmaker = grid_maker.GridMaker()
gmaker_hr = grid_maker.GridMaker(0.4, 64)
gmaker_gaus = grid_maker.GridMaker(gaussian_radius_multiple=-1.5)

grid = torch.empty(gmaker.grid_dimension(num_types))
grid_hr = torch.empty(gmaker_hr.grid_dimension(num_types))
grid_gaus = torch.empty(gmaker_gaus.grid_dimension(num_types))

"""SET DEVICE"""
device = 'cpu'
coords, type_index, center, type_radii, atom_radii = \
        coords.to(device), type_index.to(device), center.to(device), type_radii.to(device), atom_radii.to(device)
#gmaker.cpu()
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
ref_grid = gmaker.forward_index(coords, center, type_index, radii=1.0)
out_grid = gmaker.forward_index(coords, center, type_index, radii=1.0, out=grid)
assert (grid - ref_grid).abs_().less_(1e-5).all().item(), 'REPRODUCTION FAIL'
assert grid is out_grid, 'INPLACE FAILE'
draw_pse(f'{save_dir}/ref.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 2: High Resolution')
gmaker_hr.forward_index(coords, center, type_index, radii=1.0, out=grid_hr)
draw_pse(f'{save_dir}/hr.pse', ligand_rdmol, pocket_rdmol, ligand_grid_hr, pocket_grid_hr, \
        ligand_channels, pocket_channels, center, gmaker_hr.get_resolution())

print('Test 3: With Atom-wise Radii')
gmaker.forward_index(coords, center, type_index, atom_radii, out=grid)
draw_pse(f'{save_dir}/atom-wise.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 4: Binary: True')
gmaker.set_binary(True)
gmaker.forward_index(coords, center, type_index, radii = 1.0, out=grid)
draw_pse(f'{save_dir}/binary.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 5: Raddi-Type-Index: True')
gmaker.set_binary(False)
gmaker.set_radii_type_indexed(True)
gmaker.forward_index(coords, center, type_index, type_radii, out=grid)
draw_pse(f'{save_dir}/type-wise.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 6: Density: Gaussian')
gmaker_gaus.forward_index(coords, center, type_index, radii = 1.0, out=grid_gaus)
draw_pse(f'{save_dir}/gaussian.pse', ligand_rdmol, pocket_rdmol, ligand_grid_gaus, pocket_grid_gaus, \
        ligand_channels, pocket_channels, center, gmaker_gaus.get_resolution())

print('Test 7: Random transform')
gmaker.set_radii_type_indexed(False)
new_coords = gmaker.do_transform(coords, center, random_translation=0.5, random_rotation=True)
gmaker.forward_index(new_coords, center, type_index, radii = 1.0, out=grid)
draw_pse(f'{save_dir}/transform.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution(), new_coords=new_coords)
