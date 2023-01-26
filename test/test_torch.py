import os
from pymolgrid.torch import GridMaker, Transform
import torch
from rdkit import Chem
try :
    from utils import draw_pse
    pymol = True
except :
    draw_pse = (lambda *x, **kwargs: None)
    pymol = False

device = 'cpu'

gmaker = GridMaker() # 0.5, 48
gmaker_hr = GridMaker(0.4, 64, device=device)
gmaker_gaus = GridMaker(gaussian_radius_multiple=-1.5)
transform = Transform(random_translation=0.5, random_rotation=True)
gmaker.to(device)
gmaker_gaus.to(device)

""" LOAD DATA """
ligand_path = './10gs/ligand.sdf'
pocket_path = './10gs/pocket.pdb'

ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
pocket_rdmol = Chem.MolFromPDBFile(pocket_path)

ligand_coords = torch.FloatTensor(ligand_rdmol.GetConformer().GetPositions())
ligand_center = ligand_coords.mean(dim=0)
pocket_coords = torch.FloatTensor(pocket_rdmol.GetConformer().GetPositions())
coords = torch.cat([ligand_coords, pocket_coords], dim=0)
center = ligand_center

""" INDEX """
print('Test Type Index')
save_dir = 'result_index'
if pymol :
    os.system(f'mkdir -p {save_dir}')

ligand_channels = ['C', 'N', 'O', 'S']
pocket_channels = ['C', 'N', 'O', 'S']
num_ligand_types = len(ligand_channels)
num_pocket_types = len(pocket_channels)
num_types = num_ligand_types + num_pocket_types
ligand_type_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
pocket_type_dict = {'C': 4, 'N': 5, 'O': 6, 'S': 7}
ligand_types = torch.LongTensor([ligand_type_dict[atom.GetSymbol()] for atom in ligand_rdmol.GetAtoms()])
pocket_types = torch.LongTensor([pocket_type_dict[atom.GetSymbol()] for atom in pocket_rdmol.GetAtoms()])
type_index = torch.cat([ligand_types, pocket_types], dim=0)

type_radii = torch.ones((num_types,), dtype=torch.float)
type_radii[:4] = 2.0
atom_radii = torch.ones((coords.size(0),))
atom_radii[100:] = 2.0

grid = torch.empty(gmaker.grid_dimension(num_types), device=device)
grid_hr = torch.empty(gmaker_hr.grid_dimension(num_types), device=device)
grid_gaus = torch.empty(gmaker_gaus.grid_dimension(num_types), device=device)

coords, type_index, center, type_radii, atom_radii = \
        coords.to(device), type_index.to(device), center.to(device), type_radii.to(device), atom_radii.to(device)

ligand_grid, pocket_grid = torch.split(grid, (num_ligand_types, num_pocket_types))
ligand_grid_hr, pocket_grid_hr = torch.split(grid_hr, (num_ligand_types, num_pocket_types))
ligand_grid_gaus, pocket_grid_gaus = torch.split(grid_gaus, (num_ligand_types, num_pocket_types))

print('Test 1: Binary: False, Raddi-Type-Index: False, Density: Mixed (Default)')
ref_grid = gmaker.forward(coords, center, type_index, radii=1.0)
out_grid = gmaker.forward(coords, center, type_index, radii=1.0, out=grid)
assert (grid - ref_grid).abs_().less_(1e-5).all().item(), 'REPRODUCTION FAIL'
assert grid is out_grid, 'INPLACE FAILE'
draw_pse(f'{save_dir}/ref.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 2: High Resolution')
gmaker_hr.forward(coords, center, type_index, radii=1.0, out=grid_hr)
draw_pse(f'{save_dir}/hr.pse', ligand_rdmol, pocket_rdmol, ligand_grid_hr, pocket_grid_hr, \
        ligand_channels, pocket_channels, center, gmaker_hr.get_resolution())

print('Test 3: With Atom-wise Radii')
gmaker.forward(coords, center, type_index, atom_radii, out=grid)
draw_pse(f'{save_dir}/atom-wise.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 4: Binary: True')
gmaker.set_binary(True)
gmaker.forward(coords, center, type_index, radii = 1.0, out=grid)
draw_pse(f'{save_dir}/binary.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 5: Raddi-Type-Index: True')
gmaker.set_binary(False)
gmaker.set_radii_type_indexed(True)
gmaker.forward(coords, center, type_index, type_radii, out=grid)
draw_pse(f'{save_dir}/type-wise.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 6: Density: Gaussian')
gmaker_gaus.forward(coords, center, type_index, radii = 1.0, out=grid_gaus)
draw_pse(f'{save_dir}/gaussian.pse', ligand_rdmol, pocket_rdmol, ligand_grid_gaus, pocket_grid_gaus, \
        ligand_channels, pocket_channels, center, gmaker_gaus.get_resolution())

print('Test 7: Random transform')
gmaker.set_radii_type_indexed(False)
new_coords = transform.forward(coords, center)
gmaker.forward(new_coords, center, type_index, radii = 1.0, out=grid)
draw_pse(f'{save_dir}/transform.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution(), new_coords=new_coords)

""" Vector """
print('Test Type Vector')
save_dir = 'result_vector'
if pymol :
    os.system(f'mkdir -p {save_dir}')

ligand_channels = ['C', 'N', 'O', 'S', 'Aromatic']
pocket_channels = ['C', 'N', 'O', 'S', 'Aromatic']
num_ligand_types = len(ligand_channels)
num_pocket_types = len(pocket_channels)
num_types = num_ligand_types + num_pocket_types

def get_vector(atom, is_pocket) :
    res = [0] * 10
    idx = 5 if is_pocket else 0
    symbol = atom.GetSymbol()
    if symbol == 'C'        : res[idx+0] = 1
    elif symbol == 'N'      : res[idx+1] = 1
    elif symbol == 'O'      : res[idx+2] = 1
    elif symbol == 'S'      : res[idx+3] = 1
    if atom.GetIsAromatic() : res[idx+4] = 1
    return res
ligand_types = torch.FloatTensor([get_vector(atom, False) for atom in ligand_rdmol.GetAtoms()])
pocket_types = torch.FloatTensor([get_vector(atom, True) for atom in pocket_rdmol.GetAtoms()])
type_vector = torch.cat([ligand_types, pocket_types], dim=0)

type_radii = torch.ones((num_types,), dtype=torch.float)
type_radii[:4] = 2.0
atom_radii = torch.ones((coords.size(0),))
atom_radii[100:] = 2.0

grid = torch.empty(gmaker.grid_dimension(num_types), device=device)
grid_hr = torch.empty(gmaker_hr.grid_dimension(num_types), device=device)
grid_gaus = torch.empty(gmaker_gaus.grid_dimension(num_types), device=device)

coords, type_vector, center, type_radii, atom_radii = \
        coords.to(device), type_vector.to(device), center.to(device), type_radii.to(device), atom_radii.to(device)

ligand_grid, pocket_grid = torch.split(grid, (num_ligand_types, num_pocket_types))
ligand_grid_hr, pocket_grid_hr = torch.split(grid_hr, (num_ligand_types, num_pocket_types))
ligand_grid_gaus, pocket_grid_gaus = torch.split(grid_gaus, (num_ligand_types, num_pocket_types))

print('Test 1: Binary: False, Raddi-Type-Index: False, Density: Mixed (Default)')
ref_grid = gmaker.forward(coords, center, type_vector, radii=1.0)
out_grid = gmaker.forward(coords, center, type_vector, radii=1.0, out=grid)
assert (grid - ref_grid).abs_().less_(1e-5).all().item(), 'REPRODUCTION FAIL'
assert grid is out_grid, 'INPLACE FAILE'
draw_pse(f'{save_dir}/ref.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 2: High Resolution')
gmaker_hr.forward(coords, center, type_vector, radii=1.0, out=grid_hr)
draw_pse(f'{save_dir}/hr.pse', ligand_rdmol, pocket_rdmol, ligand_grid_hr, pocket_grid_hr, \
        ligand_channels, pocket_channels, center, gmaker_hr.get_resolution())

print('Test 3: With Atom-wise Radii')
gmaker.forward(coords, center, type_vector, atom_radii, out=grid)
draw_pse(f'{save_dir}/atom-wise.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 4: Binary: True')
gmaker.set_binary(True)
gmaker.forward(coords, center, type_vector, radii = 1.0, out=grid)
draw_pse(f'{save_dir}/binary.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 5: Raddi-Type-Index: True')
gmaker.set_binary(False)
gmaker.set_radii_type_indexed(True)
gmaker.forward(coords, center, type_vector, type_radii, out=grid)
draw_pse(f'{save_dir}/type-wise.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution())

print('Test 6: Density: Gaussian')
gmaker_gaus.forward(coords, center, type_vector, radii = 1.0, out=grid_gaus)
draw_pse(f'{save_dir}/gaussian.pse', ligand_rdmol, pocket_rdmol, ligand_grid_gaus, pocket_grid_gaus, \
        ligand_channels, pocket_channels, center, gmaker_gaus.get_resolution())

print('Test 7: Random transform')
gmaker.set_radii_type_indexed(False)
new_coords = transform.forward(coords, center)
gmaker.forward(new_coords, center, type_vector, radii = 1.0, out=grid)
draw_pse(f'{save_dir}/transform.pse', ligand_rdmol, pocket_rdmol, ligand_grid, pocket_grid, \
        ligand_channels, pocket_channels, center, gmaker.get_resolution(), new_coords=new_coords)

