import torch

import pymolgrid.grid_maker as grid_maker
from rdkit import Chem
import time

ligand_path = './10gs/ligand.sdf'
pocket_path = './10gs/pocket.pdb'
batch_size = 16
num_iteration = 25
num_trial = 5
resolution = 0.5
dimension = 48

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

type_vector = torch.zeros(coords.size(0), num_types)
ligand_type_vector = type_vector[:ligand_rdmol.GetNumAtoms()]
pocket_type_vector = type_vector[ligand_rdmol.GetNumAtoms():]
ligand_type_vector[ligand_types==6, 0] = 1
ligand_type_vector[ligand_types==7, 1] = 1
ligand_type_vector[ligand_types==8, 2] = 1
ligand_type_vector[ligand_types==16, 3] = 1
pocket_type_vector[pocket_types==6, 4] = 1
pocket_type_vector[pocket_types==7, 5] = 1
pocket_type_vector[pocket_types==8, 6] = 1
pocket_type_vector[pocket_types==16, 7] = 1

"""START"""
gmaker = grid_maker.GridMaker(resolution, dimension, gaussian_radius_multiple=-1.5, device='cuda')
dims = gmaker.grid_dimension(type_vector.size(1))
grid = torch.empty((batch_size,) + dims, device='cuda')

"""SET DEVICE"""
device = 'cuda'
coords, type_index, type_vector, center = \
        coords.to(device), type_index.to(device), type_vector.to(device), center.to(device)
#grid = grid.to(device)
#gmaker.cuda()

"""TYPE-VECTOR"""
def run_test_vector(gmaker, grid, coords, center, type_vector, radii, random_translation = 1.0, random_rotation = True) :
    for i in range(batch_size) :
        gmaker.forward_vector(coords, center, type_vector, radii, random_translation, random_rotation, out=grid[i])
    return grid

print('Sanity Check')
st = time.time()
run_test_vector(gmaker, grid, coords, center, type_vector, 1.0)
end = time.time()
print(f'time {(end-st)}')
print()

for i in range(num_trial) :
    print(f'trial {i}')
    st = time.time()
    for _ in range(num_iteration) :
        grid = run_test_vector(gmaker, grid, coords, center, type_vector, 1.0)
    end = time.time()
    print(f'total time {(end-st)}')
    print(f'time per run {(end-st) / batch_size / num_iteration}')
    print()

"""TYPE-INDEX"""
print('Type Index')
def run_test_index(gmaker, grid, coords, center, type_index, radii, random_translation = 1.0, random_rotation = True) :
    for i in range(batch_size) :
        gmaker.forward_index(coords, center, type_index, radii, random_translation, random_rotation, out=grid[i])
    return grid

print('Sanity Check')
st = time.time()
run_test_index(gmaker, grid, coords, center, type_index, 1.0)
end = time.time()
print(f'time {(end-st)}')
print()

for i in range(num_trial) :
    print(f'trial {i}')
    st = time.time()
    for _ in range(num_iteration) :
        grid = run_test_index(gmaker, grid, coords, center, type_index, 1.0)
    end = time.time()
    print(f'total time {(end-st)}')
    print(f'time per run {(end-st) / batch_size / num_iteration}')
    print()
