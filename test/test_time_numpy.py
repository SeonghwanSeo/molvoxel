from pymolgrid import GridMaker
import numpy as np
from rdkit import Chem
import time

batch_size = 16
num_iteration = 25
num_trial = 5
resolution = 0.5
dimension = 48
gmaker = GridMaker(resolution, dimension, gaussian_radius_multiple=-1.5)

""" LOAD DATA """
ligand_path = './10gs/ligand.sdf'
pocket_path = './10gs/pocket.pdb'

ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
pocket_rdmol = Chem.MolFromPDBFile(pocket_path)

ligand_coords = ligand_rdmol.GetConformer().GetPositions()
ligand_center = ligand_coords.mean(axis=0)
pocket_coords = pocket_rdmol.GetConformer().GetPositions()
coords = np.concatenate([ligand_coords, pocket_coords], axis=0)
center = ligand_center

ligand_channels = ['C', 'N', 'O', 'S']
pocket_channels = ['C', 'N', 'O', 'S']
num_ligand_types = len(ligand_channels)
num_pocket_types = len(pocket_channels)
num_types = num_ligand_types + num_pocket_types
ligand_type_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
pocket_type_dict = {'C': 4, 'N': 5, 'O': 6, 'S': 7}
ligand_types = np.array([ligand_type_dict[atom.GetSymbol()] for atom in ligand_rdmol.GetAtoms()])
pocket_types = np.array([pocket_type_dict[atom.GetSymbol()] for atom in pocket_rdmol.GetAtoms()])
type_index = np.concatenate([ligand_types, pocket_types], axis=0)

def get_vector(atom, is_pocket) :
    res = [0] * 8
    idx = 4 if is_pocket else 0
    symbol = atom.GetSymbol()
    if symbol == 'C'        : res[idx+0] = 1
    elif symbol == 'N'      : res[idx+1] = 1
    elif symbol == 'O'      : res[idx+2] = 1
    elif symbol == 'S'      : res[idx+3] = 1
    return res
ligand_types = np.array([get_vector(atom, False) for atom in ligand_rdmol.GetAtoms()])
pocket_types = np.array([get_vector(atom, True) for atom in pocket_rdmol.GetAtoms()])
type_vector = np.concatenate([ligand_types, pocket_types], axis=0)

grid = np.empty((batch_size,) + gmaker.grid_dimension(num_types))

def run_test_index(gmaker, grid, coords, center, type_index, radii, random_translation = 0.5, random_rotation = True) :
    for i in range(batch_size) :
        gmaker.forward_index(coords, center, type_index, radii, random_translation, random_rotation, out=grid[i])
    return grid
def run_test_vector(gmaker, grid, coords, center, type_vector, radii, random_translation = 0.5, random_rotation = True) :
    for i in range(batch_size) :
        gmaker.forward_vector(coords, center, type_vector, radii, random_translation, random_rotation, out=grid[i])
    return grid

""" SANITY CHECK """
print('Sanity Check')
for _ in range(5) :
    index_out = run_test_index(gmaker, grid, coords, center, type_index, 1.0, random_translation=0.0, random_rotation=False).copy()
    vector_out = run_test_vector(gmaker, grid, coords, center, type_vector, 1.0, random_translation=0.0, random_rotation=False).copy()
    assert np.less(np.abs(index_out - index_out[0]), 1e-5).all(), 'REPRODUCTION FAIL'
    assert np.less(np.abs(vector_out - vector_out[0]), 1e-5).all(), 'REPRODUCTION FAIL'
    assert np.less(np.abs(index_out[0] - vector_out[0]), 1e-5).all(), 'REPRODUCTION FAIL'
print('PASS\n')

"""TYPE-INDEX"""
print('Test Type Index')
for i in range(num_trial) :
    print(f'trial {i}')
    st = time.time()
    for _ in range(num_iteration) :
        grid = run_test_index(gmaker, grid, coords, center, type_index, 1.0)
    end = time.time()
    print(f'total time {(end-st)}')
    print(f'time per run {(end-st) / batch_size / num_iteration}')
    print()

"""TYPE-VECTOR"""
print('Test Type Vector')
for i in range(num_trial) :
    print(f'trial {i}')
    st = time.time()
    for _ in range(num_iteration) :
        grid = run_test_vector(gmaker, grid, coords, center, type_vector, 1.0)
    end = time.time()
    print(f'total time {(end-st)}')
    print(f'time per run {(end-st) / batch_size / num_iteration}')
    print()
