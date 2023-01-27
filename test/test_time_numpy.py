from pymolgrid import Voxelizer
import numpy as np
from rdkit import Chem
import time

batch_size = 16
num_iteration = 25
num_trial = 10
resolution = 0.5
dimension = 48
voxelizer = Voxelizer(resolution, dimension)

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
num_ligand_channels = len(ligand_channels)
num_pocket_channels = len(pocket_channels)
num_channels = num_ligand_channels + num_pocket_channels
ligand_type_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
pocket_type_dict = {'C': 4, 'N': 5, 'O': 6, 'S': 7}
ligand_types = np.array([ligand_type_dict[atom.GetSymbol()] for atom in ligand_rdmol.GetAtoms()])
pocket_types = np.array([pocket_type_dict[atom.GetSymbol()] for atom in pocket_rdmol.GetAtoms()])
atom_types = np.concatenate([ligand_types, pocket_types], axis=0)

def get_features(atom, is_pocket) :
    res = [0.0] * 8
    idx = 4 if is_pocket else 0
    symbol = atom.GetSymbol()
    if symbol == 'C'        : res[idx+0] = 1.0
    elif symbol == 'N'      : res[idx+1] = 1.0
    elif symbol == 'O'      : res[idx+2] = 1.0
    elif symbol == 'S'      : res[idx+3] = 1.0
    return res
ligand_features = np.array([get_features(atom, False) for atom in ligand_rdmol.GetAtoms()])
pocket_features = np.array([get_features(atom, True) for atom in pocket_rdmol.GetAtoms()])
atom_features = np.concatenate([ligand_features, pocket_features], axis=0)

grid = voxelizer.get_empty_grid(num_channels, batch_size)

def run_test_type(voxelizer, grid, coords, center, atom_types, radii, random_translation = 0.5, random_rotation = True) :
    for i in range(batch_size) :
        voxelizer.forward_type(coords, center, atom_types, radii, random_translation, random_rotation, out=grid[i])
    return grid
def run_test_feature(voxelizer, grid, coords, center, atom_features, radii, random_translation = 0.5, random_rotation = True) :
    for i in range(batch_size) :
        voxelizer.forward_feature(coords, center, atom_features, radii, random_translation, random_rotation, out=grid[i])
    return grid

""" SANITY CHECK """
print('Sanity Check')
for _ in range(5) :
    type_out = run_test_type(voxelizer, grid, coords, center, atom_types, 1.0, random_translation=0.0, random_rotation=False).copy()
    feature_out = run_test_feature(voxelizer, grid, coords, center, atom_features, 1.0, random_translation=0.0, random_rotation=False).copy()
    assert np.less(np.abs(type_out - type_out[0]), 1e-5).all(), 'REPRODUCTION FAIL'
    assert np.less(np.abs(feature_out - feature_out[0]), 1e-5).all(), 'REPRODUCTION FAIL'
    assert np.less(np.abs(type_out[0] - feature_out[0]), 1e-5).all(), 'REPRODUCTION FAIL'
print('PASS\n')

""" ATOM TYPE """
print('Test Atom Type')
for _ in range(num_iteration) :
    grid = run_test_type(voxelizer, grid, coords, center, atom_types, 1.0)

st_tot = time.time()
for i in range(num_trial) :
    print(f'trial {i}')
    st = time.time()
    for _ in range(num_iteration) :
        grid = run_test_type(voxelizer, grid, coords, center, atom_types, 1.0)
    end = time.time()
    print(f'total time {(end-st)}')
    print(f'time per run {(end-st) / batch_size / num_iteration}')
    print()
end_tot = time.time()
print(f'times per run {(end_tot-st_tot) / batch_size / num_iteration / num_trial}\n')

""" ATOM FEATURE """
print('Test Atom Feature')
for _ in range(num_iteration) :
    grid = run_test_feature(voxelizer, grid, coords, center, atom_features, 1.0)

st_tot = time.time()
for i in range(num_trial) :
    print(f'trial {i}')
    st = time.time()
    for _ in range(num_iteration) :
        grid = run_test_feature(voxelizer, grid, coords, center, atom_features, 1.0)
    end = time.time()
    print(f'total time {(end-st)}')
    print(f'time per run {(end-st) / batch_size / num_iteration}')
    print()
end_tot = time.time()
print(f'times per run {(end_tot-st_tot) / batch_size / num_iteration / num_trial}')
