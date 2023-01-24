import torch

import pymolgrid.grid_maker as grid_maker
from pymolgrid.pymol import write_grid_to_dx_file
from rdkit import Chem
import time

DATA = './1n4k/complex_ligand_pocket.pdb'
batch_size = 16
num_iteration = 25
num_trial = 5
resolution = 0.5
dimension = 48



mol = Chem.MolFromPDBFile(DATA)
coords = torch.FloatTensor(mol.GetConformer().GetPositions())
center = coords.mean(dim=0)

atom_labels = torch.LongTensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])
#print('Atom Set', set(atom_labels.tolist()))
type_vector = torch.zeros((mol.GetNumAtoms(), 6))
type_vector[atom_labels==6, 0] = 1
type_vector[atom_labels==7, 1] = 1
type_vector[atom_labels==8, 2] = 1
type_vector[atom_labels==15, 3] = 1
type_vector[atom_labels==16, 4] = 1
type_vector[[atom.GetIsAromatic() for atom in mol.GetAtoms()], 5] = 1

coords, center, type_vector, atom_labels = coords.cuda(), center.cuda(), type_vector.cuda(), atom_labels.cuda()

"""TYPE-VECTOR"""
print('Type Vector')
def run_test_vector(gmaker, out, coords, center, type_vector, radii, random_translation = 1.0, random_rotation = True) :
    for i in range(batch_size) :
        gmaker.forward_vector(coords, center, type_vector, radii, random_translation, random_rotation, out=out[i])
    return out

print('Sanity Check')
gmaker = grid_maker.GridMaker(resolution, dimension, gaussian_radius_multiple=-1.5, gpu=True)
dims = gmaker.grid_dimension(type_vector.size(1))
out = torch.empty((batch_size,) + dims, device='cuda')
st = time.time()
run_test_vector(gmaker, out, coords, center, type_vector, 1.0)
end = time.time()
print(f'time {(end-st)}')
print()

for i in range(num_trial) :
    print(f'trial {i}')
    st = time.time()
    for _ in range(num_iteration) :
        out = run_test_vector(gmaker, out, coords, center, type_vector, 1.0)
    end = time.time()
    print(f'total time {(end-st)}')
    print(f'time per run {(end-st) / batch_size / num_iteration}')
    print()


"""TYPE-INDEX"""
print('Type Index')
def run_test_index(gmaker, out, coords, center, atom_labels, radii, random_translation = 1.0, random_rotation = True) :
    for i in range(batch_size) :
        gmaker.forward_index(coords, center, atom_labels, radii, random_translation, random_rotation, out=out[i])
    return out

print('Sanity Check')
dims = gmaker.grid_dimension(atom_labels.max().item()+1)
out = torch.empty((batch_size,) + dims, device='cuda')
st = time.time()
run_test_index(gmaker, out, coords, center, atom_labels, 1.0)
end = time.time()
print(f'time {(end-st)}')
print()

for i in range(num_trial) :
    print(f'trial {i}')
    st = time.time()
    for _ in range(num_iteration) :
        out = run_test_index(gmaker, out, coords, center, atom_labels, 1.0)
    end = time.time()
    print(f'total time {(end-st)}')
    print(f'time per run {(end-st) / batch_size / num_iteration}')
    print()
