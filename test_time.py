import torch

from pymolgrid import grid_maker
from pymolgrid.pymol import write_grid_to_dx_file
from rdkit import Chem
import time

DATA = './1n4k/complex_ligand_pocket.pdb'

mol = Chem.MolFromPDBFile(DATA)
coords = torch.FloatTensor(mol.GetConformer().GetPositions())
center = coords.mean(dim=0)

atom_labels = torch.LongTensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])
print('Atom Set', set(atom_labels.tolist()))
type_vector = torch.zeros((mol.GetNumAtoms(), 6))
type_vector[atom_labels==6, 0] = 1
type_vector[atom_labels==7, 1] = 1
type_vector[atom_labels==8, 2] = 1
type_vector[atom_labels==15, 3] = 1
type_vector[atom_labels==16, 4] = 1
type_vector[[atom.GetIsAromatic() for atom in mol.GetAtoms()], 5] = 1

type_radii = torch.ones((6,), dtype=torch.float)
type_radii[3:] = 3.0
type_radii[5:] = 2.0
atom_radii = torch.ones((coords.size(0),))
atom_radii[100:] = 2.0

init_coords, init_center, init_type_vector = coords.clone(), center.clone(), type_vector.clone()

def run_test(gmaker, coords, center, type_vector, radii, random_translation = 0.0, random_rotation = False) :
    dims = gmaker.grid_dimension(6)
    out = torch.empty(dims)
    gmaker.forward_vector(coords, center, type_vector, radii, random_translation, random_rotation, out=out)
    return out

resolution = 0.5
dimension = 48
gmaker = grid_maker.GridMaker(resolution, dimension, gaussian_radius_multiple=-1.5)
for _ in range(10) :
    run_test(gmaker, coords, center, type_vector, 1.0)

gmaker = grid_maker.GridMaker(resolution, dimension, gaussian_radius_multiple=-1.5)
st = time.time()
for _ in range(1000) :
    out = run_test(gmaker, coords, center, type_vector, 1.0)
end = time.time()
print(out.view(out.size(0), -1).sum(dim=-1))
print(f'time{(end-st)}')
print('d1 ', gmaker.times_d1)
print('d2 ', gmaker.times_d - gmaker.times_d1)
print('c  ', gmaker.times_c)
print('ob ', gmaker.times_ob)
print('m  ', gmaker.times_m)
print('set', gmaker.times_set)
