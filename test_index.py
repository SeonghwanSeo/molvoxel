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

type_radii = torch.ones((17,), dtype=torch.float)
type_radii[11:] = 2.0
atom_radii = torch.ones((coords.size(0),))
atom_radii[100:] = 2.0

init_coords, init_center, init_atom_labels = coords.clone(), center.clone(), atom_labels.clone()

def run_test(gmaker, coords, center, atom_labels, radii, random_translation = 0.0, random_rotation = False) :
    dims = gmaker.grid_dimension(17)
    out = torch.empty(dims)
    st = time.time()
    gmaker.forward_index(coords, center, atom_labels, radii, random_translation, random_rotation, out=out)
    end = time.time()
    print(f'time per mol: {(end-st):.3f}')
    nochange =  ((coords - init_coords).abs().sum().item() < 1e-12) & \
                ((center - init_center).abs().sum().item() < 1e-12) & \
                ((atom_labels - init_atom_labels).abs().sum().item() < 1e-12)
    return out

resolution = 0.3
dimension = 64
gmaker = grid_maker.GridMaker(resolution, dimension)
print('Test 1: Binary: False, Raddi-Type-Index: False, Density: Mixed')
print('Test 1-1: No random transform')
ref_out = gmaker.forward_index(coords, center, atom_labels, radii=1.0)
for label in [6, 7, 8, 15] :
    write_grid_to_dx_file(f'dx_index/ref_{label}.dx', ref_out[label], center, resolution)

for _ in range(10) :
    out = run_test(gmaker, coords, center, atom_labels, 1.0)
    assert (out - ref_out).abs().sum().item() < 1e-12

print('Test 1-2: Random translation')
out = run_test(gmaker, coords, center, atom_labels, 1.0, random_translation = 1.0)

print('Test 1-3: Random rotation')
out = run_test(gmaker, coords, center, atom_labels, 1.0, random_rotation = True)

print('Test 1-3: Random transform')
out = run_test(gmaker, coords, center, atom_labels, 1.0, random_translation = 1.0, random_rotation = True)

print('Test 1-4: With Atom-wise Radii')
out = run_test(gmaker, coords, center, atom_labels, atom_radii)

print('Test 2: Binary: True, Radii-Type-Index: False')
gmaker.set_binary(True)
out = run_test(gmaker, coords, center, atom_labels, 1.0)

print('Test 3: Binary: True, Raddi-Type-Index: True')
gmaker.set_binary(True)
gmaker.set_radii_type_indexed(True)
out = run_test(gmaker, coords, center, atom_labels, type_radii)

print('Test 4: Binary: False, Raddi-Type-Index: False, Density: Gaussian')
gmaker = grid_maker.GridMaker(resolution, dimension, gaussian_radius_multiple=-1.5)
out = run_test(gmaker, coords, center, atom_labels, 1.0)

print('Test 5: Check Block Size Affect')
grid_maker.BLOCKDIM = 48
gmaker = grid_maker.GridMaker(resolution, dimension)
