import torch
import os

from pymolgrid import grid_maker
from pymolgrid.pymol import write_grid_to_dx_file
from rdkit import Chem
import time

os.system('mkdir -p result_vector')
save_dir = 'result_vector'

def run_test(gmaker, coords, center, type_vector, radii, random_translation = 0.0, random_rotation = False) :
    dims = gmaker.grid_dimension(6)
    out = torch.empty(dims)
    gmaker.forward_vector(coords, center, type_vector, radii, random_translation, random_rotation, out=out)
    return out

def save_dx(out, center, resolution, prefix) :
    for idx, label in enumerate(['C','N','O','P','S','Ar']) :
        write_grid_to_dx_file(f'{save_dir}/{prefix}_{label}.dx', out[idx], center, resolution)

def save_mol(mol, coords, prefix) :
    mol = Chem.Mol(mol)
    conformer = mol.GetConformer()
    for i in range(mol.GetNumAtoms()) :
        conformer.SetAtomPosition(i, coords[i].tolist())
    w = Chem.PDBWriter(f'{save_dir}/{prefix}.pdb')
    w.write(mol)
    w.close()

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

print('Test 1: Binary: False, Raddi-Type-Index: False, Density: Mixed')
resolution, dimension = 0.5, 48
gmaker = grid_maker.GridMaker(resolution, dimension)
ref_out = gmaker.forward_vector(coords, center, type_vector, radii=1.0)
save_dx(ref_out, center, resolution, 'ref')
save_mol(mol, coords, 'ref')

print('Test 2: High Resolution')
gmaker_hr = grid_maker.GridMaker(0.3, 64)
hr_out = run_test(gmaker, coords, center, type_vector, 1.0)
save_dx(hr_out, center, 0.3, 'hr')

print('Test 3: With Atom-wise Radii')
out = run_test(gmaker, coords, center, type_vector, atom_radii)
save_dx(out, center, resolution, 'test3')

print('Test 4: Binary: True, Radii-Type-Index: False')
gmaker.set_binary(True)
out = run_test(gmaker, coords, center, type_vector, 1.0)
save_dx(out, center, resolution, 'test4')

print('Test 5: Binary: True, Raddi-Type-Index: True')
gmaker.set_binary(True)
gmaker.set_radii_type_indexed(True)
out = run_test(gmaker, coords, center, type_vector, type_radii)
save_dx(out, center, resolution, 'test5')

print('Test 6: Binary: False, Raddi-Type-Index: False, Density: Gaussian')
gmaker_gaus = grid_maker.GridMaker(gaussian_radius_multiple=-1.5)
out = run_test(gmaker_gaus, coords, center, type_vector, 1.0)
save_dx(out, center, resolution, 'test6')

gmaker.set_binary(False)
gmaker.set_radii_type_indexed(False)

print('Test 7-1: Random translation')
new_coords = gmaker.do_transform(coords, center, random_translation=1.0)
out = run_test(gmaker, new_coords, center, type_vector, 1.0)
save_dx(out, center, resolution, 'test7-1')
save_mol(mol, new_coords, 'test7-1')

print('Test 7-2: Random rotation')
new_coords = gmaker.do_transform(coords, center, random_rotation=True)
out = run_test(gmaker, new_coords, center, type_vector, 1.0)
save_dx(out, center, resolution, 'test7-2')
save_mol(mol, new_coords, 'test7-2')

print('Test 7-3: Random transform')
new_coords = gmaker.do_transform(coords, center, random_translation=1.0, random_rotation=True)
out = run_test(gmaker, new_coords, center, type_vector, 1.0)
save_dx(out, center, resolution, 'test7-3')
save_mol(mol, new_coords, 'test7-3')
