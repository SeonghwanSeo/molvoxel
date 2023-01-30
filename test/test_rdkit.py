import os
from pymolgrid import Voxelizer
from pymolgrid.rdkit.binding import RDMolWrapper
from pymolgrid.rdkit.getter import RDMolChannelGetter, RDAtomChannelGetter, RDBondChannelGetter
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
try :
    from pymolgrid.pymol import visualize_mol, visualize_complex
    assert sys.argv[1] == '-y'
    pymol = True
    os.system('mkdir -p result_rdkit')
except :
    pymol = False

voxelizer = Voxelizer(resolution=0.5, dimension=48, atom_scale=1.5, density='gaussian', \
                    channel_wise_radii=False) # Default

def atom_function(atom) :
    dic = {6: 0, 7: 1, 8: 2, 16: 3}
    res = [0] * 5
    res[dic[atom.GetAtomicNum()]] = 1
    if atom.GetIsAromatic() :
        res[4] = 1
    return res
atom_getter = RDAtomChannelGetter(atom_function, ['C', 'N', 'O', 'S', 'Aromatic'])
bond_getter = RDBondChannelGetter.default()
mol_getter = RDMolChannelGetter(atom_getter, bond_getter, prefix='')
wrapper = RDMolWrapper([mol_getter])
voxelizer.decorate(wrapper)

""" LOAD DATA """
ligand_path = './10gs/ligand.sdf'
pocket_path = './10gs/pocket.pdb'

ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
pocket_rdmol = Chem.MolFromPDBFile(pocket_path)

""" SINGLE MOL TEST """
grids, inputs = voxelizer.run([ligand_rdmol], return_inputs=True)
grid_dict = voxelizer.wrapper.split_channel(grids)[0]
if pymol :
    visualize_mol('result_rdkit/ligand.pse', ligand_rdmol, grid_dict, inputs['center'], voxelizer.resolution)

""" COMPLEX TEST """
ligand_getter = RDMolChannelGetter(atom_getter, bond_getter, prefix='')
pocket_getter = RDMolChannelGetter(atom_getter, bond_getter, prefix='')
wrapper = RDMolWrapper([ligand_getter, pocket_getter])
voxelizer.decorate(wrapper)

grids, inputs = voxelizer.run([ligand_rdmol, pocket_rdmol], return_inputs=True)
ligand_grid_dict, pocket_grid_dict = voxelizer.wrapper.split_channel(grids)

if pymol :
    visualize_complex('result_rdkit/complex.pse', ligand_rdmol, pocket_rdmol, ligand_grid_dict, pocket_grid_dict, center, 0.5)
