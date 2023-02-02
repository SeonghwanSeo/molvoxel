import os
import sys
from pymolgrid.voxelizer import Voxelizer
from pymolgrid.rdkit.wrapper import MolWrapper, MolSystemWrapper, ComplexWrapper
from pymolgrid.rdkit.imagemaker import MolImageMaker, MolSystemImageMaker
from pymolgrid.rdkit.getter import AtomTypeGetter, BondTypeGetter
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

save_dir = 'result_rdkit'
if '-y' in sys.argv :
    pymol = True
    from pymolgrid.pymol import Visualizer
    visualizer = Visualizer()
    os.system(f'mkdir -p {save_dir}')
else :
    pymol = False
    visualizer = None

voxelizer = Voxelizer(dimension = 32)

""" LOAD DATA """
ligand_path = './10gs/10gs_ligand.sdf'
protein_path = './10gs/10gs_protein_nowater.pdb'

ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
protein_rdmol = Chem.MolFromPDBFile(protein_path)

ligand_center = ligand_rdmol.GetConformer().GetPositions().mean(axis=0)

""" SINGLE MOL TEST """
rdmol = ligand_rdmol
center = ligand_center
atom_getter = AtomTypeGetter(['C', 'N', 'O', 'S'])
bond_getter = BondTypeGetter.default()

imagemaker = MolImageMaker(atom_getter, None, channel_type='types')
wrapper = MolWrapper(imagemaker, voxelizer, visualizer)
grid = wrapper.run(rdmol, center, radii=1.0)
if pymol :
    wrapper.visualize(f'{save_dir}/types.pse', ligand_rdmol, grid, center)

imagemaker = MolImageMaker(atom_getter, bond_getter, channel_type='types')
wrapper = MolWrapper(imagemaker, voxelizer, visualizer)
grid = wrapper.run(rdmol, center, radii=1.0)
if pymol :
    wrapper.visualize(f'{save_dir}/bond_types.pse', ligand_rdmol, grid, center)

imagemaker = MolImageMaker(atom_getter, None, channel_type='features')
wrapper = MolWrapper(imagemaker, voxelizer, visualizer)
grid = wrapper.run(rdmol, center, radii=1.0)
if pymol :
    wrapper.visualize(f'{save_dir}/features.pse', ligand_rdmol, grid, center)

imagemaker = MolImageMaker(atom_getter, bond_getter, channel_type='features')
wrapper = MolWrapper(imagemaker, voxelizer, visualizer)
grid = wrapper.run(rdmol, center, radii=1.0)
if pymol :
    wrapper.visualize(f'{save_dir}/bond_features.pse', ligand_rdmol, grid, center)

unknown_atom_getter = AtomTypeGetter(['C', 'N', 'O'], unknown=True)
imagemaker = MolImageMaker(unknown_atom_getter, bond_getter, channel_type='types')
wrapper = MolWrapper(imagemaker, voxelizer, visualizer)
grid = wrapper.run(rdmol, center, radii=1.0)
if pymol :
    wrapper.visualize(f'{save_dir}/unknownS.pse', ligand_rdmol, grid, center)

""" SYSTEM TEST """
rdmol_list = [ligand_rdmol, protein_rdmol]
center = ligand_center
atom_getter = AtomTypeGetter(['C', 'N', 'O', 'S'])
bond_getter = BondTypeGetter.default()

imagemaker = MolSystemImageMaker([atom_getter, None], [atom_getter, bond_getter], channel_type='types')
wrapper = MolSystemWrapper(imagemaker, voxelizer, ['Ligand', 'Protein'], visualizer)
grid = wrapper.run(rdmol_list, center, radii=1.0)
if pymol :
    wrapper.visualize(f'{save_dir}/system.pse', [ligand_rdmol, protein_rdmol], grid, center)

""" COMPLEX TEST """
imagemaker = MolSystemImageMaker([atom_getter, bond_getter], [atom_getter, None], channel_type='features')
wrapper = ComplexWrapper(imagemaker, voxelizer, visualizer)
grid = wrapper.run(ligand_rdmol, protein_rdmol, center, radii=1.0)
if pymol :
    wrapper.visualize(f'{save_dir}/complex.pse', ligand_rdmol, protein_rdmol, grid, center)

