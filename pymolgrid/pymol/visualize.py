import os
from rdkit import Chem
import tempfile
from pathlib import Path

from rdkit.Chem import Mol
from numpy.typing import ArrayLike
from typing import Dict

import pymol
from pymol import cmd

from .dx import write_grid_to_dx_file
from .atom import ATOMSYMBOL

PROTEIN = 'Protein'
CARTOON = 'Cartoon'
LIGAND = 'Ligand'
MOLECULE = 'Molecule'

ligand_grid_color_dict = ATOMSYMBOL.copy()
ligand_grid_color_dict.update(ligand_grid_color_dict)
molecule_grid_color_dict = ligand_grid_color_dict

protein_grid_color_dict = ATOMSYMBOL.copy()
protein_grid_color_dict['C'] = 'aqua'
protein_grid_color_dict.update(protein_grid_color_dict)

def __launch_pymol() :
    pymol.pymol_argv = ['pymol', '-pcq']
    pymol.finish_launching(args=['pymol', '-pcq', '-K'])
    cmd.reinitialize()
    cmd.feedback('disable', 'all', 'everything')

def visualize_mol(
    pse_path: str,
    rdmol: Mol,
    grid_dict: Dict[str, ArrayLike],
    center: ArrayLike,
    resolution: float,
) :
    __launch_pymol()
    cmd.set_color('aqua', '[0, 150, 255]')

    temp_dir = tempfile.TemporaryDirectory()
    temp_dirpath = Path(temp_dir.name)
    temp_mol_path = str(temp_dirpath / f'{MOLECULE}.sdf')
    temp_grid_path = str(temp_dirpath / 'grid.dx')

    __save_rdmol(rdmol, temp_mol_path)
    cmd.load(temp_mol_path)
    cmd.color('green', MOLECULE)

    dx_dict = []
    for key, grid in grid_dict.items() :
        write_grid_to_dx_file(temp_grid_path, grid, center, resolution)
        cmd.load(temp_grid_path)
        dx = key
        cmd.set_name('grid', dx)
        if key in molecule_grid_color_dict :
            cmd.color(molecule_grid_color_dict[key], dx)
        dx_dict.append(dx)
    cmd.group('Voxel', ' '.join(dx_dict))

    temp_dir.cleanup()

    cmd.enable('all')

    cmd.hide('everything', 'all')
    cmd.show('sticks', MOLECULE)
    cmd.show('everything', 'Voxel')
    cmd.util.cnc('all')

    cmd.bg_color('black')
    cmd.set('dot_width', 2.5)

    cmd.save(pse_path)

def visualize_complex(
    pse_path: str,
    ligand_rdmol: Mol,
    protein_rdmol: Mol,
    ligand_grid_dict: Dict[str, ArrayLike],
    protein_grid_dict: Dict[str, ArrayLike],
    center: ArrayLike,
    resolution: str,
) :
    __launch_pymol()
    cmd.set_color('aqua', '[0, 150, 255]')

    temp_dir = tempfile.TemporaryDirectory()
    temp_dirpath = Path(temp_dir.name)
    temp_ligand_path = str(temp_dirpath / f'{LIGAND}.sdf')
    temp_protein_path = str(temp_dirpath / f'{PROTEIN}.pdb')
    temp_grid_path = str(temp_dirpath / 'grid.dx')

    __save_rdmol(ligand_rdmol, temp_ligand_path)
    cmd.load(temp_ligand_path)
    cmd.color('green', LIGAND)

    __save_rdmol(protein_rdmol, temp_protein_path)
    cmd.load(temp_protein_path)
    cmd.copy(CARTOON, PROTEIN)
    cmd.color('aqua', PROTEIN)
    cmd.color('cyan', CARTOON)

    cmd.group('Molecule', f'{LIGAND} {PROTEIN} {CARTOON}')

    ligand_dx_dict = []
    for key, grid in ligand_grid_dict.items() :
        write_grid_to_dx_file(temp_grid_path, grid, center, resolution)
        cmd.load(temp_grid_path)
        dx = 'Ligand_' + key
        cmd.set_name('grid', dx)
        if key in ligand_grid_color_dict :
            cmd.color(ligand_grid_color_dict[key], dx)
        ligand_dx_dict.append(dx)
    cmd.group('LigandVoxel', ' '.join(ligand_dx_dict))

    protein_dx_dict = []
    for key, grid in protein_grid_dict.items() :
        write_grid_to_dx_file(temp_grid_path, grid, center, resolution)
        cmd.load(temp_grid_path)
        dx = 'Protein_' + key
        cmd.set_name('grid', dx)
        if key in protein_grid_color_dict :
            cmd.color(protein_grid_color_dict[key], dx)
        protein_dx_dict.append(dx)
    cmd.group('ProteinVoxel', ' '.join(protein_dx_dict))

    temp_dir.cleanup()

    cmd.enable('all')

    cmd.hide('everything', 'all')
    cmd.show('sticks', LIGAND)
    cmd.show('sticks', PROTEIN)
    cmd.show('cartoon', CARTOON)
    cmd.show('everything', 'LigandVoxel')
    cmd.show('everything', 'ProteinVoxel')
    cmd.util.cnc('all')

    cmd.disable(CARTOON)
    cmd.bg_color('black')
    cmd.set('dot_width', 2.5)

    cmd.save(pse_path)
            
def __save_rdmol(rdmol, save_path, coords = None) :
    rdmol = Chem.Mol(rdmol)
    if coords is not None :
        conf = rdmol.GetConformer()
        for i in range(rdmol.GetNumAtoms()) :
            conf.SetAtomPosition(i, coords[i].tolist())

    ext = os.path.splitext(save_path)[-1]
    assert ext in ['.pdb', '.sdf']
    if ext == '.pdb' :
        w = Chem.PDBWriter(save_path)
    else :
        w = Chem.SDWriter(save_path)
    w.write(rdmol)
    w.close()
