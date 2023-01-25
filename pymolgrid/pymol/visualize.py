import os
import subprocess
import sys
from time import sleep
from rdkit import Chem

import pymol
from pymol import cmd

from .dx import write_grid_to_dx_file
from .atom import ATOMSYMBOL

POCKET = 'pocket'
CARTOON = 'pocketCartoon'
LIGAND = 'ligand'

class PyMOLVisualizer() :
    def __init__(
        self,
        ligand_grid_color_dict = {},
        pocket_grid_color_dict = {},
    ) :
        self.ligand_grid_color_dict = ATOMSYMBOL.copy()
        self.ligand_grid_color_dict.update(ligand_grid_color_dict)

        self.pocket_grid_color_dict = ATOMSYMBOL.copy()
        self.pocket_grid_color_dict['C'] = 'myblue'
        self.pocket_grid_color_dict.update(pocket_grid_color_dict)

    def run(
        self,
        pse_path,
        ligand_rdmol,
        pocket_rdmol,
        ligand_grid_dict,
        pocket_grid_dict,
        center,
        resolution,
    ) :
        self.start_pymol()
        save_rdmol(ligand_rdmol, 'tmp_ligand.sdf')
        save_rdmol(pocket_rdmol, 'tmp_pocket.pdb')
        cmd.load('tmp_ligand.sdf')
        cmd.load('tmp_pocket.pdb')

        cmd.set_name('tmp_ligand', LIGAND)
        cmd.set_name('tmp_pocket', POCKET)
        cmd.copy(CARTOON, POCKET)

        cmd.hide('everything', 'all')

        cmd.show('sticks', POCKET)
        cmd.color('myblue', POCKET)
        
        cmd.show('cartoon', CARTOON)
        cmd.color('mylightblue', CARTOON)

        cmd.show('sticks', LIGAND)
        cmd.color('green', LIGAND)

        cmd.util.cnc('all')
        cmd.group('Structures', f'{POCKET} {CARTOON} {LIGAND}')

        ligand_dx_dict = []
        for key, grid in ligand_grid_dict.items() :
            write_grid_to_dx_file('tmp_grid.dx', grid, center, resolution)
            cmd.load('tmp_grid.dx')
            dx = 'ligand_' + key
            cmd.set_name('tmp_grid', dx)
            if key in self.ligand_grid_color_dict :
                cmd.color(self.ligand_grid_color_dict[key], dx)
            ligand_dx_dict.append(dx)
        cmd.group('LigandGrid', ' '.join(ligand_dx_dict))
        cmd.show('dots', 'LigandGrid')

        pocket_dx_dict = []
        for key, grid in pocket_grid_dict.items() :
            write_grid_to_dx_file('tmp_grid.dx', grid, center, resolution)
            cmd.load('tmp_grid.dx')
            dx = 'pocket_' + key
            cmd.set_name('tmp_grid', dx)
            if key in self.pocket_grid_color_dict :
                cmd.color(self.pocket_grid_color_dict[key], dx)
            pocket_dx_dict.append(dx)
        cmd.group('PocketGrid', ' '.join(pocket_dx_dict))
        cmd.show('dots', 'PocketGrid')

        cmd.enable('all')
        cmd.disable(CARTOON)
            
        self.set_final_representation()
        cmd.save(pse_path)
        os.remove('tmp_pocket.pdb')
        os.remove('tmp_ligand.sdf')
        os.remove('tmp_grid.dx')

    def set_initial_representations(self):
        """General settings for PyMOL"""
        cmd.set('depth_cue', 0)  # Turn off depth cueing (no fog)
        cmd.set('cartoon_side_chain_helper', 1)  # Improve combined visualization of sticks and cartoon
        cmd.set('cartoon_fancy_helices', 1)  # Nicer visualization of helices (using tapered ends)
        cmd.set('transparency_mode', 1)  # Turn on multilayer transparency
        cmd.set('dash_radius', 0.05)

        """Defines a colorset with matching colors. Provided by Joachim."""
        cmd.set_color('myblue', '[43, 131, 186]')
        cmd.set_color('mylightblue', '[158, 202, 225]')

        # Set clipping planes for full view
        cmd.clip('far', -1000)
        cmd.clip('near', 1000)

    def set_final_representation(self) :
        cmd.bg_color('black')
        cmd.set('dot_width', 2.0)

    def start_pymol(self, options='-pcq'):
        """Starts up PyMOL and sets general options. Quiet mode suppresses all PyMOL output.
        Command line options can be passed as the second argument."""
        pymol.pymol_argv = ['pymol', '%s' % options]
        pymol.finish_launching(args=['pymol', options, '-K'])
        cmd.reinitialize()
        """General settings for PyMOL"""
        self.set_initial_representations()
        cmd.feedback('disable', 'all', 'everything')

    def set_initial_representations(self):
        """General settings for PyMOL"""
        self.standard_settings()
        cmd.set('dash_gap', 0)  # Show not dashes, but lines for the pliprofiler
        cmd.set('ray_shadow', 0)  # Turn on ray shadows for clearer ray-traced images
        cmd.set('cartoon_color', 'mylightblue')

        # Set clipping planes for full view
        cmd.clip('far', -1000)
        cmd.clip('near', 1000)

    def standard_settings(self):
        """Sets up standard settings for a nice visualization."""
        cmd.set('bg_rgb', [1.0, 1.0, 1.0])  # White background
        cmd.set('depth_cue', 0)  # Turn off depth cueing (no fog)
        cmd.set('cartoon_side_chain_helper', 1)  # Improve combined visualization of sticks and cartoon
        cmd.set('cartoon_fancy_helices', 1)  # Nicer visualization of helices (using tapered ends)
        cmd.set('transparency_mode', 1)  # Turn on multilayer transparency
        cmd.set('dash_radius', 0.05)
        self.set_custom_colorset()

    @staticmethod
    def set_custom_colorset():
        """Defines a colorset with matching colors. Provided by Joachim."""
        cmd.set_color('myorange', '[253, 174, 97]')
        cmd.set_color('mygreen', '[171, 221, 164]')
        cmd.set_color('myred', '[215, 25, 28]')
        cmd.set_color('myblue', '[43, 131, 186]')
        cmd.set_color('mylightblue', '[158, 202, 225]')
        cmd.set_color('mylightgreen', '[229, 245, 224]')

def save_rdmol(rdmol, save_path, coords = None) :
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
