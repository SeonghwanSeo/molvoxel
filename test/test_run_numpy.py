import sys
import os
from pymolgrid.rdkit.wrapper import ComplexWrapper
from pymolgrid.rdkit.imagemaker import MolSystemImageMaker
from pymolgrid.rdkit.getter import AtomTypeGetter, BondTypeGetter, AtomFeatureGetter
import numpy as np
from rdkit import Chem

from utils import apply_coord

def main(Voxelizer, RandomTransform, pymol) :
    if pymol :
        from pymolgrid.pymol import Visualizer

    """ SET FUNCTION """
    def test(imagemaker, ligand_rdmol, protein_rdmol, channel_radii, atom_radii, save_dir) :
        if pymol :
            os.system(f'mkdir -p {save_dir}')
            visualizer = Visualizer()
        else :
            visualizer = None

        ligand_coords = ligand_rdmol.GetConformer().GetPositions()
        ligand_center = ligand_coords.mean(axis=0)
        center = ligand_center

        voxelizer = Voxelizer() #resolution=0.5, dimension=64, atom_scale=1.5, radii_type='scalar', density='gaussian'
        voxelizer_small = Voxelizer(0.5, 16, blockdim = 16)
        voxelizer_hr = Voxelizer(0.4, 64)

        transform = RandomTransform(random_translation=0.5, random_rotation=True)

        wrapper = ComplexWrapper(imagemaker, voxelizer, visualizer)
        wrapper_small = ComplexWrapper(imagemaker, voxelizer_small, visualizer)
        wrapper_hr = ComplexWrapper(imagemaker, voxelizer_hr, visualizer)

        grid = wrapper.get_empty_grid()

        print('Test 1: Binary: False, Channel-Wise Radii: False, Density: Gaussian (Default)')
        test_name = 'ref'
        ref_grid = wrapper.run(ligand_rdmol, protein_rdmol, center, radii=1.0)
        out_grid = wrapper.run(ligand_rdmol, protein_rdmol, center, radii=1.0, out=grid)
        assert grid is out_grid, 'INPLACE FAILE'
        assert np.less(np.abs(np.subtract(grid.tolist(), ref_grid.tolist())), 1e-5).all(), 'REPRODUCTION FAIL'
        if pymol :
            wrapper.visualize(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, grid, center)

        print('Test 2: Small (One Block)')
        test_name = 'small'
        out_grid = wrapper_small.run(ligand_rdmol, protein_rdmol, center, radii=1.0)
        if pymol :
            wrapper_small.visualize(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, out_grid, center)

        print('Test 3: High Resolution')
        test_name = 'hr'
        out_grid = wrapper_hr.run(ligand_rdmol, protein_rdmol, center, radii=1.0)
        if pymol :
            wrapper_hr.visualize(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, out_grid, center)

        print('Test 4: Channel-Wise Radii: True')
        test_name = 'channel-wise'
        voxelizer.radii_type = 'channel-wise'
        wrapper.run(ligand_rdmol, protein_rdmol, center, channel_radii, out = grid)
        if pymol :
            wrapper.visualize(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, grid, center)

        print('Test 5: Atom-wise Radii: True')
        test_name = 'atom-wise'
        voxelizer.radii_type = 'atom-wise'
        wrapper.run(ligand_rdmol, protein_rdmol, center, atom_radii, out = grid)
        if pymol :
            wrapper.visualize(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, grid, center)

        print('Test 6: Density: Binary')
        test_name = 'binary'
        voxelizer.density = 'binary'
        voxelizer.radii_type = 'scalar'
        wrapper.run(ligand_rdmol, protein_rdmol, center, radii=1.0, out = grid)
        if pymol :
            wrapper.visualize(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, grid, center)

        print('Test 7: Random Transform')
        test_name = 'transform'
        voxelizer.density = 'gaussian'
        protein_coords = protein_rdmol.GetConformer().GetPositions()
        T = transform.get_transform()
        new_ligand_coords, new_protein_coords = T(ligand_coords, center), T(protein_coords, center)
        ligand_rdmol, protein_rdmol = apply_coord(ligand_rdmol, new_ligand_coords), apply_coord(protein_rdmol, new_protein_coords)
        wrapper.run(ligand_rdmol, protein_rdmol, center, radii=1.0, out = grid)
        if pymol :
            wrapper.visualize(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, grid, center)

    """ LOAD DATA """
    ligand_path = './10gs/10gs_ligand.sdf'
    protein_path = './10gs/10gs_protein_nowater.pdb'

    ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
    protein_rdmol = Chem.MolFromPDBFile(protein_path)

    """ Atom Types """
    print('# Test Atom Type #')
    save_dir = 'result_type'

    atom_getter = AtomTypeGetter(['C', 'N', 'O', 'S'])
    bond_getter = BondTypeGetter([Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.AROMATIC])
    imagemaker = MolSystemImageMaker([atom_getter, bond_getter], [atom_getter, bond_getter], channel_type='types')

    channel_radii = np.ones((imagemaker.num_channels,))
    channel_radii[:4] = 2.0

    num_atoms = ligand_rdmol.GetNumAtoms() + ligand_rdmol.GetNumBonds() + protein_rdmol.GetNumAtoms() + protein_rdmol.GetNumBonds()
    atom_radii = np.ones((num_atoms,))
    atom_radii[:ligand_rdmol.GetNumAtoms()] = 2.0

    test(imagemaker, ligand_rdmol, protein_rdmol, channel_radii, atom_radii, save_dir)

    """ Vector """
    print('# Test Atom Feature #')
    save_dir = 'result_feature'
    channels = ['C', 'N', 'O', 'S', 'Aromatic']
    def get_features(atom) :
        symbol_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
        res = [0] * 5
        symbol = atom.GetSymbol()
        res[symbol_dict[symbol]] = 1
        if atom.GetIsAromatic() : res[4] = 1
        return res

    atom_getter = AtomFeatureGetter(get_features, channels)
    bond_getter = BondTypeGetter([Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.AROMATIC])
    imagemaker = MolSystemImageMaker([atom_getter, None], [atom_getter, bond_getter], channel_type='features')

    channel_radii = np.ones((imagemaker.num_channels,))
    channel_radii[[4, 9]] = 2.0

    num_atoms = ligand_rdmol.GetNumAtoms() + protein_rdmol.GetNumAtoms() + protein_rdmol.GetNumBonds()
    atom_radii = np.ones((num_atoms,))
    atom_radii[ -protein_rdmol.GetNumBonds() : ] = 2.0

    test(imagemaker, ligand_rdmol, protein_rdmol, channel_radii, atom_radii, save_dir)

if __name__ == '__main__' :
    if '-y' in sys.argv :
        pymol = True
    else :
        pymol = False

    from pymolgrid.voxelizer import Voxelizer, RandomTransform
    main(Voxelizer, RandomTransform, pymol)
