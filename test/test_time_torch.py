import numpy as np
from rdkit import Chem
import time
from test_time_numpy import load_data, run_test_type, run_test_feature

def main_torch(device) :
    from pymolgrid.torch import Voxelizer
    batch_size = 16
    num_iteration = 25
    num_trial = 5
    resolution = 0.5
    dimension = 48

    coords, center, atom_types, atom_features, num_channels = load_data()
    voxelizer = Voxelizer(resolution, dimension, device=device)
    coords = voxelizer.asarray(coords, 'coords')
    center = voxelizer.asarray(center, 'center')
    atom_types = voxelizer.asarray(atom_types, 'type')
    atom_features = voxelizer.asarray(atom_features, 'feature')
    grid = voxelizer.get_empty_grid(num_channels, batch_size)

    """ sanity check """
    print('sanity check')
    for _ in range(2) :
        type_out = run_test_type(voxelizer, grid, coords, center, atom_types, 1.0, random_translation=0.0, random_rotation=False).clone()
        feature_out = run_test_feature(voxelizer, grid, coords, center, atom_features, 1.0, random_translation=0.0, random_rotation=False).clone()
        assert (type_out - type_out[0]).abs().less_(1e-5).all().item(), 'reproduction fail'
        assert (feature_out - feature_out[0]).abs().less_(1e-5).all().item(), 'reproduction fail'
        assert (type_out[0] - feature_out[0]).abs().less_(1e-5).all().item(), 'reproduction fail'
    print('pass\n')

    """ atom type """
    print('test atom type')
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

    """ atom feature """
    print('test atom feature')
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

if __name__ == '__main__' :
    main_torch('cpu')
