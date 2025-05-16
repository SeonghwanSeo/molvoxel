import sys

from test_run_numpy_single import main

if __name__ == "__main__":
    if "-y" in sys.argv:
        pymol = True
    else:
        pymol = False

    from molvoxel.voxelizer.numba import RandomTransform, Voxelizer

    main(Voxelizer, RandomTransform, pymol)
