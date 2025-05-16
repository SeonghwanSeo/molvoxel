from test_time_numpy import main

if __name__ == "__main__":
    import sys

    from molvoxel.voxelizer.torch import Voxelizer

    if "-g" in sys.argv:
        device = "cuda"
    else:
        device = "cpu"

    resolution = 0.5
    dimension = 48
    voxelizer = Voxelizer(resolution, dimension, device=device)
    main(voxelizer)
