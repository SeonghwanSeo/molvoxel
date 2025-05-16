# for better typing
from .base import BaseRandomTransform as RandomTransform
from .base import BaseVoxelizer as Voxelizer


def create_random_transform(
    random_translation: float = 0.0,
    random_rotation: bool = False,
    library: str = "numpy",
    **kwargs,
) -> RandomTransform:
    assert library in ["numba", "numpy", "torch"]
    if library == "numba":
        from .numba import RandomTransform as TypeRandomTransform
    elif library == "numpy":
        from .numpy import RandomTransform as TypeRandomTransform
    else:
        from .torch import RandomTransform as TypeRandomTransform
    return TypeRandomTransform(random_translation, random_rotation, **kwargs)


def create_voxelizer(
    resolution: float = 0.5,
    dimension: int = 64,
    radii_type: str = "scalar",
    density_type: str = "gaussian",
    library: str = "numpy",
    **kwargs,
) -> Voxelizer:
    assert library in ["numba", "numpy", "torch"]
    if library == "numba":
        from .numba import Voxelizer as TypeVoxelizer
    elif library == "numpy":
        from .numpy import Voxelizer as TypeVoxelizer
    else:
        from .torch import Voxelizer as TypeVoxelizer
    return TypeVoxelizer(resolution, dimension, radii_type, density_type, **kwargs)
