# MolVoxel: Molecular Voxelization Tool
MolVoxel is an easy-to-use **Molecular Voxelization Tool** implemented in Python.

It requires few dependencies, so you can install it very simply. If you want to use numba version, just install numba.

### Dependencies

- Required
  - Numpy
- Optional
  - SciPy - `from molvoxel.voxelizer.numpy import Voxelizer`
  - Numba - `from molvoxel.voxelizer.numba import Voxelizer`
  - PyTorch - `from molvoxel.voxelizer.torch import Voxelizer`, **CUDA Available**
  - RDKit, pymol-open-source

## Quick Start

### Create Voxelizer Object
```python
import molvoxel

# Default (Gaussian sigma = 1/3)
voxelizer = molvoxel.create_voxelizer(resolution=0.5, dimension=64, density_type='gaussian', library='numpy')
# Set gaussian sigma = 0.5, spatial dimension = (48, 48, 48)
voxelizer = molvoxel.create_voxelizer(dimension=48, density_type='gaussian', sigma=0.5, library='numba')
# Set binary density
voxelizer = molvoxel.create_voxelizer(density_type='binary', library='torch')
# CUDA
voxelizer = molvoxel.create_voxelizer(library='torch', device='cuda')
```

### Create Voxel
#### Numpy, Numba

```python
from rdkit import Chem  # rdkit is not required packages
import numpy as np

def get_atom_features(atom) :
  symbol, aromatic = atom.GetSymbol(), atom.GetIsAromatic()
  return [symbol == 'C', symbol == 'N', symbol == 'O', symbol == 'S', aromatic]

mol = Chem.SDMolSupplier('test/10gs/ligand.sdf')[0]
channels = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
coords = mol.GetConformer().GetPositions()                  # (V, 3)
center = coords.mean(axis = 0)                              # (3,)
atom_types = [channels[atom.GetSymbol()] for atom in mol.GetAtoms()]
atom_types = np.array(atom_types)                           # (V,)
atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
atom_features = np.array(atom_features)                     # (V, 5)
radii = 1.0

image = voxelizer.forward_single(coords, center, radii)                   # (1, 64, 64, 64)
image = voxelizer.forward_types(coords, center, atom_types, radii)        # (4, 64, 64, 64)
image = voxelizer.forward_features(coords, center, atom_features, radii)  # (5, 64, 64, 64)
```

#### PyTorch - Cuda Available

```python
# PyTorch is required
import torch

device = 'cuda' # or 'cpu'
coords = torch.FloatTensor(coords).to(device)               # (V, 3)
center = torch.FloatTensor(center).to(device)               # (3,)
atom_types = torch.LongTensor(atom_types).to(device)        # (V,)
atom_features = torch.FloatTensor(atom_features).to(device) # (V, 5)

image = voxelizer.forward_single(coords, center, radii)                   # (1, 64, 64, 64)
image = voxelizer.forward_types(coords, center, atom_types, radii)        # (4, 32, 32, 32)
image = voxelizer.forward_features(coords, center, atom_features, radii)  # (5, 32, 32, 32)
```
---

## Installation

```shell
# Required: numpy
# Not Required, but Recommended: RDKit
# Optional - Numpy: scipy
# Optional - Numba: numba
# Optional - PyTorch : torch
# Optional - Visualization : pymol-open-source (conda)
git clone https://github.com/SeonghwanSeo/molvoxel.git
cd molvoxel/
pip install -e .

# With extras_require
# pip install -e '.[numpy, numba, torch, rdkit]'
```

## RDKit Wrapper
``` python
# RDKit is required
from molvoxel.rdkit import AtomTypeGetter, BondTypeGetter, MolPointCloudMaker, MolWrapper
atom_getter = AtomTypeGetter(['C', 'N', 'O', 'S'])
bond_getter = BondTypeGetter.default()		# (SINGLE, DOUBLE, TRIPLE, AROMATIC)

pointcloudmaker = MolPointCloudMaker(atom_getter, bond_getter, channel_type='types')
wrapper = MolWrapper(pointcloudmaker, voxelizer, visualizer)
image = wrapper.run(rdmol, center, radii=1.0)
```
