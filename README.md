[![pypi](https://img.shields.io/pypi/v/molvoxel.svg?logo=pypi)](https://pypi.org/project/molvoxel/)
![versions](https://img.shields.io/pypi/pyversions/molvoxel.svg?logo=python)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# MolVoxel: Molecular Voxelization Tool

MolVoxel is an Easy-to-Use **Molecular Voxelization Tool** implemented in Python.

It requires minimal dependencies, so it's very simple to install and use. If you want to use numba version, just install numba additionally.

### Dependencies

- Required
  - Numpy, SciPy
- Optional
  - Numba
  - PyTorch, **CUDA Available**
  - RDKit, pymol-open-source

### Citation

```
@article{seo2023pharmaconet,
  title = {PharmacoNet: Accelerating Large-Scale Virtual Screening by Deep Pharmacophore Modeling},
  author = {Seo, Seonghwan and Kim, Woo Youn},
  journal = {arXiv preprint arXiv:2310.00681},
  year = {2023},
  url = {https://arxiv.org/abs/2310.00681},
}
```



## Quick Start

### Installation

```shell
pip install molvoxel
pip install molvoxel[numba, torch, rdkit]	# Optional Dependencies
```

### Configuring Voxelizer Object
```python
import molvoxel
# Default (Resolution: 0.5, dimension: 64, density_type: gaussian, sigma: 0.5, library='numpy')
voxelizer = molvoxel.create_voxelizer()
# Set gaussian sigma = 1.0, spatial dimension = (48, 48, 48) with numba library
voxelizer = molvoxel.create_voxelizer(dimension=48, density_type='gaussian', sigma=1.0, library='numba')
# Set binary density with torch library
voxelizer = molvoxel.create_voxelizer(density_type='binary', library='torch')
# CUDA
voxelizer = molvoxel.create_voxelizer(library='torch', device='cuda')
```

### Voxelization
#### Numpy, Numba

```python
from rdkit import Chem  # rdkit is not required packages
import numpy as np

def get_atom_features(atom):
    symbol, aromatic = atom.GetSymbol(), atom.GetIsAromatic()
    return [symbol == 'C', symbol == 'N', symbol == 'O', symbol == 'S', aromatic]

mol = Chem.SDMolSupplier('./test/10gs/10gs_ligand.sdf')[0]
channels = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
coords = mol.GetConformer().GetPositions()                                      # (V, 3)
center = coords.mean(axis=0)                                                    # (3,)
atom_types = np.array([channels[atom.GetSymbol()] for atom in mol.GetAtoms()])  # (V,)
atom_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])  # (V, 5)
atom_radius = 1.0                                                               # scalar

image = voxelizer.forward_single(coords, center, atom_radius)                   # (1, 64, 64, 64)
image = voxelizer.forward_types(coords, center, atom_types, atom_radius)        # (4, 64, 64, 64)
image = voxelizer.forward_features(coords, center, atom_features, atom_radius)  # (5, 64, 64, 64)
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

image = voxelizer.forward_single(coords, center, atom_radius)                   # (1, 64, 64, 64)
image = voxelizer.forward_types(coords, center, atom_types, atom_radius)        # (4, 32, 32, 32)
image = voxelizer.forward_features(coords, center, atom_features, atom_radius)  # (5, 32, 32, 32)
```



## Voxelization

### Input

- $X \in \mathbb{R}^{N\times3}$ : Coordinates of $N$ atoms
- $R \in \mathbb{R}^N$ : Radii of $N$ atoms
- $F \in \mathbb{R}^{N\times C}$ : Atomic Features of $N$ atoms - $C$ channels.

### Kernel

$d$: distance, $r$: atom radius

#### Gaussian Kernel

$\sigma$: gaussian sigma (default=0.5)

$$
f(d, r, \sigma) =
\begin{cases}
	\exp
		\left(
			-0.5(\frac{d/r}{\sigma})^2
		\right)	& \text{if}~d \leq r \\
	0					& \text{else}
\end{cases}
$$

#### Binary Kernel

$$
f(d, r) =
\begin{cases}
	1	& \text{if}~d \leq r \\
	0	& \text{else}
\end{cases}
$$

### Output

- $I \in \mathbb{R}^{D \times H \times W \times C}$ : Output Image with $C$ channels.
- $G \in \mathbb{R}^{D\times H\times W \times 3}$ : 3D Grid of $I$.

$$
I_{d,h,w,:} = \sum_{n}^{N} F_n \times f(||X_n - G_{d,h,w}||,R_n,\sigma)
$$



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

