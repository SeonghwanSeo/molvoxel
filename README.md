# PyMolGrid: Molecular Voxelization
PyMolGrid is easy-to-use **Molecular Voxelization Tool** implemented in Python.

It just requires Numpy and SciPy, so you can install it very simply.
Moreover, it supports **PyTorch**, you can use **CUDA** if available.

## Quick Start

### Python (Numpy)

```python
from pymolgrid import Voxelizer
import numpy as np
from rdkit import Chem  # rdkit is not required packages

def get_atom_features(atom) :
  symbol, arom = atom.GetSymbol(), atom.GetIsAromatic()
  return [symbol == 'C', symbol == 'N', symbol == 'O', symbol == 'S', arom]

mol = Chem.SDMolSupplier('test/10gs/ligand.sdf')[0]
channels = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
coords = mol.GetConformer().GetPositions()                  # (V, 3)
center = coords.mean(axis = 0)                              # (3,)
atom_types = [channels[atom.GetSymbol()] for atom in mol.GetAtoms()]
atom_types = np.array(atom_types)                           # (V,)
atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
atom_features = np.array(atom_features)                     # (V, 5)
radii = 1.0

voxelizer = Voxelizer(resolution = 0.5, dimension = 64)
out = voxelizer(coords, center, atom_types, radii)          # (4, 64, 64, 64)
out = voxelizer(coords, center, atom_features, radii)       # (5, 64, 64, 64)
```

### Python (PyTorch - Cuda Available)

```python
from pymolgrid.torch import Voxelizer
import torch    # IMPORTANT! import torch after import pymolgrid (torch vs scipy)

device = 'cuda' # or 'cpu'
coords = torch.FloatTensor(coords).to(device)               # (V, 3)
center = torch.FloatTensor(center).to(device)               # (3,)
atom_types = torch.LongTensor(atom_types).to(device)        # (V,)
atom_features = torch.FloatTensor(atom_features).to(device) # (V, 5)

voxelizer = Voxelizer(resolution = 0.5, dimension = 32, device = device)
out = voxelizer(coords, center, atom_types, radii)          # (4, 32, 32, 32)
out = voxelizer(coords, center, atom_features, radii)       # (5, 32, 32, 32)
```

---

## Installation

```shell
# Required: numpy, scipy
# Not Required, but Recommended: RDKit
# Optional - PyTorch : torch
# Optional - Visualization : pymol-open-source
git clone https://github.com/SeonghwanSeo/pymolgrid.git
cd pymolgrid/
pip install -e .
```

