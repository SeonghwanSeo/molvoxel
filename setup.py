from setuptools import setup, find_packages

with open('./README.md', encoding='utf-8') as f:
    long_description = f.read()

PACKAGES = find_packages('./')

setup(
    name='molvoxel',
    version='0.1.3',
    description='MolVoxel:Easy-to-Use Molecular Voxelization Tool',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Seonghwan Seo',
    author_email='shwan0106@gmail.com',
    url='https://github.com/SeonghwanSeo/molvoxel',
    packages=PACKAGES,
    install_requires=['numpy', 'scipy'],
    extras_require={
            'numba': ['numba'],
            'torch': ['torch'],
            'rdkit': ['rdkit'],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Development Status :: 4 - Beta',

        'Operating System :: OS Independent',

        'License :: OSI Approved :: MIT License',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
