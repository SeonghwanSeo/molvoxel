from setuptools import setup, find_packages

setup(
    name='pymolgrid',
    version='0.1.0',
    description='PyMolGrid: python library to generate grid from molecular data',
    author='Seonghwan Seo',
    author_email='shwan0106@gmail.com',
    url='https://github.com/SeonghwanSeo/pymolgrid',
    #packages=find_packages('pymolgrid'),
    packages=['pymolgrid/'],
    install_requires=[
        'numpy',
        'scipy',
        'rdkit-pypi'
    ],
)
