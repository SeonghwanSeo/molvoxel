from setuptools import setup, find_packages

setup(
    name='pymolgrid',
    version='0.1.0',
    description='PyMolGrid: Python Library to Generate Grid from 3D Molecular Structure',
    author='Seonghwan Seo',
    author_email='shwan0106@gmail.com',
    url='https://github.com/SeonghwanSeo/pymolgrid',
    packages=['pymolgrid/'],
    install_requires=['numpy', 'scipy'],
)
