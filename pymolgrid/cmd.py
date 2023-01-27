import argparse
import os
import logging

from pymolgrid import config

description = f"The PyMolGrid Version {config.__version__} " \
              "is a Molecular Representation Tool for analyzing protein-ligand complex. " \
              "If you are using PyMolGrid in your work, please cite: " \
              f"NaN " \
              f"Supported and maintained by: {config.__author__}."

def parsing() :
    parser = ArgumentParser(prog="PyMolGrid", description=description)


