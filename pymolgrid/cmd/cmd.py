import argparse
import os
import logging

from pymolgrid import config

"""
description = f"The Protein-Ligand Interaction Profiler (PLIP) Version {config.__version__} " \
              "is a command-line based tool to analyze interactions in a protein-ligand complex. " \
              "If you are using PLIP in your work, please cite: " \
              f"{config.__citation_information__} " \
              f"Supported and maintained by: {config.__maintainer__}"

"""

description = f"The PyMolGrid, a Molecular Voxelization Tool. WebPage: {config.__url__}"

def parsing() :
    parser = argparse.ArgumentParser(prog="PyMolGrid", description=description)
    return parser.parse_args()

if __name__ == '__main__' :
    args =parsing()
