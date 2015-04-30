#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="katsdpimager",
    version="0.1.dev0",
    description="GPU-accelerated radio-astronomy imager",
    author="Bruce Merry and Ludwig Schwardt",
    packages=find_packages(),
    scripts=["scripts/katsdpimager.py"],
    install_requires=['numpy', 'katsdpsigproc', 'pyrap.tables', 'pyfits']
)
