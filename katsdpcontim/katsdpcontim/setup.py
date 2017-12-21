import codecs
from glob import glob
import os
from os.path import join as pjoin
import re

from setuptools import setup, find_packages

PKG = 'katsdpcontim'

def find_version():
    """ Extract version from version.py """
    version_file = pjoin(PKG, 'version.py')

    with codecs.open(version_file, 'r', encoding='utf-8') as f:
        VERSION_RE = r"^__version__ = ['\"]([^'\"]*)['\"]"
        version_match = re.search(VERSION_RE, f.read(), re.M)

        if version_match:
            return version_match.group(1)

        raise RuntimeError("Unable to find version string.")

DESCRIPTION = "MeerKAT SDP Continuum Pipeline"

setup(name=PKG,
    version=find_version(),
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    url='https://github.com/ska-sa/katsdppipelines/tree/master/katsdpcontim',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    author='Simon Perkins',
    author_email='sperkins@ska.ac.za',
    install_requires=['attrs >= 17.2.0',
                      'cerberus >= 1.1',
                      'numpy >= 1.13.1',
                      'pretty >= 0.1',
                      'ruamel.yaml >= 0.15.23',
                      'six >= 1.10.0'],
    scripts=glob(pjoin('scripts', '*.py')),
    packages=find_packages(),
    package_data={PKG: [pjoin('conf', '*.in')]}
)
