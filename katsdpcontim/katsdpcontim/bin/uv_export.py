import argparse
import os.path

import numpy as np

import UV

import katdal

import katsdpcontim
from katsdpcontim import KatdalAdapter, obit_context
from katsdpcontim.uv_export import uv_export
from katsdpcontim.util import parse_python_assigns

# uv_export.py -n pks1934 /var/kat/archive2/data/MeerKATAR1/telescope_products/2017/07/15/1500148809.h5

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("katdata", help="hdf5 observation file")
    parser.add_argument("-l", "--label", default="MeerKAT")
    parser.add_argument("-n", "--name", help="AIPS name")
    parser.add_argument("-c", "--class", default="raw", dest="aclass",
                                        help="AIPS class")
    parser.add_argument("-d", "--disk", default=1,
                                        help="AIPS disk")
    parser.add_argument("-s", "--seq", default=1,
                                        help="AIPS sequence")
    parser.add_argument("--nvispio", default=1024,
                                        help="Number of visibilities "
                                             "read/written per IO call")
    parser.add_argument("-ks", "--select", default="scans='track';spw=0",
                                        type=parse_python_assigns,
                                        help="katdal select statement "
                                             "Should only contain python "
                                             "assignment statements to python "
                                             "literals, separated by semi-colons")
    parser.add_argument("--blavg", default=False, action="store_true",
                                    help="Apply baseline dependent averaging")
    return parser

args = create_parser().parse_args()

# Use the hdf5file name if none is supplied
if args.name is None:
    path, file  = os.path.split(args.katdata)
    base_filename, ext = os.path.splitext(file)
    args.name = base_filename

KA = KatdalAdapter(katdal.open(args.katdata))

with obit_context():
    uv_export(KA, args.name, args.aclass, args.seq, args.disk,
        dtype="AIPS", kat_select=args.select)
