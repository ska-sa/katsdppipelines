import argparse
import os.path

import numpy as np
from pretty import pretty

import katdal

import katsdpcontim
from katsdpcontim import (KatdalAdapter, obit_context, AIPSPath,
                        uv_factory, uv_export,
                        uv_history_obs_description,
                        uv_history_selection)
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
                                     type=int,
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

KA = KatdalAdapter(katdal.open(args.katdata))

with obit_context():
    # Construct file object
    aips_path = KA.aips_path(args.name, args.disk,
                            args.aclass, args.seq, dtype="AIPS")

    # Apply the katdal selection
    KA.select(**args.select)

    # Fall over on empty selections
    if not KA.size > 0:
        raise ValueError("The katdal selection produced an empty dataset"
                        "\n'%s'\n" % pretty(args.select))


    # UV file location variables
    with uv_factory(aips_path=aips_path, mode="w",
                        nvispio=args.nvispio,
                        table_cmds=KA.default_table_cmds(),
                        desc=KA.uv_descriptor()) as uvf:

        # Write history
        uv_history_obs_description(KA, uvf)
        uv_history_selection(args.select, uvf)

        # Perform export to the file
        uv_export(KA, uvf)

    # Possibly perform baseline dependent averaging
    if args.blavg == True:
        task_kwargs = aips_path.task_input_kwargs()
        task_kwargs.update(aips_path.task_output_kwargs(aclass='uvav'))
        blavg = task_factory("UVBlAvg", **task_kwargs)

        blavg.go()
