"""
Basic version of the continuum imaging pipeline.

(1) Opens a katdal observation
(2) For each selected scan
    (a) Writes an AIPS UV scan file
    (b) Runs Baseline Dependent Averaging Obit task
        on scan UV file to produce blavg UV file.
    (c) Merges blavg UV file into global merge file.
(3) Runs MFImage Obit task on global merge file.
(4) Prints calibration solutions
"""

import argparse
import collections
import logging
import os.path

import six
import numpy as np
from pretty import pprint, pretty

import katdal

import katsdpcontim
from katsdpcontim import (KatdalAdapter, obit_context, AIPSPath,
                        UVFacade, task_factory, task_input_kwargs,
                        task_output_kwargs, katdal_aips_path,
                        uv_export, uv_factory)
from katsdpcontim.util import parse_python_assigns

# uv_export.py -n pks1934 /var/kat/archive2/data/MeerKATAR1/telescope_products/2017/07/15/1500148809.h5


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("katdata", help="Katdal observation file")
    # parser.add_argument("output", type=parse_output,
    #                             help="Merged observation file")
    parser.add_argument("--nvispio", default=1024, type=int)
    parser.add_argument("-ks", "--select", default="scans='track';spw=0",
                                        type=parse_python_assigns,
                                        help="katdal select statement "
                                             "Should only contain python "
                                             "assignment statements to python "
                                             "literals, separated by semi-colons")
    return parser

args = create_parser().parse_args()

KA = katsdpcontim.KatdalAdapter(katdal.open(args.katdata))

log = logging.getLogger('katsdpcontim')

_MERGE_PATH = None
_MERGE_UVF = None

def get_merge_uvf(KA, descriptor):
    """
    Defer creation of the merged UVFacade until
    a descriptor is available from averaged scan data
    to condition it.
    """
    global _MERGE_PATH
    global _MERGE_UVF
    global _MERGE_FIRSTVIS

    if _MERGE_PATH is None:
        # Create the path
        _MERGE_PATH = katdal_aips_path(KA, aclass='merge', seq=1)
        log.warn("Creating '%s'" % _MERGE_PATH)
        # Create the UV object
        _MERGE_UVF = uv_factory(aips_path=_MERGE_PATH, mode="w",
                                nvispio=args.nvispio, katdata=KA,
                                desc=descriptor)

        # Close, update descriptor and reopen
        # _MERGE_UVF.close()
        # _MERGE_UVF.update_descriptor(descriptor)
        # _MERGE_UVF = uv_factory(aips_path=_MERGE_PATH, mode="w",
        #                         nvispio=args.nvispio)

    return _MERGE_PATH, _MERGE_UVF


with obit_context():
    select_args = args.select.copy()

    # Perform katdal selection
    KA.select(**args.select)

    # Get selected indices as python ints
    scan_indices = [int(i) for i in KA.scan_indices]
    scan_paths = []

    # FORTRAN indexing
    merge_firstVis = 1

    nx_rows = []

    # Export each scan individually, baseline average
    # and merge it
    for si in scan_indices:
        # Update the select arguments with just this scan
        select_args['scans'] = si

        # Get path, with sequence based on scan index
        scan_path = katdal_aips_path(KA, aclass='raw', seq=si)
        scan_paths.append(scan_path)
        uv_export(KA, scan_path, nvispio=args.nvispio,
                                kat_select=select_args)

        task_kwargs = task_input_kwargs(scan_path)
        blavg_path = scan_path.copy(aclass='uvav')
        task_kwargs.update(task_output_kwargs(blavg_path))

        blavg = task_factory("UVBlAvg", **task_kwargs)
        blavg.go()

        # Retrieve the single scan index.
        # The time centroids and interval should be correct
        # but the visibility indices need to be repurposed
        scan_uvf = UVFacade(scan_path)
        assert len(scan_uvf.tables["AIPS NX"].rows) == 1
        nx_row = scan_uvf.tables["AIPS NX"].rows[0].copy()

        # Remove scan once baseline averaging is complete
        scan_uvf.Zap()

        blavg_uvf = uv_factory(aips_path=blavg_path, mode='r',
                                        nvispio=args.nvispio)

        blavg_desc = blavg_uvf.Desc.Dict
        blavg_nvis = blavg_desc['nvis']

        # Get the merge file if it hasn't yet been created,
        # conditioning it with the baseline averaged file
        # descriptor. This is because baseline averaged files
        # have integration time as an additional random parameter
        # so the merged file will need to take this into account.
        merge_path, merge_uvf = get_merge_uvf(KA, blavg_desc)

        # Record the starting visibility
        # for this scan in the merge file
        nx_row['START VIS'] = [merge_firstVis]

        log.info("Merging '%s' into '%s'" % (blavg_path, merge_path))

        for blavg_firstVis in six.moves.range(1, blavg_nvis+1, args.nvispio):
            # How many visibilities do we write in this iteration?
            numVisBuff = min(blavg_nvis+1 - blavg_firstVis, args.nvispio)

            # Update read file descriptor
            blavg_desc = blavg_uvf.Desc.Dict
            blavg_desc.update(numVisBuff=numVisBuff)
            blavg_uvf.Desc.Dict = blavg_desc

            # Update write file descriptor
            merge_desc = merge_uvf.Desc.Dict
            merge_desc.update(numVisBuff=numVisBuff)
            merge_uvf.Desc.Dict = merge_desc

            # Read, copy, write
            blavg_uvf.Read(firstVis=blavg_firstVis)
            merge_uvf.np_visbuf[:] = blavg_uvf.np_visbuf
            merge_uvf.Write(firstVis=merge_firstVis)

            # Update starting positions
            blavg_firstVis += numVisBuff
            merge_firstVis += numVisBuff

        # Record the ending visilibity
        # for this scan in the merge file
        nx_row['END VIS'] = [merge_firstVis-1]

        # Append row to index table
        merge_uvf.tables["AIPS NX"].rows.append(nx_row)

        # Remove baseline averaged table
        blavg_uvf.Zap()

    # Create an empty calibration table
    merge_uvf.tables["AIPS NX"].write()

    merge_uvf.attach_CL_from_NX_table(KA.max_antenna_number)

    # Close merge file
    merge_uvf.close()

    # Run imaging task on merged file
    task_kwargs = task_input_kwargs(merge_path)
    task_kwargs.update(task_output_kwargs(merge_path, name=None, aclass=None, seq=None))
    pprint(task_kwargs)
    mfimage = task_factory("MFImage", prtLv=5,**task_kwargs)
    mfimage.go()

    # Re-open and print empty calibration solutions
    merge_uvf = uv_factory(aips_path=merge_path, mode='r',
                                    nvispio=args.nvispio)

    log.info("Calibration Solutions")
    log.info(pretty(merge_uvf.tables["AIPS CL"].rows))
    merge_uvf.close()

