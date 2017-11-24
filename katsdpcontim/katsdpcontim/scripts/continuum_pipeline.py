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
from copy import deepcopy
import logging
import os.path
from os.path import join as pjoin

import pkg_resources
import six
import numpy as np
from pretty import pprint, pretty

import katdal

import katsdpcontim
from katsdpcontim import (KatdalAdapter, obit_context, AIPSPath,
                        UVFacade, task_factory, uv_export, uv_factory)
from katsdpcontim.util import parse_python_assigns

log = logging.getLogger('katsdpcontim')

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("katdata", help="Katdal observation file")
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
uv_merge_path = KA.aips_path(aclass='merge', seq=1)

with obit_context():
    # The merged UV observation file. We wait until
    # we have a baseline averaged file to condition it with
    merge_uvf = None

    # Save katdal selection
    global_select = args.select.copy()
    scan_select = args.select.copy()

    # Perform katdal selection
    # retrieving selected scan indices as python ints
    # so that we can do per scan selection
    KA.select(**global_select)
    global_desc = KA.uv_descriptor()
    global_table_cmds = KA.default_table_cmds()
    scan_indices = [int(i) for i in KA.scan_indices]

    # FORTRAN indexing
    merge_firstVis = 1

    # Export each scan individually, baseline average
    # and merge it
    for si in scan_indices:
        # Clear katdal selection and set to global selection
        KA.select()
        KA.select(**global_select)

        # Get path, with sequence based on scan index
        scan_path = uv_merge_path.copy(aclass='raw', seq=si)

        log.info("Creating '%s'" % scan_path)

        # Create a UV file for the scan
        with uv_factory(aips_path=scan_path, mode="w",
                            nvispio=args.nvispio,
                            table_cmds=global_table_cmds,
                            desc=global_desc) as uvf:

            # Perform katdal selection on the specific scan
            scan_select['scans'] = si
            KA.select(**scan_select)

            # Perform export to the file
            uv_export(KA, uvf)

        task_kwargs = scan_path.task_input_kwargs()
        blavg_path = scan_path.copy(aclass='uvav')
        task_kwargs.update(blavg_path.task_output_kwargs())

        log.info("Time-dependent baseline "
                "averaging '%s' to '%s'" % (scan_path, blavg_path))
        blavg = task_factory("UVBlAvg", **task_kwargs)
        blavg.go()

        # Retrieve the single scan index.
        # The time centroids and interval should be correct
        # but the visibility indices need to be repurposed
        scan_uvf = uv_factory(aips_path=scan_path, mode='r',
                                        nvispio=args.nvispio)

        assert len(scan_uvf.tables["AIPS NX"].rows) == 1, scan_uvf.tables["AIPS NX"].rows
        nx_row = scan_uvf.tables["AIPS NX"].rows[0].copy()

        # If we've performed channel averaging, the FREQ dimension
        # and FQ tables in the baseline averaged file will be smaller
        # than that of the scan file. This data must either be
        # (1) Used to condition a new merge UV file
        # (2) compared against the method in the existing merge UV file
        blavg_uvf = uv_factory(aips_path=blavg_path, mode='r',
                                          nvispio=args.nvispio)
        blavg_desc = blavg_uvf.Desc.Dict
        blavg_nvis = blavg_desc['nvis']

        blavg_fq_keywords = dict(blavg_uvf.tables["AIPS FQ"].keywords)
        blavg_fq_rows = blavg_uvf.tables["AIPS FQ"].rows

        # Get the merge file if it hasn't yet been created,
        # conditioning it with the baseline averaged file
        # descriptor. Baseline averaged files
        # have integration time as an additional random parameter
        # so the merged file will need to take this into account.
        if merge_uvf is None:
            log.info("Creating '%s'" % uv_merge_path)

            # Use the FQ table rows and keywords to create
            # the merge UV file.
            blavg_table_cmds = deepcopy(global_table_cmds)
            fq_cmd = blavg_table_cmds["AIPS FQ"]
            fq_cmd["keywords"] = blavg_fq_keywords
            fq_cmd["rows"] = blavg_fq_rows

            # Create the UV object
            merge_uvf = uv_factory(aips_path=uv_merge_path, mode="w",
                                    nvispio=args.nvispio,
                                    table_cmds=blavg_table_cmds,
                                    desc=blavg_desc)


        # Now do some sanity checks of the merge UV file's metadata
        # against that of the blavg UV file. Mostly we check that the
        # frequency information is the same.
        merge_desc = merge_uvf.Desc.Dict
        merge_fq_keywords = dict(merge_uvf.tables["AIPS FQ"].keywords)
        merge_fq_rows = merge_uvf.tables["AIPS FQ"].rows

        # Compare
        # (1) UV FITS shape parameters
        # (2) FQ table keywords
        # (3) FQ table rows
        assert all(merge_desc[k] == blavg_desc[k] for k in (
                    'inaxes', 'cdelt', 'crval', 'naxis', 'crota', 'crpix',
                     'ilocu', 'ilocv', 'ilocw', 'ilocb', 'iloct', 'ilocsu',
                    'jlocc', 'jlocs', 'jlocf', 'jlocif', 'jlocr', 'jlocd'))
        assert all(merge_fq_keywords[k] == blavg_fq_keywords[k] for k in blavg_fq_keywords.keys())
        assert len(merge_fq_rows) == len(blavg_fq_rows)
        assert all(all(mr[k] == br[k] for k in br.keys()) for mr, br in zip(merge_fq_rows, blavg_fq_rows))

        # Record the starting visibility
        # for this scan in the merge file
        nx_row['START VIS'] = [merge_firstVis]

        log.info("Merging '%s' into '%s'" % (blavg_path, uv_merge_path))

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

        # Remove scan and baseline averaged files once merged
        log.info("Zapping '%s'" % scan_uvf.aips_path)
        scan_uvf.Zap()
        log.info("Zapping '%s'" % blavg_uvf.aips_path)
        blavg_uvf.Zap()

    # Write the index table
    merge_uvf.tables["AIPS NX"].write()

    # Create an empty calibration table
    merge_uvf.attach_CL_from_NX_table(KA.max_antenna_number)

    # Close merge file
    merge_uvf.close()

    # Run MFImage task on merged file,
    # using no-self calibration config options (mfimage_nosc.in)
    task_kwargs = uv_merge_path.task_input_kwargs()
    task_kwargs.update(uv_merge_path.task_output_kwargs(name=None, aclass=None, seq=None))
    mfimage_cfg = pkg_resources.resource_filename('katsdpcontim', pjoin('conf', 'mfimage_nosc.in'))
    mfimage = task_factory("MFImage", mfimage_cfg, taskLog='', prtLv=5,**task_kwargs)
    mfimage.go()

    # Re-open and print empty calibration solutions
    merge_uvf = uv_factory(aips_path=uv_merge_path, mode='r',
                                    nvispio=args.nvispio)

    log.info("Calibration Solutions")
    log.info(pretty(merge_uvf.tables["AIPS CL"].rows))
    merge_uvf.close()


