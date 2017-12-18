"""
Basic version of the continuum imaging pipeline.

(1) Opens a katdal observation
(2) For each selected scan
    (a) Writes an AIPS UV scan file
    (b) Runs Baseline Dependent Averaging Obit task
        on scan UV file to produce blavg UV file.
    (c) Merges blavg UV file into global merge file.
(3) Runs MFImage Obit task on global merge file.
(4) Writes calibration solutions and clean components to telstate
"""

import argparse
import collections
from copy import deepcopy
import json
import logging
import os.path
from os.path import join as pjoin
import sys

import numpy as np
import pkg_resources
from pretty import pprint, pretty
import psutil
import six

import katdal
from katsdptelstate import TelescopeState

import katsdpcontim
from katsdpcontim import (KatdalAdapter, obit_context, AIPSPath,
                        task_factory,
                        img_factory,
                        uv_export,
                        uv_history_obs_description,
                        uv_history_selection,
                        uv_factory)
from katsdpcontim.util import (parse_python_assigns,
                        post_process_args,
                        fractional_bandwidth)

log = logging.getLogger('katsdpcontim')

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("katdata", help="Katdal observation file")
    parser.add_argument("-d", "--disk", default=1, type=int,
                                        help="AIPS disk")
    parser.add_argument("--nvispio", default=1024, type=int)
    parser.add_argument("-cbid", "--capture-block-id", default=None, type=str,
                                        help="Capture Block ID. Unique identifier "
                                             "for the observation on which the "
                                             "continuum pipeline is run.")
    parser.add_argument("-ts", "--telstate-server", default='', type=str,
                                        help="Address of the telstate server")
    parser.add_argument("-sbid", "--sub-band-id", default=0, type=int,
                                        help="Sub-band ID. Unique integer "
                                             "identifier for the sub-band "
                                             "on which the continuum pipeline "
                                             "is run.")
    parser.add_argument("-ks", "--select", default="scans='track';spw=0",
                                        type=parse_python_assigns,
                                        help="katdal select statement "
                                             "Should only contain python "
                                             "assignment statements to python "
                                             "literals, separated by semi-colons")
    return parser

args = create_parser().parse_args()


# Standard MFImage output classes for UV and CLEAN images
UV_CLASS = "MFImag"
IMG_CLASS = "IClean"

with obit_context():
    KA = katsdpcontim.KatdalAdapter(katdal.open(args.katdata))
    uv_merge_path = KA.aips_path(aclass='merge', seq=None)
    log.info("Exporting to '%s'" % uv_merge_path)

    # Perform argument postprocessing
    args = post_process_args(args, KA)

    # Backed by a fake REDIS server
    telstate = TelescopeState(args.telstate_server)

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

    # Fall over on empty selections
    if not KA.size > 0:
        raise ValueError("The katdal selection produced an empty dataset"
                        "\n'%s'\n" % pretty(global_select))

    global_desc = KA.uv_descriptor()
    global_table_cmds = KA.default_table_cmds()
    scan_indices = [int(i) for i in KA.scan_indices]

    def _source_info(KA):
        """
        Infer MFImage source names and file  outputs for each sources.
        For each source, MFImage outputs UV data and a CLEAN file.
        """

        # Source names
        uv_sources = [s["SOURCE"][0].strip() for s in KA.uv_source_rows]

        uv_files = [AIPSPath(name=s, disk=args.disk, aclass=UV_CLASS,
                                                seq=None, atype="UV")
                                                    for s in uv_sources]

        clean_files = [AIPSPath(name=s, disk=args.disk, aclass=IMG_CLASS,
                                                seq=None, atype="MA")
                                                    for s in uv_sources]

        return uv_sources, uv_files, clean_files

    uv_sources, uv_files, clean_files = _source_info(KA)

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

        # Get the AIPS source for logging purposes
        aips_source = KA.catalogue[KA.target_indices[0]]
        aips_source_name = aips_source["SOURCE"][0].strip()

        blavg_kwargs = scan_path.task_input_kwargs()
        blavg_path = scan_path.copy(aclass='uvav')
        blavg_kwargs.update(blavg_path.task_output_kwargs())

        blavg_params = dict()
        blavg_kwargs.update(blavg_params)

        log.info("Time-dependent baseline averaging "
                "'%s' to '%s'" % (scan_path, blavg_path))

        blavg = task_factory("UVBlAvg", **blavg_kwargs)
        blavg.go()

        # Retrieve the single scan index.
        # The time centroids and interval should be correct
        # but the visibility indices need to be repurposed
        scan_uvf = uv_factory(aips_path=scan_path, mode='r',
                                        nvispio=args.nvispio)

        assert len(scan_uvf.tables["AIPS NX"].rows) == 1, scan_uvf.tables["AIPS NX"].rows
        nx_row = scan_uvf.tables["AIPS NX"].rows[0].copy()
        scan_desc = scan_uvf.Desc.Dict
        scan_nvis = scan_desc['nvis']

        # If we've performed channel averaging, the FREQ dimension
        # and FQ tables in the baseline averaged file will be smaller
        # than that of the scan file. This data must either be
        # (1) Used to condition a new merge UV file
        # (2) compared against the method in the existing merge UV file
        blavg_uvf = uv_factory(aips_path=blavg_path, mode='r',
                                          nvispio=args.nvispio)
        blavg_desc = blavg_uvf.Desc.Dict
        blavg_nvis = blavg_desc['nvis']

        # Record something about the baseline averaging process
        param_str = ', '.join("%s=%s" % (k,v) for k,v in blavg_params.items())
        blavg_history = ("Scan %d '%s' averaged %s to %s visiblities. UVBlAvg(%s)" %
                (si, aips_source_name, scan_nvis, blavg_nvis, param_str))
        log.info(blavg_history)

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

            # Write history
            uv_history_obs_description(KA, merge_uvf)
            uv_history_selection(global_select, merge_uvf)

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

        merge_uvf.append_history(blavg_history)

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

        # Record the ending visibility
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

    uv_seq = max(f.seq for f in uv_files)
    clean_seq = max(f.seq for f in clean_files)

    # Run MFImage task on merged file,
    # using no-self calibration config options (mfimage_nosc.in)
    mfimage_kwargs = uv_merge_path.task_input_kwargs()
    mfimage_kwargs.update(uv_merge_path.task_output_kwargs(name='', aclass=IMG_CLASS, seq=clean_seq))
    mfimage_kwargs.update(uv_merge_path.task_output2_kwargs(name='', aclass=UV_CLASS, seq=uv_seq))
    mfimage_cfg = pkg_resources.resource_filename('katsdpcontim', pjoin('conf', 'mfimage_nosc.in'))
    mfimage_kwargs.update(maxFBW=fractional_bandwidth(blavg_desc)/20.0)
    mfimage_kwargs.update(maxFBW=fractional_bandwidth(blavg_desc)/20.0,
                          nThreads=psutil.cpu_count(logical=True))

    log.info("MFImage arguments %s" % pretty(mfimage_kwargs))

    mfimage = task_factory("MFImage", mfimage_cfg, taskLog='IMAGE.log', prtLv=5,**mfimage_kwargs)
    mfimage.go()

    # AIPS Table entries follow this kind of schema where data
    # is stored in singleton lists, while book-keeping entries
    # are not.
    # { 'DELTAX' : [1.4], 'NumFields' : 4, 'Table name' : 'AIPS CC' }
    # Strip out book-keeping keys and flatten lists
    DROP = { "Table name", "NumFields", "_status" }
    def _condition(row):
        """ Flatten singleton lists and drop book-keeping keys """
        return { k: v[0] for k, v in row.items() if k not in DROP }

    # Create a view based the capture block ID
    view = '_'.join((args.capture_block_id, "sub_band_%d" % args.sub_band_id))
    ts_view = telstate.view(view)

    # MFImage outputs a UV file per source.  Iterate through each source:
    # (1) Extract complex gains from attached "AIPS SN" table
    # (2) Write them to telstate
    for si, (uv_file, uv_source) in enumerate(zip(uv_files, uv_sources)):
        target = "target%d" % si

        # Create contexts
        uvf_ctx = uv_factory(aips_path=uv_file, mode='r')
        json_ctx = open(target + "-solutions.json", "w")

        with uvf_ctx as uvf, json_ctx as json_f:
            try:
                sntab = uvf.tables["AIPS SN"]
            except KeyError:
                log.warn("No calibration solutions in '%s'" % uv_file)
            else:
                # Handle cases for single/dual pol gains
                if "REAL2" in sntab.rows[0]:
                    def _extract_gains(row):
                        return np.array([row["REAL1"] + 1j*row["IMAG1"],
                                         row["REAL2"] + 1j*row["IMAG2"]])
                else:
                    def _extract_gains(row):
                        return np.array([row["REAL1"] + 1j*row["IMAG1"]])

                # Write each complex gain out per antenna
                for row in (_condition(r) for r in sntab.rows):
                    # Convert time back from AIPS to katdal UTC
                    time = row["TIME"]*86400.0 + KA.midnight
                    # Convert from AIPS FORTRAN indexing to katdal C indexing
                    ant = "m%03d_gains" % (row["ANTENNA NO."]-1)

                    # Store complex gain in telstate
                    key = ts_view.SEPARATOR.join((target,ant))
                    ts_view.add(key, _extract_gains(row), ts=time)

                    # Dump each row to file
                    json.dump(row, json_f)
                    json_f.write("\n")

        uvf.Zap()

    # MFImage outputs a CLEAN image per source.  Iterate through each source:
    # (1) Extract clean components from attached "AIPS CC" table
    # (2) Write them to telstate
    for si, (clean_file, uv_source) in enumerate(zip(clean_files, uv_sources)):
        target = "target%d" % si

        # Create contexts
        img_ctx = img_factory(aips_path=clean_file, mode='r')
        json_ctx = open(target + "-clean.json", "w")

        with img_ctx as cf, json_ctx as json_f:
            try:
                cctab = cf.tables["AIPS CC"]
            except KeyError:
                log.warn("No clean components in '%s'" % clean_file)
            else:
                # Condition all rows up front
                rows = [_condition(r) for r in cctab.rows]

                # Store them in telstate
                key = ts_view.SEPARATOR.join((target, "clean_components"))
                ts_view.add(key, rows)

                # Dump each row to file
                for row in rows:
                    json.dump(row, json_f)
                    json_f.write("\n")

        cf.Zap()


    merge_uvf.close()
