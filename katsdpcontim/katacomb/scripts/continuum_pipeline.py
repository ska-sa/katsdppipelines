#!/usr/bin/env python
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
import logging
import os.path
from os.path import join as pjoin
import sys

import numpy as np
import pkg_resources

import katdal
from katsdptelstate import TelescopeState

import katacomb
from katacomb import (KatdalAdapter, obit_context, AIPSPath,
                        ContinuumPipeline,
                        task_factory,
                        uv_factory,
                        uv_export,
                        uv_history_obs_description,
                        uv_history_selection,
                        export_calibration_solutions,
                        export_clean_components)
from katacomb.aips_path import next_seq_nr
from katacomb.util import (parse_python_assigns,
                        get_and_merge_args,
                        log_exception,
                        post_process_args,
                        fractional_bandwidth)

log = logging.getLogger('katacomb')

def create_parser():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)

    parser.add_argument("katdata",
                        help="Katdal observation file")

    parser.add_argument("-d", "--disk",
                        default=1, type=int,
                        help="AIPS disk")

    parser.add_argument("--nvispio", default=1024, type=int)

    parser.add_argument("-cbid", "--capture-block-id",
                        default=None, type=str,
                        help="Capture Block ID. Unique identifier "
                             "for the observation on which the "
                             "continuum pipeline is run.")

    parser.add_argument("-ts", "--telstate",
                        default='', type=str,
                        help="Address of the telstate server")

    parser.add_argument("-sbid", "--sub-band-id",
                        default=0, type=int,
                        help="Sub-band ID. Unique integer identifier for the sub-band "
                             "on which the continuum pipeline is run.")

    parser.add_argument("-ks", "--select",
                        default="scans='track'; spw=0; corrprods='cross'",
                        type=log_exception(log)(parse_python_assigns),
                        help="katdal select statement "
                             "Should only contain python "
                             "assignment statements to python "
                             "literals, separated by semi-colons")

    TDF_URL = "https://github.com/bill-cotton/Obit/blob/master/ObitSystem/Obit/TDF"


    parser.add_argument("-ba", "--uvblavg",
                        default="",
                        type=log_exception(log)(parse_python_assigns),
                        help="UVBLAVG task parameter assignment statement. "
                             "Should only contain python "
                             "assignment statements to python "
                             "literals, separated by semi-colons. "
                             "See %s/UVBlAvg.TDF for valid parameters. " % TDF_URL)


    parser.add_argument("-mf", "--mfimage",
                        default="",
                        type=log_exception(log)(parse_python_assigns),
                        help="MFImage task parameter assignment statement. "
                             "Should only contain python "
                             "assignment statements to python "
                             "literals, separated by semi-colons. "
                             "See %s/MFImage.TDF for valid parameters. " % TDF_URL)

    parser.add_argument("--clobber",
                        default="scans, avgscans",
                        type=lambda s: set(v.strip() for v in s.split(',')),
                        help="Class of AIPS/Obit output files to clobber. "
                             "'scans' => Individual scans. "
                             "'avgscans' => Averaged individual scans. "
                             "'merge' => Observation file containing merged, "
                                                            "averaged scans. "
                             "'clean' => Output CLEAN files. "
                             "'mfimage' => Output MFImage files. ")

    parser.add_argument("--config",
                        default="/obitconf",
                        type=str,
                        help="Directory containing default configuration "
                             ".yaml files for mfimage and uvblavg. ")


    return parser

args = create_parser().parse_args()

# Open the observation
katdata = katdal.open(args.katdata)

post_process_args(args, katdata)

# Get defaults for uvblavg and mfimage and merge user supplied ones
uvblavg_args = get_and_merge_args(args.config + '/uvblavg.yaml', args.uvblavg)
mfimage_args = get_and_merge_args(args.config + '/mfimage.yaml', args.mfimage)

# Set up telstate link then create
# a view based the capture block ID and sub-band ID
telstate = TelescopeState(args.telstate)
sub_band_id_str = "sub_band%d" % args.sub_band_id
view = telstate.SEPARATOR.join((args.capture_block_id, sub_band_id_str))
ts_view = telstate.view(view)

# Create Continuum Pipeline
pipeline = ContinuumPipeline(katdata, ts_view,
                            katdal_select=args.select,
                            uvblavg_params=uvblavg_args,
                            mfimage_params=mfimage_args,
                            nvispio=args.nvispio,
                            disk=args.disk)

# Execute it
pipeline.execute()
