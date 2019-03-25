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
from katsdpservices import setup_logging
from katsdptelstate import TelescopeState

import katacomb
import katacomb.configuration as kc
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
                        fractional_bandwidth,
                        setup_aips_disks)

log = logging.getLogger('katacomb')

def create_parser():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)

    parser.add_argument("katdata",
                        help="Katdal observation file")

    parser.add_argument('--access-key',
                        help='S3 access key to access the data')

    parser.add_argument('--secret-key',
                        help='S3 secret key to access the data')

    parser.add_argument('--token',
                        help='JWT to access the MeerKAT archive')

    parser.add_argument("-w", "--workdir",
                        default=None, type=str,
                        help="Location of scratch space. An AIPS disk (called aipsdisk) "
                             "will be created in this space and logfiles of Obit "
                             "tasks will be written here.")

    parser.add_argument("--nvispio", default=10240, type=int)

    parser.add_argument("-cbid", "--capture-block-id",
                        default=None, type=str,
                        help="Capture Block ID. Unique identifier "
                             "for the observation on which the "
                             "continuum pipeline is run.")

    parser.add_argument("-ts", "--telstate",
                        default='', type=str,
                        help="Address of the telstate server")

    parser.add_argument("-oid", "--output-id",
                        default="continuum_image", type=str,
                        help="Label the product of the continuum pipeline. "
                             "Used to generate telstate keys.")

    parser.add_argument("-ks", "--select",
                        default="scans='track'; spw=0; corrprods='cross'",
                        type=log_exception(log)(parse_python_assigns),
                        help="katdal select statement "
                             "Should only contain python "
                             "assignment statements to python "
                             "literals, separated by semi-colons.")

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
                        default=os.path.sep + "obitconf",
                        type=str,
                        help="Directory containing default configuration "
                             ".yaml files for mfimage and uvblavg. ")

    parser.add_argument("--nif", default=8, type=int,
                        help="Number of AIPS 'IFs' to equally subdivide the band. "
                             "NOTE: Must divide the number of channels after any "
                             "katdal selection.")
    return parser

setup_logging()
parser = create_parser()
args = parser.parse_args()

# Open the observation
if (args.access_key is not None) != (args.secret_key is not None):
    parser.error('--access-key and --secret-key must be used together')
if args.access_key is not None and args.token is not None:
    parser.error('--access-key/--secret-key cannot be used with --token')
open_kwargs = {}
if args.access_key is not None:
    open_kwargs['credentials'] = (args.access_key, args.secret_key)
elif args.token is not None:
    open_kwargs['token'] = args.token
katdata = katdal.open(args.katdata, applycal='all', **open_kwargs)

post_process_args(args, katdata)

# Get defaults for uvblavg and mfimage and merge user supplied ones
uvblavg_args = get_and_merge_args(pjoin(args.config,'uvblavg.yaml'), args.uvblavg)
mfimage_args = get_and_merge_args(pjoin(args.config,'mfimage.yaml'), args.mfimage)

# Set up configuration from args.workdir
if args.workdir is not None:
    aipsdirs = [(None, pjoin(args.workdir, args.capture_block_id + '_aipsdisk'))]
    kc.set_config(aipsdirs=aipsdirs)
    setup_aips_disks()

# Set up telstate link then create
# a view based the capture block ID and output ID
telstate = TelescopeState(args.telstate)
view = telstate.SEPARATOR.join((args.capture_block_id, args.output_id))
ts_view = telstate.view(view)

katdal_select = args.select
katdal_select['nif'] = args.nif

# Create Continuum Pipeline
pipeline = ContinuumPipeline(katdata, ts_view,
                            katdal_select=katdal_select,
                            uvblavg_params=uvblavg_args,
                            mfimage_params=mfimage_args,
                            nvispio=args.nvispio)

# Execute it
pipeline.execute()
