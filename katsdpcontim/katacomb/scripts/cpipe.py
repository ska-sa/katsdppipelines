import argparse
import logging
import random

import katdal
from katsdptelstate import TelescopeState

from katacomb.util import (parse_python_assigns,
                        log_exception)

from katacomb.continuum_pipeline import ContinuumPipeline
from katacomb.mock_dataset import MockDataSet

from ephem.stars import stars
import katpoint
import numpy as np
import six


def create_dataset():
        nchan = 1024
        nvispio = 1024

        spws = [{
            'centre_freq' : .856e9 + .856e9 / 2.,
            'num_chans' : nchan,
            'channel_width' : .856e9 / nchan,
            'sideband' : 1,
            'band' : 'L',
        }]

        target_names = random.sample(stars.keys(), 5)

        # Pick 5 random stars as targets
        targets = [katpoint.Target("%s, star" % t) for t in
                                                target_names]

        # Set up varying scans
        scans = [('slew', 1, targets[0]), ('track', 3, targets[0]),
                 ('slew', 2, targets[1]), ('track', 5, targets[1]),
                 ('slew', 1, targets[2]), ('track', 8, targets[2]),
                 ('slew', 2, targets[3]), ('track', 9, targets[3]),
                 ('slew', 1, targets[4]), ('track', 10, targets[4])]

        # Create Mock dataset and wrap it in a KatdalAdapter
        return MockDataSet(spws=spws, dumps=scans)

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
                        # Averaging FOV is 1.0, turn on frequency averaging,
                        # average eight channels together. Average a maximum
                        # integration time of 2 minutes
                        default="FOV=1.0; maxInt=2.0; avgFreq=1; chAvg=2",
                        #default="FOV=1.0; avgFreq=1; chAvg=8; maxInt=2.0",
                        type=log_exception(log)(parse_python_assigns),
                        help="UVBLAVG task parameter assignment statement. "
                             "Should only contain python "
                             "assignment statements to python "
                             "literals, separated by semi-colons. "
                             "See %s/UVBlAvg.TDF for valid parameters. " % TDF_URL)


    parser.add_argument("-mf", "--mfimage",
                        # FOV of 1.2 degrees, 5000 Clean cycles,
                        # 3 phase self-cal loops with a solution interval
                        # of four minutes
                        default="FOV=1.2; Niter=5000; "
                                "maxPSCLoop=3; minFluxPSC=0.0; solPInt=4.0",
                        type=log_exception(log)(parse_python_assigns),
                        help="MFIMAGE task parameter assignment statement. "
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


    return parser

args = create_parser().parse_args()

katdata = create_dataset()

if args.capture_block_id is None:
    args.capture_block_id = katdata.experiment_id

# Set up telstate link then create
# a view based the capture block ID and sub-band ID
telstate = TelescopeState(args.telstate)
sub_band_id_str = "sub_band%d" % args.sub_band_id
view = telstate.SEPARATOR.join((args.capture_block_id, sub_band_id_str))
ts_view = telstate.view(view)

pipeline = ContinuumPipeline(katdata, ts_view,
                            katdal_select=args.select,
                            uvblavg_params=args.uvblavg,
                            mfimage_params=args.mfimage,
                            nvispio=args.nvispio)

pipeline.export_scans()
