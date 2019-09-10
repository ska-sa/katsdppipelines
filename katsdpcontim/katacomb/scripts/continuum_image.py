#!/usr/bin/env python
import argparse
import logging
import os

import katdal

import katacomb.configuration as kc
from katacomb import ImagePipeline
from katacomb.util import (log_exception,
                           parse_python_assigns,
                           get_and_merge_args,
                           setup_aips_disks,
                           recursive_merge)


log = logging.getLogger('katacomb')

def configure_logging(args):
    log_handler = logging.StreamHandler()
    fmt = "[%(levelname)s] %(message)s"
    log_handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(log_handler)
    log.setLevel(args.log_level.upper())

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("katdata",
                        help="Katdal observation reference.")

    parser.add_argument("-a", "--applycal",
                        default="default", type=str,
                        help="Apply calibration solutions to visibilities "
                             "before imaging. The list of desired solutions "
                             "is comma separated and each takes the form "
                             "'stream'.'product' where 'stream' is either of "
                             "'l1' (cal) or 'l2' (self-cal) and product is "
                             "one of 'K','G','B' for l1 and 'GPHASE', 'GAMP_PHASE' "
                             "for l2. You can also select 'default' (Apply l1.K, l1.G, l1.B "
                             "and l2.GPHASE) or 'all' (Apply all available solutions). "
                             "Default: %(default)s (You probably want this.)")

    parser.add_argument("-w", "--workdir",
                        default='/scratch', type=str,
                        help="Location of scratch space. An AIPS disk "
                             "will be created in this space. Default: %(default)s")

    parser.add_argument("-o", "--outputdir",
                        default='/scratch', type=str,
                        help="Output directory. FITS image files named <target_name>.fits "
                             "will be placed here for each target. Default: %(default)s")

    parser.add_argument("--nvispio", default=10240, type=int,
                        help="Number of visibilities per write when copying data from archive. "
                             "Default: %(default)s")

    parser.add_argument("-t", "--targets", default=None, type=str,
                        help="Comma separated list of target names to image. "
                             "Default: All targets")

    parser.add_argument("-c", "--channels", default=None, type=str,
                        help="Range of channels to image, must be of the form <start>,<end>. "
                             "Default: Image all (unmasked) channels.")

    TDF_URL = "https://github.com/bill-cotton/Obit/blob/master/ObitSystem/Obit/TDF"

    parser.add_argument("-ba", "--uvblavg",
                        default="avgFreq=1; chAvg=4",
                        type=log_exception(log)(parse_python_assigns),
                        help="UVBlAvg task parameter assignment statement. "
                             "Should only contain python "
                             "assignment statements to python "
                             "literals, separated by semi-colons. "
                             "See " + TDF_URL +"/UVBlAvg.TDF for valid parameters. "
                             "Default: %(default)s")

    parser.add_argument("--uvblavg-config",
                        default=os.path.sep + "obitconf" + os.path.sep + "uvblavg.yaml",
                        type=str,
                        help="Configuration yaml file for UVBlAvg. Default: %(default)s")

    parser.add_argument("-mf", "--mfimage",
                        default="",
                        type=log_exception(log)(parse_python_assigns),
                        help="MFImage task parameter assignment statement. "
                             "Should only contain python "
                             "assignment statements to python "
                             "literals, separated by semi-colons. "
                             "See %s/MFImage.TDF for valid parameters. " % TDF_URL)

    parser.add_argument("--mfimage-config",
                        default=os.path.sep + "obitconf" + os.path.sep + "mfimage.yaml",
                        type=str,
                        help="Configuration yaml file for MFImage. Default: %(default)s")

    parser.add_argument("--nif", default=8, type=int,
                        help="Number of AIPS 'IFs' to equally subdivide the band. "
                             "NOTE: Must divide the number of channels. "
                             "Default: %(default)s")

    parser.add_argument("--log-level", type=str, default="INFO",
                        help='Logging level. Default: %(default)s')

    parser.add_argument("--pols", default="HH,VV", type=str,
                        help="Which polarisations to copy from the archive. "
                             "Default: %(default)s")

    parser.add_argument("-ks", "--select",
                        default="scans='track'; corrprods='cross'",
                        type=log_exception(log)(parse_python_assigns),
                        help="katdal select statement "
                             "Should only contain python "
                             "assignment statements to python "
                             "literals, separated by semi-colons.")

    parser.add_argument("--open-args", default="",
                        type=log_exception(log)(parse_python_assigns),
                        help="kwargs to pass to katdal.open() "
                             "Should only contain python "
                             "assignment statements to python "
                             "literals, separated by semi-colons.")

    parser.add_argument("--clobber",
                        default="scans, avgscans",
                        type=lambda s: set(v.strip() for v in s.split(',')),
                        help="Class of AIPS/Obit output files to clobber. "
                             "'scans' => Individual scans. "
                             "'avgscans' => Averaged individual scans. "
                             "'merge' => Observation file containing merged, "
                             "averaged scans. "
                             "'clean' => Output CLEAN files. "
                             "'mfimage' => Output MFImage files.")

    parser.add_argument("--prtlv", default=2, type=int,
                        help="Integer between 0 and 5 indicating the desired "
                             "verbosity of Obit tasks. 0=None 5=Maximum. "
                             "Default: %(default)s")

    return parser

parser = create_parser()
args = parser.parse_args()
configure_logging(args)

katdata = katdal.open(args.katdata, applycal=args.applycal, **args.open_args)

# Set up katdal selection based on arguments
kat_select = {'pol': args.pols,
              'nif': args.nif}

if args.targets:
    kat_select['targets'] = args.targets
if args.channels:
    start_chan, end_chan = split(args.channels,',')
    kat_select['channel'] = slice(start_chan, end_chan)

# Command line katdal selection overrides command line options
kat_select = recursive_merge(args.select, kat_select)

# Get defaults for uvblavg and mfimage and merge user supplied ones
uvblavg_args = get_and_merge_args(args.uvblavg_config, args.uvblavg)
mfimage_args = get_and_merge_args(args.mfimage_config, args.mfimage)

# Get the default config.
dc = kc.get_config()

# capture_block_id is used to generate AIPS disk filenames
capture_block_id = katdata.obs_params['capture_block_id']

# Set up aipsdisk configuration from args.workdir
aipsdirs = [(None, os.path.join(args.workdir, capture_block_id + '_aipsdisk'))]
log.info('Using AIPS data area: %s' % (aipsdirs[0][1]))

# Set up output configuration from args.outputdir
fitsdirs = dc['fitsdirs']

# Append outputdir to fitsdirs
fitsdirs += [(None, args.outputdir)]
log.info('Using output data area: %s' % (args.outputdir))

kc.set_config(aipsdirs=aipsdirs, fitsdirs=fitsdirs,
              output_id='', cb_id=capture_block_id)

setup_aips_disks()

pipeline = ImagePipeline(katdata,
                         katdal_select=kat_select,
                         uvblavg_params=uvblavg_args,
                         mfimage_params=mfimage_args,
                         nvispio=args.nvispio,
                         clobber=args.clobber,
                         prtlv=args.prtlv)

# Execute it
pipeline.execute()
