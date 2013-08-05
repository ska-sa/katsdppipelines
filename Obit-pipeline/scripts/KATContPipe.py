#! /usr/bin/env python
import sys
from katim import KATPipe
from optparse import OptionParser
from ConfigParser import NoSectionError

usage = "%prog [options] h5fileToImage"
description = "Make an image from an h5 file."
parser = OptionParser( usage=usage, description=description)
parser.add_option("--parms", default=None, help="Overwrite the default imaging parameters using a parameter file.")
parser.add_option("--outputdir", default='./', help="Specify the output data directory.")
parser.add_option("--scratchdir", default=None, help="Specify the scratch directory.")
parser.add_option("--targets", default=None, help="List of targets to load (You'll need calibrators in this list!!)")
(options, args) = parser.parse_args()
if len(args) < 1:
    parser.print_help()
    sys.exit()
try:
    KATPipe.K7ContPipeline(args,parmFile=options.parms,outputdir=options.outputdir,scratchdir=options.scratchdir,targets=options.targets)
finally:
    pass
