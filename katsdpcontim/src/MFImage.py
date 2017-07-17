import argparse
import logging
import os
import os.path

import six

import OTObit
import ObitTask

from aips_context import aips_context
from aips_parser import parse_aips_config

logging.basicConfig(level=logging.INFO)

def create_parser():
    """ Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-c", "--config", required=True)
    return parser

def create_mf_imager(args):
    """ Create MF Image """
    imager = ObitTask.ObitTask("MFImage")

    path, file  = os.path.split(args.input)
    base_filename, ext = os.path.splitext(file)

    try:
        # Set the AIPS user
        imager.userno = OSystem.PGetAIPSuser()
    except Exception as e:
        logging.exception("Exception setting AIPS User")

    # Get aips config
    cfg = parse_aips_config(args.config)

    # Override AIPS disk configuration options
    cfg.update({
        "DataType": "FITS",
        "FITSdirs": [os.getcwd()],
        "AIPSdirs": [os.getcwd()],
        "inFile": args.input,
        "inDisk": 0,
        "inSeq": 0,
        "outFile": ".out.fits",
        "out2File": ".out.fits",
    })

    # Set output disk options to input disk options
    cfg.update({
        "outDType": cfg["DataType"],
        "outDisk": cfg["inDisk"],
        "outSeq": cfg["inSeq"],
        "out2Disk": cfg["inDisk"],
        "out2Seq": cfg["inSeq"],
    })

    # This don't work with OBIT for some reason
    for k in ('nFITS','nAIPS', 'AIPSuser'):
        cfg.pop(k, None)

    # Dump configuration options onto the Imager Task
    for k, v in six.iteritems(cfg):
        setattr(imager, k, v)

    return imager

args = create_parser().parse_args()
with aips_context():
    imager = create_mf_imager(args)

    print imager.Sources

    imager.g

