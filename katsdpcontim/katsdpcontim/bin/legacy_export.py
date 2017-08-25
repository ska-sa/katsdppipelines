"""
This is a stripped down version of h5touvfits.py from
the original Obit pipeline, ``katim`` and upgraded to
use ``katsdpcontim's`` Obit environment handling.

It exists to test correctness of export against the
newer export in katsdpcontim.
Just like ``h5touvfits``, it uses a FITS template file
as a basis for creating a file on an AIPS disk.
Aside from this, most of the export logic exists in
:meth:`katim.KATH5toAIPS.KAT2AIPS`.
"""

#! /usr/bin/env python
import argparse
import os
import shutil

import numpy as np
import katdal
import pyfits

import OTObit, ObitTalkUtil
import AIPS, UV

from katim import KATH5toAIPS
from katim import KATCal

from katsdpcontim import obit_context, obit_err, handle_obit_err
from katsdpcontim.util import parse_katdal_select

import warnings
warnings.simplefilter('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("katdata", help="hdf5 observation file")
    parser.add_argument("-l", "--label", default="MeerKAT")
    parser.add_argument("-n", "--name", help="AIPS name")
    parser.add_argument("-c", "--class", default="legacy", dest="aclass",
                                        help="AIPS class")
    parser.add_argument("-d", "--disk", default=1,
                                        help="AIPS disk")
    parser.add_argument("-s", "--seq", default=1,
                                        help="AIPS sequence")
    parser.add_argument("-ks", "--select", default="scans='track';spw=0",
                                        type=parse_katdal_select,
                                        help="katdal select statement "
                                             "Should only contain python "
                                             "assignment statements to python "
                                             "literals, separated by semi-colons")

    return parser

args = create_parser().parse_args()

if args.name is None:
    path, ext = os.path.splitext(args.katdata)
    args.name = os.path.basename(path)

with obit_context():
    err = obit_err()
    fitsdisk = 0
    uv_name = '.'.join((args.name, 'uvfits'))

    katdata = katdal.open(args.katdata)
    katdata.select(**args.select)
    numchans = len(katdata.channel_freqs)

    # Condition the uvfits template
    templatefile=ObitTalkUtil.FITSDir.FITSdisks[fitsdisk]+'MKATTemplate.uvtab.gz'

    uvfits = pyfits.open(templatefile)
    # Resize the visibility table
    vistable = uvfits[1].columns
    vistable.del_col('VISIBILITIES')
    newvis = pyfits.Column(name='VISIBILITIES',
            format='%dE'%(3*4*numchans),
            dim='(3,4,%d,1,1,1)'%(numchans,),
            array=np.zeros((1,1,1,numchans,4,3,),
            dtype=np.float32))
    vistable.add_col(newvis)
    vishdu = pyfits.BinTableHDU.from_columns(vistable)
    for key in uvfits[1].header.keys():
        if (key not in vishdu.header.keys()) and (key != 'HISTORY'):
            vishdu.header[key] = uvfits[1].header[key]

    newuvfits = pyfits.HDUList([uvfits[0], vishdu, uvfits[2], uvfits[3],
                                uvfits[4], uvfits[5], uvfits[6]])
    newuvfits.writeto(uv_name, clobber=True)
    uv = OTObit.uvlod(uv_name, 0, args.name, args.aclass,
                    args.disk, args.seq, err)

    obsdata = KATH5toAIPS.KAT2AIPS(katdata, uv,
                                    args.disk, fitsdisk, err,
        calInt=1.0, stop_w=False, apply_cal=False)

    uv.Header(err)
    handle_obit_err("Error calling Header", err)