import argparse
from pprint import pprint

import katdal

import ObitTalkUtil
import UV

import katsdpcontim
from katsdpcontim import KatdalAdapter, UVFacade, handle_obit_err, obit_context
from katsdpcontim.uvfits_utils import open_aips_file_from_fits_template

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5file")
    return parser

args = create_parser().parse_args()

K = katdal.open(args.h5file)
KA = KatdalAdapter(K)

pprint({k: v for k, v in KA._targets.items()})

pprint(KA._classify_targets())

descriptor = KA.uv_descriptor()

pprint(descriptor)
pprint(KA.uv_antenna_rows)
pprint(KA.uv_spw_rows)
pprint(KA.uv_source_rows)

with obit_context():
    print ObitTalkUtil.ListAIPSDirs()
    print ObitTalkUtil.ListFITSDirs()

    uv = open_aips_file_from_fits_template(0, 'test')
    uv = UVFacade(uv)
    uv.update_descriptor(KA.uv_descriptor())
    uv.create_antenna_table(KA.uv_antenna_header, KA.uv_antenna_rows)
    uv.create_frequency_table(KA.uv_spw_header, KA.uv_spw_rows)
    uv.create_source_table(KA.uv_source_header, KA.uv_source_rows)