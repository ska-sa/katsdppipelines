import argparse
import logging
from pprint import pprint

import katdal

import ObitTalkUtil
import UV

import katsdpcontim
from katsdpcontim import KatdalAdapter, UVFacade, handle_obit_err, obit_context
from katsdpcontim.uvfits_utils import open_aips_file_from_fits_template

logging.basicConfig(level=logging.INFO)

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

    pprint(uv.Desc.Dict)

    uv = UVFacade(uv)
    uv.update_descriptor(KA.uv_descriptor())
    uv.create_antenna_table(KA.uv_antenna_header, KA.uv_antenna_rows)
    uv.create_frequency_table(KA.uv_spw_header, KA.uv_spw_rows)
    uv.create_source_table(KA.uv_source_header, KA.uv_source_rows)

    import itertools
    import time

    import numpy as np
    import six

    ntime, nchans, ncorrprods = K.shape
    nstokes = KA.nstokes
    cp = KA.correlator_products()

    # Lexicographically sort correlation products on (a1, a2, cid)
    sort_fn = lambda x: (cp[x].ant1_index,cp[x].ant2_index,cp[x].cid)
    cp_argsort = np.asarray(sorted(range(len(cp)), key=sort_fn))
    corr_products = [cp[i] for i in cp_argsort]
    refwave = KA.refwave

    # AIPS baseline ID for each data product
    aips_baselines = np.asarray([cp.ant1_index*256.0 + cp.ant2_index
                                                for cp in corr_products],
                                                        dtype=np.float32)

    # Derive starting time in unix seconds
    tm = K.timestamps[1:2][0]
    tx = time.gmtime(tm)
    time0 = tm - tx[3]*3600.0 - tx[4]*60.0 - tx[5]

    for scan, state, target in K.scans():
        # Retrieve scan data (ntime, nchan, nbl*npol), casting to float32
        # nbl*npol is all mixed up though
        times = K.timestamps[:]
        vis = K.vis[:].astype(np.complex64)
        weights = K.weights[:].astype(np.float32)
        flags = K.flags[:]

        print vis.shape
        print weights.shape
        print flags.shape

        ntimes = len(times)

        # This rearranges so that we have (ntime, nchan, nbl, npol)
        vis = vis[:,:,cp_argsort].reshape(ntimes, nchans, -1, nstokes)
        weights = weights[:,:,cp_argsort].reshape(ntimes, nchans, -1, nstokes)
        flags = flags[:,:,cp_argsort].reshape(ntimes, nchans, -1, nstokes)

        # This transposes so that we have (ntime, nbl, npol, nchan)
        vis = vis.transpose(0,2,3,1)
        weights = weights.transpose(0,2,3,1)
        flags = flags.transpose(0,2,3,1)

        print vis.shape, vis.nbytes / (1024.*1024.)
        print weights.shape, weights.nbytes / (1024.*1024.)
        print flags.shape, flags.nbytes / (1024.*1024.)

        # Apply flags by negating weights
        weights[np.where(flags)] = -32767.0

        # Compute UVW coordinates from correlator products
        # (3, ntimes, nbl, npol)
        uvw = (np.stack([target.uvw(cp.ant1, antenna=cp.ant2, timestamp=times)
                            for cp in corr_products], axis=2)
                                .reshape(3, ntimes, -1, nstokes))

        # UVW coordinates (in frequency?)
        aips_uvw = uvw / refwave
        # Timestamps in days
        aips_time = (times - time0) / 86400.0

