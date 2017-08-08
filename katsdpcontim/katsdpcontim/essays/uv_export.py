import argparse
import logging
from pprint import pprint

import katdal

import ObitTalkUtil
import UV

import katsdpcontim
from katsdpcontim import KatdalAdapter, UVFacade, handle_obit_err, obit_context, obit_err
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

    err = obit_err()

    # Create the AIPS UV file
    uv = UV.newPAUV("myuv", "stuff", "Raw", 1, 1, False, err)
    uv.Open(UV.READWRITE, err)
    handle_obit_err("Error creating UV file", err)

    uvf = UVFacade(uv)
    uvf.create_antenna_table(KA.uv_antenna_header, KA.uv_antenna_rows)
    uvf.create_frequency_table(KA.uv_spw_header, KA.uv_spw_rows)
    uvf.create_source_table(KA.uv_source_header, KA.uv_source_rows)

    # Reopen the file
    # uv.Open(UV.READWRITE, err)
    # handle_obit_err("Error Opening UV file", err)

    # Update the UV descriptor with MeerKAT metadata
    uvf.update_descriptor(KA.uv_descriptor())

    # Write 1024 visibilities at a time
    uv.List.set("nVisPIO", 1024)

    # WRITEONLY correctly creates a buffer on the UV object
    # READWRITE only creates a buffer
    # on the UV object if the underlying file exists...
    uv.Open(UV.WRITEONLY, err)
    handle_obit_err("Error opening UV file", err)

    # Configure number of visibilities written in a batch
    d = uv.Desc.Dict
    d['nvis'] = 1024          # Max vis written
    d['numVisBuff'] = 1024    # NumVisBuff is actual number of vis written
    uv.Desc.Dict = d

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

    for i, (scan, state, target) in enumerate(K.scans()):
        # Retrieve scan data (ntime, nchan, nbl*npol), casting to float32
        # nbl*npol is all mixed up at this point
        times = K.timestamps[:]
        vis = K.vis[:].astype(np.complex64)
        weights = K.weights[:].astype(np.float32)
        flags = K.flags[:]

        # Apply flags by negating weights
        weights[np.where(flags)] = -32767.0

        # AIPS visibility is [real, imag, weight]
        # (ntime, 3, nchan, nbl*npol)
        vis = np.stack([vis.real, vis.imag, weights], axis=1)

        ntimes = len(times)

        # Reorganise correlation product dim so that
        # polarisations are grouped per baseline.
        # Then split correlations into baselines and polarisation
        # producing (ntime, 3, nchan, nbl, npol)
        vis = vis[:,:,:,cp_argsort].reshape(ntimes, 3, nchans, -1, nstokes)

        # This transposes so that we have (ntime, nbl, 3, npol, nchan)
        vis = vis.transpose(0,3,1,4,2)

        print vis.shape, vis.nbytes / (1024.*1024.)

        # Compute UVW coordinates from correlator products
        # (3, ntimes, nbl, npol)
        uvw = (np.stack([target.uvw(cp.ant1, antenna=cp.ant2, timestamp=times)
                            for cp in corr_products], axis=2)
                                .reshape(3, ntimes, -1, nstokes))

        # UVW coordinates (in frequency?)
        aips_uvw = uvw / refwave
        # Timestamps in days
        aips_time = (times - time0) / 86400.0

        vis_buffer = np.frombuffer(uv.VisBuf, count=-1, dtype=np.float32)
        print vis_buffer.shape

        # Just write the visibility buffer back for the moment.
        # Likely very wrong, but test writes.
        uv.Write(err, firstVis=i+1)
        handle_obit_err("Error opening UV file", err)
