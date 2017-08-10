import argparse
import logging
from pprint import pprint

import numpy as np

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

nVisPIO = 1024

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

    # Set number of visibilities read/written at a time
    uv.List.set("nVisPIO", nVisPIO)

    # WRITEONLY correctly creates a buffer on the UV object
    # READWRITE only creates a buffer
    # on the UV object if the underlying file exists...
    uv.Open(UV.WRITEONLY, err)
    handle_obit_err("Error opening UV file", err)

    # Configure number of visibilities written in a batch
    desc = uv.Desc.Dict

    # Number of random parameters
    nrparm = desc['nrparm']

    # Random parameter indices
    ilocu = desc['ilocu']     # U
    ilocv = desc['ilocv']     # V
    ilocw = desc['ilocw']     # W
    iloct = desc['iloct']     # time
    ilocb = desc['ilocb']     # baseline id
    ilocsu = desc['ilocsu']   # source id

    inaxes = tuple(desc['inaxes'][:6])  # Visibility shape, strip out trailing 0
    flat_inaxes = np.product(inaxes)
    lrec = nrparm + flat_inaxes         # Length of record in vis buffer

    import itertools
    import time

    import numpy as np
    import six

    nstokes = KA.nstokes
    cp = KA.correlator_products()

    # Lexicographically sort correlation products on (a1, a2, cid)
    sort_fn = lambda x: (cp[x].ant1_index,cp[x].ant2_index,cp[x].cid)
    cp_argsort = np.asarray(sorted(range(len(cp)), key=sort_fn))
    corr_products = [cp[i] for i in cp_argsort]
    refwave = KA.refwave

    # AIPS baseline ID for each data product
    aips_baselines_flat = np.asarray([cp.ant1_index*256.0 + cp.ant2_index
                                                for cp in corr_products],
                                                        dtype=np.float32)

    uv_source_map = KA.uv_source_map

    # Derive starting time in unix seconds
    time0 = K.timestamps[0]

    for si, (scan, state, target) in enumerate(K.scans()):
        # Retrieve UV source information for this scan
        try:
            aips_source = uv_source_map[target.name]
        except KeyError:
            logging.warn("Target '{}' will not be exported".format(target.name))
            continue
        else:
            # Retrieve the source ID
            aips_source_id = aips_source['ID. NO.'][0]

        # Retrieve scan data (ntime, nchan, nbl*npol), casting to float32
        # nbl*npol is all mixed up at this point
        times = K.timestamps[:]
        vis = K.vis[:].astype(np.complex64)
        weights = K.weights[:].astype(np.float32)
        flags = K.flags[:]

        # Get dimension shapes
        ntime, nchan, ncorrprods = vis.shape

        # Apply flags by negating weights
        weights[np.where(flags)] = -32767.0

        # AIPS visibility is [real, imag, weight]
        # (ntime, 3, nchan, nbl*npol)
        vis = np.stack([vis.real, vis.imag, weights], axis=1)

        # Reorganise correlation product dim so that
        # polarisations are grouped per baseline.
        # Then split correlations into baselines and polarisation
        # producing (ntime, 3, nchan, nbl, npol)
        vis = vis[:,:,:,cp_argsort].reshape(ntime, 3, nchan, -1, nstokes)

        # This transposes so that we have (ntime, nbl, 3, npol, nchan)
        vis = vis.transpose(0,3,1,4,2)
        # Get dimensions, mainly to get baselines
        ntime, nbl, _, _, nchan = vis.shape

        # Reshape to introduce nif, ra and dec
        # It's now (ntime, nbl, 3, npol, nchan, nif, ra, dec)
        vis = np.reshape(vis, (ntime, nbl) + inaxes)

        print "Visibilities of shape {} and size {:.2f}MB".format(vis.shape, vis.nbytes / (1024.*1024.))

        # Compute UVW coordinates from correlator products
        # (3, ntimes, nbl, npol)
        uvw = (np.stack([target.uvw(cp.ant1, antenna=cp.ant2, timestamp=times)
                            for cp in corr_products], axis=2)
                                .reshape(3, ntime, -1, nstokes))

        # Get the shape
        aips_baselines = aips_baselines_flat.reshape(nbl, nstokes)[:,0]

        # UVW coordinates (in frequency?)
        aips_uvw = uvw / refwave
        # Timestamps in days
        aips_time = (times - time0) / 86400.0

        def _write_buffer(uv, firstVis, numVisBuff):
            """ Handle writes """
            # Update descriptor
            desc = uv.Desc.Dict
            desc['numVisBuff'] = numVisBuff
            # If firstVis is passed through via the descriptor it uses C indexing (0)
            # desc['firstVis'] = firstVis - 1
            uv.Desc.Dict = desc

            nbytes = numVisBuff*lrec*np.dtype(np.float32).itemsize
            logging.info("Writing {:.2f}MB visibilities".format(nbytes / (1024.*1024.)))

            # If firstVis is passed through to this method, it uses FORTRAN indexing (1)
            uv.Write(err, firstVis=firstVis)
            handle_obit_err("Error writing UV file", err)

        vis_buffer = np.frombuffer(uv.VisBuf, count=-1, dtype=np.float32)

        firstVis = 0
        numVisBuff = 0

        for t in range(ntime):
            for bl in range(nbl):
                idx = numVisBuff*lrec
                # UVW coordinates are the same for each stokes parameter
                vis_buffer[idx+ilocu] = uvw[0,t,bl,0]
                vis_buffer[idx+ilocv] = uvw[1,t,bl,0]
                vis_buffer[idx+ilocw] = uvw[2,t,bl,0]
                vis_buffer[idx+iloct] = aips_time[t]
                vis_buffer[idx+ilocb] = aips_baselines[bl]
                vis_buffer[idx+ilocsu] = aips_source_id
                vis_buffer[idx+nrparm:idx+nrparm+flat_inaxes] = vis[t,bl].ravel()

                numVisBuff += 1
                firstVis += 1

                # Hit the limit, write
                if numVisBuff == nVisPIO:
                    _write_buffer(uv, firstVis, numVisBuff)
                    numVisBuff = 0

        # Write out any remaining visibilities
        if numVisBuff > 0:
            _write_buffer(uv, firstVis, numVisBuff)
            numVisBuff = 0
