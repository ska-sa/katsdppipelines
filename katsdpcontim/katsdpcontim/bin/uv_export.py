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
    inlabel = "myuv"
    inname = "stuff"
    inclass = "raw"
    inseq = 1
    indisk = 1
    uv = UV.newPAUV(inlabel, inname, inclass, indisk, inseq, False, err)
    uv.Open(UV.READWRITE, err)
    handle_obit_err("Error creating UV file", err)

    uvf = UVFacade(uv)
    uvf.create_antenna_table(KA.uv_antenna_header, KA.uv_antenna_rows)
    uvf.create_frequency_table(KA.uv_spw_header, KA.uv_spw_rows)
    uvf.create_source_table(KA.uv_source_header, KA.uv_source_rows)
    uvf.create_calibration_table(KA.uv_calibration_header, {})

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
    corr_products = np.asarray([cp[i] for i in cp_argsort])
    refwave = KA.refwave

    # Take baseline products so that we don't recompute
    # UVW coordinates for all correlator products
    bl_products = corr_products.reshape(-1, nstokes)[:,0]
    nbl, = bl_products.shape

    # AIPS baseline IDs
    aips_baselines = np.asarray([bp.ant1_index*256.0+bp.ant2_index for bp
                                                            in bl_products],                                                            dtype=np.float32)
    # UV file location variables
    firstVis = 1    # FORTRAN indexing
    numVisBuff = 0  # Number of visibilities in the buffer

    uv_source_map = KA.uv_source_map

    # Get starting time
    time0 = K.start_time.secs

    # NX table rows
    nx_rows = []

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
        vis = vis[:,:,:,cp_argsort].reshape(ntime, 3, nchan, nbl, nstokes)

        # This transposes so that we have (ntime, nbl, 3, npol, nchan)
        vis = vis.transpose(0,3,1,4,2)

        # Reshape to introduce nif, ra and dec
        # It's now (ntime, nbl, 3, npol, nchan, nif, ra, dec)
        vis = np.reshape(vis, (ntime, nbl) + inaxes)

        print "Visibilities of shape {} and size {:.2f}MB".format(
                    vis.shape, vis.nbytes / (1024.*1024.))

        # Compute UVW coordinates from baselines
        # (3, ntimes, nbl)
        uvw = (np.stack([target.uvw(bp.ant1, antenna=bp.ant2, timestamp=times)
                                                for bp in bl_products], axis=2)
                                .reshape(3, ntime, nbl))

        # UVW coordinates (in frequency?)
        aips_uvw = uvw / refwave
        # Timestamps in days
        aips_time = (times - time0) / 86400.0

        def _write_buffer(uv, firstVis, numVisBuff):
            """
            Use as follows:

            .. code-block:: python

                firstVis, numVisBuff = _write_buffer(uv, firstVis, numVisBuff)

            Parameters
            ----------
            uv: Obit UV object
            firstVis: integer
                First visibility to write in the file (FORTRAN indexing)
            numVisBuff: integer
                Number of visibilities to write to the file.

            Returns
            -------
            tuple
                (firstVis + numVisBuff, 0)

            """
            # Update descriptor
            desc = uv.Desc.Dict
            desc['numVisBuff'] = numVisBuff
            # If firstVis is passed through via the descriptor it uses C indexing (0)
            # desc['firstVis'] = firstVis - 1
            uv.Desc.Dict = desc

            nbytes = numVisBuff*lrec*np.dtype(np.float32).itemsize
            logging.info("firstVis={} numVisBuff={} Writing {:.2f}MB visibilities".format(
                firstVis, numVisBuff, nbytes / (1024.*1024.)))

            # If firstVis is passed through to this method, it uses FORTRAN indexing (1)
            uv.Write(err, firstVis=firstVis)
            handle_obit_err("Error writing UV file", err)

            # Pass through firstVis and 0 numVisBuff
            return firstVis + numVisBuff, 0

        # Starting visibility of this scan
        start_vis = firstVis
        vis_buffer = np.frombuffer(uv.VisBuf, count=-1, dtype=np.float32)

        for t in range(ntime):
            for bl in range(nbl):
                # Index within vis_buffer
                idx = numVisBuff*lrec

                # Write random parameters
                vis_buffer[idx+ilocu] = uvw[0,t,bl]        # U
                vis_buffer[idx+ilocv] = uvw[1,t,bl]        # V
                vis_buffer[idx+ilocw] = uvw[2,t,bl]        # W
                vis_buffer[idx+iloct] = aips_time[t]       # time
                vis_buffer[idx+ilocb] = aips_baselines[bl] # baseline id
                vis_buffer[idx+ilocsu] = aips_source_id    # source id

                flat_vis = vis[t,bl].ravel()
                vis_buffer[idx+nrparm:idx+nrparm+flat_vis.size] = flat_vis

                numVisBuff += 1

                # Hit the limit, write
                if numVisBuff == nVisPIO:
                    firstVis, numVisBuff = _write_buffer(uv, firstVis, numVisBuff)

        # Write out any remaining visibilities
        if numVisBuff > 0:
            firstVis, numVisBuff = _write_buffer(uv, firstVis, numVisBuff)

        # Create an index for this scan
        nx_rows.append({
            # Book-keeping
            'Table name': 'AIPS NX',
            'NumFields': 8,
            '_status': [0],

            'TIME': [(aips_time[-1] - aips_time[0])/2],      # Time Centroid
            'TIME INTERVAL': [aips_time[-1] - aips_time[0]],
            'SOURCE ID': [aips_source_id],
            'SUBARRAY': [1],                  # Should match 'AIPS AN' header
            'FREQ ID': [1],                   # Should match 'AIPS FQ' row
            'START VIS': [start_vis],
            'END VIS': [firstVis-1]
        })

    # Create the index table...
    uvf.create_index_table({}, nx_rows)

    uv.Close(err)
    handle_obit_err("Error closing UV file", err)
