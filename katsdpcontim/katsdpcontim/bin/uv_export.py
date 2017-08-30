import argparse
import logging
import os
import os.path
from pprint import pprint

import numpy as np

import UV

import katdal

import katsdpcontim
from katsdpcontim import (KatdalAdapter, UVFacade,
                        uv_factory,
                        handle_obit_err, obit_context,
                        obit_err,)
from katsdpcontim.util import parse_katdal_select

log = logging.getLogger('katsdpcontim')

# uv_export.py -n pks1934 /var/kat/archive2/data/MeerKATAR1/telescope_products/2017/07/15/1500148809.h5

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("katdata", help="hdf5 observation file")
    parser.add_argument("-l", "--label", default="MeerKAT")
    parser.add_argument("-n", "--name", help="AIPS name")
    parser.add_argument("-c", "--class", default="raw", dest="aclass",
                                        help="AIPS class")
    parser.add_argument("-d", "--disk", default=1,
                                        help="AIPS disk")
    parser.add_argument("-s", "--seq", default=1,
                                        help="AIPS sequence")
    parser.add_argument("--nvispio", default=1024,
                                        help="Number of visibilities "
                                             "read/written per IO call")
    parser.add_argument("-ks", "--select", default="scans='track';spw=0",
                                        type=parse_katdal_select,
                                        help="katdal select statement "
                                             "Should only contain python "
                                             "assignment statements to python "
                                             "literals, separated by semi-colons")
    return parser

args = create_parser().parse_args()

# Use the hdf5file name if none is supplied
if args.name is None:
    path, file  = os.path.split(args.katdata)
    base_filename, ext = os.path.splitext(file)
    args.name = base_filename

KA = KatdalAdapter(katdal.open(args.katdata))

with obit_context():
    err = obit_err()

    # UV file location variables
    uvf = uv_factory(dtype="AIPS", name=args.name, disk=args.disk,
                    aclass=args.aclass, seq=args.seq, mode="w",
                    katdata=KA, nvispio=args.nvispio)

    log.info("Created '%s' on AIPS disk '%d'" % (uvf.name, args.disk))
    firstVis = 1    # FORTRAN indexing
    numVisBuff = 0  # Number of visibilities in the buffer

    # NX table rows
    nx_rows = []

    # Perform selection on the katdal object
    KA.select(**args.select)

    for si, (u, v, w, time, baselines, source_id, vis) in KA.uv_scans():

        def _write_buffer(uvf, firstVis, numVisBuff):
            """
            Use as follows:

            .. code-block:: python

                firstVis, numVisBuff = _write_buffer(uv, firstVis, numVisBuff)

            Parameters
            ----------
            uvf: :class:`UVFacade` object
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
            desc = uvf.Desc.Dict
            desc.update(numVisBuff=numVisBuff)
            uvf.Desc.Dict = desc

            nbytes = numVisBuff*lrec*np.dtype(np.float32).itemsize
            log.info("Writing {:.2f}MB visibilities. firstVis={} numVisBuff={}."
                .format(nbytes / (1024.*1024.), firstVis, numVisBuff))

            # If firstVis is passed through to this method, it uses FORTRAN indexing (1)
            uvf.Write(firstVis=firstVis)

            # Pass through new firstVis and 0 numVisBuff
            return firstVis + numVisBuff, 0

        # Starting visibility of this scan
        start_vis = firstVis
        vis_buffer = np.frombuffer(uvf.VisBuf, count=-1, dtype=np.float32)

        # Number of random parameters
        desc = uvf.Desc.Dict
        nrparm = desc['nrparm']
        lrec = desc['lrec']       # Length of visibility buffer record

        # Random parameter indices
        ilocu = desc['ilocu']     # U
        ilocv = desc['ilocv']     # V
        ilocw = desc['ilocw']     # W
        iloct = desc['iloct']     # time
        ilocb = desc['ilocb']     # baseline id
        ilocsu = desc['ilocsu']   # source id

        ntime, nbl = u.shape

        for t in range(ntime):
            for bl in range(nbl):
                # Index within vis_buffer
                idx = numVisBuff*lrec

                # Write random parameters
                vis_buffer[idx+ilocu] = u[t,bl]           # U
                vis_buffer[idx+ilocv] = v[t,bl]           # V
                vis_buffer[idx+ilocw] = w[t,bl]           # W
                vis_buffer[idx+iloct] = time[t]           # time
                vis_buffer[idx+ilocb] = baselines[bl]     # baseline id
                vis_buffer[idx+ilocsu] = source_id        # source id

                # Flatten visibilities for buffer write
                flat_vis = vis[t,bl].ravel()
                vis_buffer[idx+nrparm:idx+nrparm+flat_vis.size] = flat_vis

                numVisBuff += 1

                # Hit the limit, write
                if numVisBuff == args.nvispio:
                    firstVis, numVisBuff = _write_buffer(uvf, firstVis, numVisBuff)

        # Write out any remaining visibilities
        if numVisBuff > 0:
            firstVis, numVisBuff = _write_buffer(uvf, firstVis, numVisBuff)

        # Create an index for this scan
        nx_rows.append({
            'TIME': [(time[-1] + time[0])/2], # Time Centroid
            'TIME INTERVAL': [time[-1] - time[0]],
            'SOURCE ID': [source_id],
            'SUBARRAY': [1],           # Should match 'AIPS AN' table version
                                       # Each AN table defines a subarray
            'FREQ ID': [1],            # Should match 'AIPS FQ' row FRQSEL
            'START VIS': [start_vis],  # FORTRAN indexing
            'END VIS': [firstVis-1]    # FORTRAN indexing
        })

    # Create the index and calibration tables
    uvf.attach_table("AIPS NX", 1)
    uvf.tables["AIPS NX"].rows = nx_rows
    uvf.tables["AIPS NX"].write()
    uvf.create_calibration_table_from_index(KA.max_antenna_number)

    uvf.close()