import logging

import numpy as np

from katsdpcontim import (KatdalAdapter, UVFacade,
                        uv_factory)

log = logging.getLogger('katsdpcontim')


def uv_export(kat_adapter, obit_file, kat_select=None):
    """
    Exports a katdal selection to an AIPS/FITS file.

    Parameters
    ----------
    kat_adapter: :class:`KatdalAdapter`
        Katdal Adapter supplying data for export
    obit_file: :class:`ObitFile`
        Obit file to which data should be exported
    kat_select (optional): dict
        Dictionary of keyword arguments to apply
        to katdal selection. Defaults to
        :code:`{ "scans" : "track", "spw": 0 }`
    """
    if kat_select is None:
        kat_select = { "scans" : "track", "spw": 0 }

    nvispio = 1024

    KA = kat_adapter

    # UV file location variables
    with uv_factory(obit_file=obit_file, mode="w",
                    katdata=KA, nvispio=nvispio) as uvf:

        log.info("Created '%s'" % obit_file)
        firstVis = 1    # FORTRAN indexing
        numVisBuff = 0  # Number of visibilities in the buffer

        # NX table rows
        nx_rows = []

        # Perform selection on the katdal object
        KA.select(**kat_select)

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
                    if numVisBuff == nvispio:
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
        uvf.attach_CL_from_NX_table(KA.max_antenna_number)
