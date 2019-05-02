import logging

import katdal
import katpoint
import numpy as np

import OTObit

from katacomb.katdal_adapter import time_chunked_scans
from katacomb.util import fmt_bytes

log = logging.getLogger('katacomb')


def _write_buffer(uvf, firstVis, numVisBuff, lrec):
    """
    Use as follows:

    .. code-block:: python

        firstVis, numVisBuff = _write_buffer(uv, firstVis, numVisBuff, lrec)

    Parameters
    ----------
    uvf : :class:`UVFacade` object
    firstVis : integer
        First visibility to write in the file (FORTRAN indexing)
    numVisBuff : integer
        Number of visibilities to write to the file.
    lrec : integer
        Length of a visibility record in bytes.

    Returns
    -------
    tuple
        (firstVis + numVisBuff, 0)

    """
    # Update descriptor
    desc = uvf.Desc.Dict
    desc.update(numVisBuff=numVisBuff)
    uvf.Desc.Dict = desc

    nbytes = numVisBuff * lrec * np.dtype(np.float32).itemsize
    log.debug("Writing '%s' visibilities. firstVis=%s numVisBuff=%s",
              fmt_bytes(nbytes), firstVis, numVisBuff)

    # If firstVis is passed through to this method, it uses FORTRAN
    # indexing (1)
    uvf.Write(firstVis=firstVis)

    # Pass through new firstVis and 0 numVisBuff
    return firstVis + numVisBuff, 0


def uv_history_obs_description(kat_adapter, uvf):
    """
    Record MeerKAT observation metadata in AIPS history

    Parameters
    ----------
    katdal_adapter : :class:`katacomb.KatdalAdapter`
        Katdal Adapter
    uvf : :class:`katacomb.UVFacade`
        UV file
    """

    uvf.append_history("="*70)
    uvf.append_history("MEERKAT OBSERVATION")
    uvf.append_history("="*70)
    uvf.append_history("name=%s" % kat_adapter.name)
    uvf.append_history("experiment_id=%s" % kat_adapter.experiment_id)
    uvf.append_history("description=%s" % kat_adapter.description)
    uvf.append_history("observer=%s" % kat_adapter.observer)
    uvf.append_history("date=%s" % kat_adapter.obsdat)
    uvf.append_history("katdal.__version__=%s" % katdal.__version__)
    uvf.append_history("katpoint.__version__=%s" % katpoint.__version__)


def uv_history_selection(selection, uvf):
    """
    Record katdal data selection parameters in AIPS history

    Parameters
    ----------
    selection : dict
        katdal selection parameters
    uvf : :class:`katacomb.UVFacade`
        UV file
    """

    uvf.append_history("="*70)
    uvf.append_history("KATDAL SELECTION PARAMETERS")
    uvf.append_history("="*70)

    for k, v in selection.items():
        uvf.append_history("%s=%s" % (k, v))


def uv_export(kat_adapter, uvf):
    """
    Exports data in a katdal selection to an AIPS/FITS file.

    Parameters
    ----------
    katdal_adapter : :class:`katacomb.KatdalAdapter`
        Katdal Adapter
    uvf : :class:`katacomb.UVFacade`
        UV file
    """
    firstVis = 1            # FORTRAN indexing
    numVisBuff = 0          # Number of visibilities in the buffer

    # Number of visibilities per IO operation
    type_, dims, value = uvf.List.Dict['nVisPIO']
    nvispio = value[0]

    desc = uvf.Desc.Dict
    # Number of random parameters
    nrparm = desc['nrparm']
    # Length of visibility buffer record
    lrec = desc['lrec']

    # Random parameter indices
    ilocu = desc['ilocu']     # U
    ilocv = desc['ilocv']     # V
    ilocw = desc['ilocw']     # W
    iloct = desc['iloct']     # time
    ilocb = desc['ilocb']     # baseline id
    ilocsu = desc['ilocsu']   # source id

    # NX table rows
    nx_rows = []

    # Iterate through kat adapter UV scans, writing their data to disk
    for si, state, aips_source, data_gen in time_chunked_scans(kat_adapter):
        source_name = aips_source["SOURCE"][0].strip()
        source_id = aips_source["ID. NO."][0]

        # Starting visibility of this scan
        start_vis = firstVis

        scan_start = kat_adapter.uv_timestamps[0]
        scan_end = kat_adapter.uv_timestamps[-1]

        # Start and end of the scan
        start = OTObit.day2dhms(scan_start)
        end = OTObit.day2dhms(scan_end)

        # Size of the scan
        nbytes = fmt_bytes(kat_adapter.size)

        log.info("'%s - %s' 'scan % 4d' writing '%s' of source '%s'",
                 start, end, si, nbytes, source_name)

        # Retrieve observational data from the data generator
        for u, v, w, time, baselines, vis in data_gen:
            vis_buffer = np.frombuffer(uvf.VisBuf, count=-1, dtype=np.float32)

            ntime, nbl = u.shape

            for t in range(ntime):
                for bl in range(nbl):
                    # Index within vis_buffer
                    idx = numVisBuff * lrec

                    # Write random parameters
                    vis_buffer[idx + ilocu] = u[t, bl]          # U
                    vis_buffer[idx + ilocv] = v[t, bl]          # V
                    vis_buffer[idx + ilocw] = w[t, bl]          # W
                    vis_buffer[idx + iloct] = time[t]           # time
                    vis_buffer[idx + ilocb] = baselines[bl]     # baseline id
                    vis_buffer[idx + ilocsu] = source_id        # source id

                    # Flatten visibilities for buffer write
                    flat_vis = vis[t, bl].ravel()
                    vis_idx = idx + nrparm
                    vis_buffer[vis_idx:vis_idx + flat_vis.size] = flat_vis

                    numVisBuff += 1

                    # Hit the limit, write
                    if numVisBuff == nvispio:
                        firstVis, numVisBuff = _write_buffer(
                            uvf, firstVis, numVisBuff, lrec)

        # Write out any remaining visibilities
        if numVisBuff > 0:
            firstVis, numVisBuff = _write_buffer(uvf, firstVis, numVisBuff, lrec)

        # Create an index for this scan
        nx_rows.append({
            'TIME': [(scan_start + scan_end) / 2],  # Time Centroid
            'TIME INTERVAL': [scan_end - scan_start],
            'SOURCE ID': [source_id],
            # Should match 'AIPS AN' table version
            # Each AN table defines a subarray
            'SUBARRAY': [1],
            'FREQ ID': [1],            # Should match 'AIPS FQ' row FRQSEL
            'START VIS': [start_vis],  # FORTRAN indexing
            'END VIS': [firstVis - 1]  # FORTRAN indexing
        })

    # Create the index and calibration tables
    uvf.attach_table("AIPS NX", 1)
    uvf.tables["AIPS NX"].rows = nx_rows
    uvf.tables["AIPS NX"].write()
    uvf.attach_CL_from_NX_table(kat_adapter.max_antenna_number)
