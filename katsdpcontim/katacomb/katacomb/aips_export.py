from __future__ import with_statement

import logging

from katacomb import (uv_factory,
                    img_factory,
                    katdal_timestamps,
                    katdal_ant_name)

import numpy as np

log = logging.getLogger('katacomb')

# AIPS Table entries follow this kind of schema where data
# is stored in singleton lists, while book-keeping entries
# are not.
# { 'DELTAX' : [1.4], 'NumFields' : 4, 'Table name' : 'AIPS CC' }
# Strip out book-keeping keys and flatten lists
_DROP = { "Table name", "NumFields", "_status" }
def _condition(row):
    """ Flatten singleton lists and drop book-keeping keys """
    return { k: v[0] for k, v in row.items() if k not in _DROP }

def export_calibration_solutions(uv_files, kat_adapter, telstate):
    """
    Exports calibration solutions from each file
    in ``uv_files`` into ``telstate``.

    Parameters
    ----------
    uv_files : list
        List of :class:`katacomb.UVFacade` objects
    kat_adapter : :class:`KatdalAdapter`
        Katdal Adapter
    telstate : :class:`katsdptelstate.Telescope`
        telstate object
    """


    # MFImage outputs a UV file per source.  Iterate through each source:
    # (1) Extract complex gains from attached "AIPS SN" table
    # (2) Write them to telstate
    for si, uv_file in enumerate(uv_files):
        try:
            with uv_factory(aips_path=uv_file, mode='r') as uvf:
                try:
                    sntab = uvf.tables["AIPS SN"]
                except KeyError:
                    log.warn("No calibration solutions in '%s'", uv_file)
                else:
                    # Handle cases for single/dual pol gains
                    if "REAL2" in sntab.rows[0]:
                        def _extract_gains(row):
                            return np.array([row["REAL1"] + 1j*row["IMAG1"],
                                             row["REAL2"] + 1j*row["IMAG2"]],
                                                dtype=np.complex64)
                    else:
                        def _extract_gains(row):
                            return np.array([row["REAL1"] + 1j*row["IMAG1"],
                                             row["REAL1"] + 1j*row["IMAG1"]],
                                                dtype=np.complex64)

                    # Write each complex gain out per antenna
                    for row in (_condition(r) for r in sntab.rows):
                        # Convert time back from AIPS to katdal UTC
                        time = katdal_timestamps(row["TIME"], kat_adapter.midnight)
                        # Convert from AIPS FORTRAN indexing to katdal C indexing
                        ant = "%s_gains" % katdal_ant_name(row["ANTENNA NO."])

                        # Store complex gain for this antenna
                        # in telstate at this timestamp
                        telstate.add(ant, _extract_gains(row), ts=time)
        except Exception as e:
            log.warn("Export of calibration solutions from '%s' failed.\n%s",
                        uv_file, str(e))



def export_clean_components(clean_files, target_indices, kat_adapter, telstate):
    """
    Exports clean components from each file in ``clean_files`` into ``telstate``.

    Parameters
    ----------
    clean_files : list
        List of :class:`katacomb.ImageFacade` objects
    target_indices : list of integers
        List of target indices associated with each CLEAN file
    kat_adapter : :class:`KatdalAdapter`
        Katdal Adapter
    telstate : :class:`katsdptelstate.Telescope`
        telstate object
    """

    # MFImage outputs a CLEAN image per source.  Iterate through each source:
    # (1) Extract clean components from attached "AIPS CC" table
    # (2) Write them to telstate

    targets = kat_adapter.katdal.catalogue.targets

    it = enumerate(zip(clean_files, target_indices))
    for si, (clean_file, ti) in it:
        try:
            with img_factory(aips_path=clean_file, mode='r') as cf:
                try:
                    merged_cctab = cf.MergeCC()
                except KeyError:
                    log.warn("No clean components in '%s'", clean_file)
                else:
                    target = "target%d" % si

                    # Condition all rows up front
                    rows = [_condition(r) for r in merged_cctab.rows]

                    # Extract description
                    description = targets[ti].description
                    data = { 'description': description, 'components': rows }

                    # Store them in telstate
                    key = telstate.SEPARATOR.join((target, "clean_components"))
                    telstate.add(key, data, immutable=True)

        except Exception as e:
            log.warn("Export of clean components from '%s' failed.\n%s",
                        clean_file, str(e))
