import logging
from functools import partial

import TableList
import UV
import UVDesc

from katsdpcontim import (AIPSTable,
                        ObitFile,
                        obit_err,
                        handle_obit_err)

log = logging.getLogger('katsdpcontim')


def uv_file_mode(mode):
    """ Returns UV file mode given string mode """
    read = 'r' in mode
    write = 'w' in mode
    readcal = 'c' in mode

    if readcal:
        return UV.READCAL
    elif read and write:
        return UV.READWRITE
    elif write:
        return UV.WRITEONLY
    # Read by default
    else:
        return UV.READONLY

def _open_aips_uv(obit_file, nvispio=1024, mode=None):
    """ Open/create the specified AIPS UV file """
    err = obit_err()

    label = "katuv" # Possibly abstract this too
    exists = False  # Test if the file exists
    uv = UV.newPAUV(label, obit_file.name, obit_file.aclass,
                        obit_file.disk, obit_file.seq,
                        exists, err, nvis=nvispio)
    handle_obit_err("Error opening uv file", err)
    return uv

def open_uv(obit_file, nvispio=1024, mode=None):
    """
    Opens an AIPS/FITS UV file and returns a wrapped :class:`UVFacade` object.

    Parameters
    ----------
    obit_file: :class:`ObitFile`
        Obit file object.
    nvispio: integer
        Number of visibilities to read/write per I/O operation
    mode(optional): str
        "r" to read, "w" to write, "rw" to read and write.
        Defaults to "r"

    Returns
    -------
    :class:`UVFacade`
        A UVFacade object
    """

    err = obit_err()

    if mode is None:
        mode = "r"

    uv_mode = uv_file_mode(mode)

    if obit_file.dtype == "AIPS":
        method = partial(_open_aips_uv, obit_file, mode=mode, nvispio=nvispio)
    elif obit_file.dtype == "FITS":
        raise NotImplementedError("FITS UV open via newPFUV "
                                  "not yet supported.")
    else:
        raise ValueError("Invalid dtype '{}'".format(obit_file.dtype))

    uv = method()
    uv.Open(uv_mode, err)
    handle_obit_err("Error opening uv file", err)

    return UVFacade(uv)

def uv_factory(**kwargs):
    """
    Factory for creating a UV observation file.
    If a `katdata` parameter is passed, this will be used
    to condition the UV file.

    Parameters
    ----------
    obit_file: :class:`ObitFile`
        Obit file class
    mode (optional): string
        File opening mode passed to :func:`uv_open`.
    nvispio: integer
        Number of visibilities to read/write per I/O operation
    katdata (optional): :class:`katdal.DataSet`
        A katdal data set. If present, this data will
        be used to condition the UV file and create
        tables.

    Returns
    -------
    :class:`UVFacade`
        Object representing the UV observation.
    """
    try:
        ofile = kwargs.pop('obit_file')
    except KeyError as e:
        raise ValueError("No 'obit_file' argument supplied.")

    mode = kwargs.pop('mode', 'r')
    nvispio = kwargs.pop('nvispio', None)

    uvf = open_uv(ofile, nvispio=nvispio, mode=mode)

    modified = False

    # If we have a katdal adapter,
    # create subtables and update the descriptor
    KA = kwargs.pop('katdata', None)

    if KA is not None:
        # Attach tables
        uvf.attach_table("AIPS AN", 1)
        uvf.attach_table("AIPS FQ", 1, numIF=KA.nif)
        uvf.attach_table("AIPS SU", 1)

        # Update their keywords
        uvf.tables["AIPS AN"].keywords.update(KA.uv_antenna_keywords)
        uvf.tables["AIPS FQ"].keywords.update(KA.uv_spw_keywords)
        uvf.tables["AIPS SU"].keywords.update(KA.uv_source_keywords)

        # Set their rows
        uvf.tables["AIPS AN"].rows = KA.uv_antenna_rows
        uvf.tables["AIPS FQ"].rows = KA.uv_spw_rows
        uvf.tables["AIPS SU"].rows = KA.uv_source_rows

        # Write them
        uvf.tables["AIPS AN"].write()
        uvf.tables["AIPS FQ"].write()
        uvf.tables["AIPS SU"].write()

        # Close them
        uvf.tables["AIPS AN"].close()
        uvf.tables["AIPS FQ"].close()
        uvf.tables["AIPS SU"].close()

        # Needs to happen after subtables
        # so that uv.TableList is updated
        uvf.update_descriptor(KA.uv_descriptor())
        modified = True

    # If modified, reopen the file to trigger descriptor
    # and header updates
    if modified:
        uvf.Open(uv_file_mode(mode))

    return uvf

class UVFacade(object):
    """
    Provides a simplified interface to an Obit UV object.

    ::

        But you've got to look past the hair and the
        cute, cuddly thing - it's all a deceptive facade

        https://www.youtube.com/watch?v=DWkMgJ2UknQ
    """
    def __init__(self, uv, **kwargs):
        """
        Constructor

        Parameters
        ----------
        uv: :class:`UV` or :class:`ObitFile`
            Either a Obit UV object or an ObitFile describing it
        mode: string
            Mode string passed through to :function:`open_uv`
            if `uv` is supplied with a :class:`ObitFile`
        """
        err = obit_err()

        # Given an ObitFile. open it.
        if isinstance(uv, ObitFile):
            self._obit_file = uv
            mode = kwargs.pop(mode, 'r')
            self._uv = open_uv(uv, mode=mode)
        # Given an Obit UV file.
        # Construct an ObitFile
        elif isinstance(uv, UV.UV):
            name = uv.Aname if uv.FileType == "AIPS" else uv.FileName
            self._obit_file = ObitFile(name, uv.Disk, uv.Aclass,
                                       uv.Aseq, dtype=uv.FileType)

            self._uv = uv
        else:
            raise TypeError("Invalid type '%s'. "
                            "Must be Obit UV object "
                            "or an ObitFile." % type(uv))


        # Open tables attached to this UV file.
        tables = TableList.PGetList(uv.TableList, err)
        handle_obit_err("Error getting '%s' table list" % self.name)

        self._tables = { name: AIPSTable(uv, name, version, 'r', err)
                                      for version, name in tables }

    def close(self):
        """ Closes the wrapped UV file """

        # Close all attached tables
        for table in self._tables.values():
            table.close()

        err = obit_err()
        self._uv.Close(err)
        handle_obit_err("Error closing uv file", err)

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    @property
    def tables(self):
        return self._tables

    def attach_table(self, name, version, **kwargs):
        self._tables[name] = AIPSTable(self._uv, name, version, 'r',
                                               obit_err(), **kwargs)

    @property
    def obit_file(self):
        return self._obit_file

    @property
    def name(self):
        return str(self._obit_file)

    @property
    def Desc(self):
        return self._uv.Desc

    @property
    def List(self):
        return self._uv.List

    @property
    def VisBuf(self):
        return self._uv.VisBuf

    def Open(self, mode):
        err = obit_err()
        self._uv.Open(mode, err)
        handle_obit_err("Error opening UV file '%s'" % self.name, err)

    def Close(self):
        return self.close()

    def Read(self, firstVis=None):
        err = obit_err()
        self._uv.Read(err, firstVis=firstVis)
        handle_obit_err("Error reading UV file '%s'" % self.name)

    def Write(self, firstVis=None):
        err = obit_err()
        self._uv.Write(err, firstVis=firstVis)
        handle_obit_err("Error writing UV file '%s'" % self.name)

    def update_descriptor(self, descriptor):
        """
        Update the UV descriptor.

        Parameters
        ----------
        descriptor: dict
            Dictionary containing updates applicable to
            :code:`uv.Desc.Dict`.
        """
        err = obit_err()

        desc = self._uv.Desc.Dict
        desc.update(descriptor)
        self._uv.Desc.Dict = desc
        self._uv.UpdateDesc(err)
        handle_obit_err("Error updating UV Descriptor on '{}'"
                            .format(self.name), err)

    def attach_CL_from_NX_table(self, max_ant_nr):
        """
        Creates a CL table associated with this UV file
        from an NX table.

        Parameters
        ----------
        max_ant_nr : integer
            Maximum antenna number written to the AIPS AN table.
        """
        err = obit_err()
        UV.PTableCLfromNX(self._uv, max_ant_nr, err)
        handle_obit_err("Error creating '%s' CL table from NX table"
                                                    % self.name, err)