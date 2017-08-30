import logging
from functools import partial

import TableList
import UV
import UVDesc

from katsdpcontim import (AIPSTable,
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

def _open_aips_uv(name, disk, aclass=None, seq=None, mode=None):
    """ Open/create the specified AIPS UV file """
    err = obit_err()

    if aclass is None:
        aclass = "raw"

    if seq is None:
        seq = 1

    label = "katuv" # Possibly abstract this too
    exists = False  # Test if the file exists
    uv = UV.newPAUV(label, name, aclass, disk, seq, exists, err)
    handle_obit_err("Error opening uv file", err)
    return uv

def _aips_filename(name, aclass, seq):
    """
    Parameters
    ----------
    name: string
        AIPS file name
    aclass: string
        AIPS file class
    seq: integer
        AIPS file sequence number

    Returns
    -------
    string
        String describing the AIPS filename
    """
    return "{}.{}.{}".format(name, aclass, seq)

def open_uv(name, disk, aclass=None, seq=None, dtype=None, mode=None):
    """
    Opens an AIPS/FITS UV file and returns a wrapped :class:`UVFacade` object.

    Parameters
    ----------
    name: str
        Name of the file.
    disk: integer
        The AIPS or FITS disk on which the file is located.
    aclass (optional): str
        The class of the AIPS file. Only applies to AIPS types.
        Defaults to "raw"
    seq (optional): integer
        The sequence of the AIPS file. Only applies to AIPS types.
        Defaults to 1.
    dtype (optional): str
        Data type, or type of file system to write to.
        Should be either "AIPS" or "FITS".
        Defaults to "AIPS".
    mode: str
        "r" to read, "w" to write, "rw" to read and write.

    Returns
    -------
    :class:`UVFacade`
        A UVFacade object
    """

    err = obit_err()

    if mode is None:
        mode = "r"

    if dtype is None:
        dtype = "AIPS"

    uv_mode = uv_file_mode(mode)

    if dtype.upper() == "AIPS":
        method = partial(_open_aips_uv, name, disk, aclass, seq, mode)
    elif dtype.upper() == "FITS":
        raise NotImplementedError("FITS UV open via newPFUV "
                                  "not yet supported.")
    else:
        raise ValueError("Invalid dtype '{}'".format(dtype))

    uv = method()
    uv.Open(uv_mode, err)
    handle_obit_err("Error opening uv file", err)

    return UVFacade(uv)

class UVFacade(object):
    """
    Provides a simplified interface to an Obit UV object.

    ::

        But you've got to look past the hair and the
        cute, cuddly thing - it's all a deceptive facade

        https://www.youtube.com/watch?v=DWkMgJ2UknQ
    """
    def __init__(self, uv):
        """
        Constructor

        Parameters
        ----------
        uv: UV
            An Obit UV object
        """
        err = obit_err()

        self._uv = uv

        # Construct a name for this object
        if uv.FileType == "AIPS":
            self._name = name = _aips_filename(uv.Aname, uv.Aclass, uv.Aseq)
        elif uv.FileType == "FITS":
            self._name = name = uv.FileName
        else:
            raise ValueError("Invalid uv.FileType '{}'".format(uv.FileType))

        tables = TableList.PGetList(uv.TableList, err)
        handle_obit_err("Error getting '%s' table list" % name)

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
    def name(self):
        return self._name

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
        handle_obit_err("Error opening UV file '%s'" % self._name, err)

    def Close(self):
        return self.close()

    def Read(self, firstVis=None):
        err = obit_err()
        self._uv.Read(err, firstVis=firstVis)
        handle_obit_err("Error reading UV file '%s'" % self._name)

    def Write(self, firstVis=None):
        err = obit_err()
        self._uv.Write(err, firstVis=firstVis)
        handle_obit_err("Error writing UV file '%s'" % self._name)

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
                            .format(self._name), err)

    def create_calibration_table_from_index(self, max_ant_nr):
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
                                                    % self._name, err)