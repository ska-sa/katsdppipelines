import logging
from textwrap import TextWrapper
import sys

import six
import numpy as np

import TableList
import UV

from katsdpcontim import (AIPSTable,
                          AIPSHistory,
                          AIPSPath,
                          obit_err,
                          handle_obit_err)

log = logging.getLogger('katsdpcontim')

""" TextWrapper for wrapping AIPS history text to 70 chars """
_history_wrapper = TextWrapper(width=70, initial_indent='',
                                subsequent_indent='  ',
                                break_long_words=True,
                                break_on_hyphens=True)

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


def open_uv(aips_path, nvispio=1024, mode=None):
    """
    Opens an AIPS/FITS UV file and returns a wrapped :class:`UVFacade` object.

    Parameters
    ----------
    aips_path: :class:`AIPSPath`
        Obit file object.
    nvispio: integer
        Number of visibilities to read/write per I/O operation
    mode(optional): str
        "r" to read, "w" to write, "rw" to read and write.
        Defaults to "r"

    Returns
    -------
    :class:`UV`
        An Obit UV object
    """

    err = obit_err()

    if mode is None:
        mode = "r"

    uv_mode = uv_file_mode(mode)
    exists = False  # Test if the file exists

    if aips_path.dtype == "AIPS":
        try:
            uv = UV.newPAUV(aips_path.label, aips_path.name,
                            aips_path.aclass, aips_path.disk,
                            aips_path.seq, exists, err, nvis=nvispio)
        except Exception:
            raise (ValueError("Error calling newPAUV on '%s'" % aips_path),
                                                None, sys.exc_info()[2])
    elif aips_path.dtype == "FITS":
        raise NotImplementedError("newPFUV calls do not currently work")

        try:
            uv = UV.newPFUV(aips_path.label, aips_path.name, aips_path.disk,
                            exists, err, nvis=nvispio)
        except Exception:
            raise (ValueError("Error calling newPFUV on '%s'" % aips_path),
                                                None, sys.exc_info()[2])
    else:
        raise ValueError("Invalid dtype '{}'".format(aips_path.dtype))

    handle_obit_err("Error opening '%s'" % aips_path, err)

    try:
        uv.Open(uv_mode, err)
    except Exception:
        raise (ValueError("Error opening '%s'" % aips_path),
                None, sys.exc_info()[2])

    handle_obit_err("Error opening '%s'" % aips_path, err)

    return uv

def uv_factory(**kwargs):
    """
    Factory for creating a UV observation file.

    Parameters
    ----------
    aips_path : :class:`AIPSPath`
        Obit file class
    mode (optional) : string
        File opening mode passed to :func:`uv_open`.
    nvispio : integer
        Number of visibilities to read/write per I/O operation
    table_cmds (optional) : dict
        A dictionary containing directives for AIPS table creation on the UV file.
        Of the form:

        .. code-block:: python

            {
                "AIPS FQ" : {
                    "attach" : {'version': 1, 'numIF': 1},
                    "keywords" : { ... },
                    "rows" : [{...}, {...}, ..., {...}],
                    "write" : True,
                },
                "AIPS SU" : {...}
            }

        where:

        * ``attach`` instructs ``uv_factory`` to attach this
          table to the UV file with the supplied ``kwargs``.
        * ``keywords`` instructs ``uv_factory`` to set the
          keywords on the table with the supplied dictionary.
        * ``rows`` instructs ``uv_factory`` to set the rows
          of the table with the supplied list of dictionaries.
        * ``write`` instructs ``uv_factory`` to write the
          rows to the table after all of the above.

    desc (optional) : dict
        A dictionary UV descriptor suitable for conditioning the
        UV file.

    Returns
    -------
    :class:`UVFacade`
        Object representing the UV observation.
    """
    try:
        ofile = kwargs.pop('aips_path')
    except KeyError:
        raise ValueError("No 'aips_path' argument supplied.")

    mode = kwargs.pop('mode', 'r')
    nvispio = kwargs.pop('nvispio', 1024)

    uv = open_uv(ofile, nvispio=nvispio, mode=mode)
    uvf = UVFacade(uv)

    table_cmds = kwargs.pop('table_cmds', {})

    reopen = False

    # Handle table creation commands
    for table, cmds in six.iteritems(table_cmds):
        # Perform any attach commands first
        try:
            attach_kwargs = cmds["attach"]
        except KeyError:
            pass
        else:
            uvf.attach_table(table, **attach_kwargs)

        # Update any table keywords
        try:
            keywords = cmds["keywords"]
        except KeyError:
            pass
        else:
            uvf.tables[table].keywords.update(keywords)

        # Update with any rows
        try:
            rows = cmds["rows"]
        except KeyError:
            pass
        else:
            uvf.tables[table].rows = rows

        # Write rows now if requested
        if cmds.get("write", False):
            uvf.tables[table].write()

    # Needs to happen after table creation
    # so that uv.TableList is updated
    desc = kwargs.pop('desc', None)

    if desc:
        uvf.update_descriptor(desc)
        reopen = True

    # Reopen the file to trigger descriptor
    # and header updates
    if reopen:
        uvf.Open(uv_file_mode(mode))

    return uvf


class UVFacade(object):
    """
    Provides a simplified interface to an Obit :class:`UV.UV` object.

    ::

        But you've got to look past the hair and the
        cute, cuddly thing - it's all a deceptive facade

        https://www.youtube.com/watch?v=73d6h_go7QI

    Note well the :meth:`UVFacade._clear_uv()` method.
    """

    def __init__(self, uv, **kwargs):
        """
        Constructor

        Parameters
        ----------
        uv: :class:`UV` or :class:`AIPSPath`
            Either a Obit UV object or an AIPSPath describing it
        mode: string
            Mode string passed through to :function:`open_uv`
            if `uv` is supplied with a :class:`AIPSPath`
        """
        self._err = err = obit_err()
        self._open_logic(uv, err, **kwargs)

    def _open_logic(self, uv, err, **kwargs):
        """
        Peforms logic for opening a UV file

        * Opening the UV File if given an AIPS path.
        * Setting up the AIPS Path if given a UV file.
        * Open any Tables attached to the UV file.

        Parameters
        ----------
        uv: :class:`UV` or :class:`AIPSPath`
            Either a Obit UV object or an AIPSPath describing it
        """

        # Given an AIPSPath. open it.
        if isinstance(uv, AIPSPath):
            self._aips_path = uv
            mode = kwargs.pop('mode', 'r')
            self._uv = uv = open_uv(uv, mode=mode)
        # Given an Obit UV file.
        # Construct an AIPSPath
        elif isinstance(uv, UV.UV):
            # FITS and AIPS files have different properties
            if uv.FileType == "FITS":
                name = uv.FileName
                aclass = None
                seq = None
            elif uv.FileType == "AIPS":
                name = uv.Aname
                aclass = uv.Aclass
                seq = uv.Aseq
            else:
                raise ValueError("Invalid FileType '%s'" % uv.FileType)

            self._aips_path = AIPSPath(name, uv.Disk, aclass,
                                       seq, dtype=uv.FileType)

            self._uv = uv
        else:
            raise TypeError("Invalid type '%s'. "
                            "Must be Obit UV object "
                            "or an AIPSPath." % type(uv))

        # Open tables attached to this UV file.
        tables = TableList.PGetList(uv.TableList, err)
        handle_obit_err("Error getting '%s' table list" % self.name)

        # History tables don't work like the other tables
        ignored_tables = ["AIPS HI"]

        self._tables = {name: AIPSTable(uv, name, version, 'r', err)
                        for version, name in tables
                        if name not in ignored_tables}

        self._tables["AIPS HI"] = AIPSHistory(uv, err)

    def close(self):
        """ Closes the wrapped UV file """

        # Close all attached tables
        for table in self._tables.values():
            table.close()

        self._tables = {}

        try:
            self._uv.Close(self._err)
        except AttributeError:
            # Closed
            return
        except Exception:
            raise (Exception("Exception closing uv file '%s'" % self.name),
                                                    None, sys.exc_info()[2])

        handle_obit_err("Error closing uv file '%s'" % self.name, self._err)
        self._clear_uv()

    @property
    def uv(self):
        """
        The Obit :class:`UV.UV` object encapsulated in the :class:`UVFacade`.

        Returns
        -------
        :class:`UV.UV`
        """
        try:
            return self._uv
        except AttributeError:
            self._open_logic(self._aips_path, self._err)

        return self._uv

    def _clear_uv(self):
        """
        Calls :code:`del` on the wrapped UV object and sets it to None.
        Without this, internal resources on the UV object
        are not released and subsequent calls to *unrelated*
        objects will fail, claiming that the UV object
        is *not* a UV object.
        """
        try:
            del self._uv
        except AttributeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()


    def append_history(self, msg):
        """
        Appends ``msg`` to the HISTORY table of this UV file.

        Long lines will be broken at 70 chars.
        """
        for line in _history_wrapper.wrap(msg):
            self._tables["AIPS HI"].append(line)

    @property
    def tables(self):
        return self._tables

    def attach_table(self, name, version, **kwargs):
        self._tables[name] = AIPSTable(self.uv, name, version, 'r',
                                       self._err, **kwargs)

    @property
    def aips_path(self):
        return self._aips_path

    @property
    def name(self):
        return str(self._aips_path)

    @property
    def Desc(self):
        return self.uv.Desc

    @property
    def List(self):
        return self.uv.List

    @property
    def VisBuf(self):
        return self.uv.VisBuf

    @property
    def np_visbuf(self):
        return np.frombuffer(self.uv.VisBuf, count=-1, dtype=np.float32)

    def Open(self, mode):
        err_msg = "Error opening UV file '%s'" % self.name

        try:
            self.uv.Open(mode, self._err)
        except Exception:
            raise (Exception(err_msg, None, sys.exc_info()[2]))

        handle_obit_err(err_msg, self._err)

    def Close(self):
        return self.close()

    def Read(self, firstVis=None):
        err_msg = "Error reading UV file '%s'" % self.name

        try:
            self.uv.Read(self._err, firstVis=firstVis)
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])

        handle_obit_err(err_msg, self._err)

    def Write(self, firstVis=None):
        err_msg = "Error writing UV file '%s'" % self.name

        try:
            self.uv.Write(self._err, firstVis=firstVis)
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])

        handle_obit_err(err_msg, self._err)

    def Zap(self):
        err_msg = "Error zapping UV file '%s'" % self.name

        try:
            self.uv.Zap(self._err)
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])

        handle_obit_err(err_msg, self._err)
        self._clear_uv()

    def update_descriptor(self, descriptor):
        """
        Update the UV descriptor.

        Parameters
        ----------
        descriptor: dict
            Dictionary containing updates applicable to
            :code:`uv.Desc.Dict`.
        """
        uv = self.uv

        desc = uv.Desc.Dict
        desc.update(descriptor)
        uv.Desc.Dict = desc

        err_msg = "Error updating descriptor on UV file '%s'" % self.name

        try:
            uv.UpdateDesc(self._err)
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])

        handle_obit_err(err_msg, self._err)

    def attach_CL_from_NX_table(self, max_ant_nr):
        """
        Creates a CL table associated with this UV file
        from an NX table.

        Parameters
        ----------
        max_ant_nr : integer
            Maximum antenna number written to the AIPS AN table.
        """
        err_msg = ("Error creating CL table "
                    "from NX table on UV file '%s'" % self.name)

        try:
            UV.PTableCLfromNX(self.uv, max_ant_nr, self._err)
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])

        handle_obit_err(err_msg, self._err)
