import logging
from functools import partial

import TableList
import UV
import UVDesc

from katsdpcontim import obit_err, handle_obit_err

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
        raise NotImplementedError("FITS UV creation via newPFUV "
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
        self._uv = uv

        # Construct a name for this object
        if uv.FileType == "AIPS":
            self._name = name = _aips_filename(uv.Aname, uv.Aclass, uv.Aseq)
        elif uv.FileType == "FITS":
            self._name = name = uv.FileName
        else:
            raise ValueError("Invalid uv.FileType '{}'".format(uv.FileType))

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def close(self):
        """ Closes the wrapped UV file """
        err = obit_err()
        self._uv.Close(err)
        handle_obit_err("Error closing uv file", err)

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

    def create_antenna_table(self, header, rows):
        """
        Creates an AN table associated with this UV file.

        Parameters
        ----------
        header: dict
            Dictionary containing updates for the antenna table
            header. Should contain:

            .. code-block:: python

                { 'RefDate' : ...,
                  'Freq': ..., }

        rows: list
            List of dictionaries describing each antenna, with
            the following form:

            .. code-block:: python

                { 'NOSTA': [1],
                  'ANNAME': ['m003'],
                  'STABXYZ': [100.0, 200.0, 300.0],
                  'DIAMETER': [13.4],
                  'POLAA': [90.0] }
        """
        err = obit_err()

        ant_table = self._uv.NewTable(Table.READWRITE, "AIPS AN", 1, err)
        handle_obit_err("Error creating '%s' AN table." % self._name, err)
        ant_table.Open(Table.READWRITE, err)
        handle_obit_err("Error opening '%s' AN table." % self._name, err)

        # Update header, forcing underlying table update
        ant_table.keys.update(header)
        Table.PDirty(ant_table)

        # Write each row to the antenna table
        for ri, row in enumerate(rows, 1):
            ant_table.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in '%s' AN table. "
                            "Row data is '%s'" % (ri, self._name, row), err)

        # Close table
        ant_table.Close(err)
        handle_obit_err("Error closing '%s' AN table." % self._name, err)

    def create_frequency_table(self, header, rows):
        """
        Creates an FQ table associated this UV file.

        Parameters
        ----------
        header: dict
            Dictionary containing updates for the antenna table
            header. Should contain number of spectral windows (1):

            .. code-block:: python

                { 'numIF' : 1 }

        rows: list
            List of dictionaries describing each spectral window, with
            the following form:

            .. code-block:: python

                {'CH WIDTH': [208984.375],
                  'FRQSEL': [1],
                  'IF FREQ': [-428000000.0],
                  'RXCODE': ['L'],
                  'SIDEBAND': [1],
                  'TOTAL BANDWIDTH': [856000000.0] }
        """
        err = obit_err()

        # If an old table exists, delete it
        if self._uv.GetHighVer("AIPS FQ") > 0:
            self._uv.ZapTable("AIPS FQ", 1, err)
            handle_obit_err("Error zapping old FQ table", err)

        # Get the number of spectral windows from the header
        noif = header['numIF']

        if not noif == 1:
            raise ValueError("Only handling 1 IF at present. "
                             "'%s' specified in header" % noif)

        if not len(rows) == 1:
            raise ValueError("Only handling 1 IF at present. "
                             "'%s' rows supplied" % len(rows))


        # Create and open a new FQ table
        fqtab = self._uv.NewTable(Table.READWRITE, "AIPS FQ",1, err, numIF=noif)
        handle_obit_err("Error creating '%s' FQ table." % self._name, err)
        fqtab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening '%s' FQ table." % self._name, err)

        # Update header, forcing underlying table update
        fqtab.keys.update(header)
        Table.PDirty(fqtab)

        # Write spectral window rows
        for ri, row in enumerate(rows, 1):
            fqtab.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in '%s' FQ table. "
                            "Row data is '%s'" % (ri, self._name, row), err)

        # Close
        fqtab.Close(err)
        handle_obit_err("Error closing '%s' FQ table." % self._name, err)

    def create_index_table(self, header, rows):
        """
        Creates an NX table associated with this UV file.
        """

        err = obit_err()

        # Create and open the SU table
        nxtab = self._uv.NewTable(Table.READWRITE, "AIPS NX", 1, err)
        handle_obit_err("Error creating '%s' NX table." % self._name, err)
        nxtab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening '%s' NX table." % self._name, err)

        # Update header, forcing underlying table update
        nxtab.keys.update(header)
        Table.PDirty(nxtab)

        # Write index table rows
        for ri, row in enumerate(rows, 1):
            nxtab.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in '%s' NX table. "
                            "Row data is '%s'" % (ri, self._name, row), err)


        nxtab.Close(err)
        handle_obit_err("Error closing NX table", err)

    def create_source_table(self, header, rows):
        """
        Creates an SU table associated with this UV file.

        Parameters
        ----------
        header: dict
            Dictionary containing updates for the source table
            header. Should contain:

            .. code-block:: python

                { 'numIF' : 1,
                  'FreqID': 1, }

        rows: list
            List of dictionaries describing each antenna, with
            the following form:

            .. code-block:: python

                {'BANDWIDTH': 855791015.625,
                  'DECAPP': [-37.17505555555555],
                  'DECEPO': [-37.23916666666667],
                  'DECOBS': [-37.17505555555555],
                  'EPOCH': [2000.0],
                  'ID. NO.': 1,
                  'RAAPP': [50.81529166666667],
                  'RAEPO': [50.65166666666667],
                  'RAOBS': [50.81529166666667],
                  'SOURCE': 'For A           '},
        """

        err = obit_err()

        # Create and open the SU table
        sutab = self._uv.NewTable(Table.READWRITE, "AIPS SU",1,err)
        handle_obit_err("Error creating '%s' SU table." % self._name, err)
        sutab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening '%s' SU table." % self._name, err)

        # Update header, forcing underlying table update
        sutab.keys.update(header)
        Table.PDirty(sutab)

        # Write rows
        for ri, row in enumerate(rows, 1):
            sutab.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in '%s' SU table. "
                            "Row data is '%s'" % (ri, self._name, row), err)

        # Close the table
        sutab.Close(err)
        handle_obit_err("Error closing '%s' SU table." % self._name, err)

    def create_calibration_table(self, header, rows):
        """
        Creates a CL table associated with this UV file.
        """
        err = obit_err()

        cltab = self._uv.NewTable(Table.READWRITE, "AIPS CL", 1, err)
        handle_obit_err("Error creating '%s' CL table." % self._name, err)
        cltab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening '%s' CL table." % self._name, err)

        # Update header, forcing underlying table update
        cltab.keys.update(header)
        Table.PDirty(cltab)

        # Write calibration table rows
        for ri, row in enumerate(rows, 1):
            cltab.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in '%s' CL table. "
                            "Row data is '%s'" % (ri, self._name, row), err)

        cltab.Close(err)
        handle_obit_err("Error closing '%s' CL table" % self._name, err)

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