import UVDesc
import Table

from obit_context import obit_err, handle_obit_err

def _sanity_check_header(header):
    """ Sanity check AN and SU header dictionaries """
    for k in ('RefDate', 'Freq'):
        if k not in header:
            raise KeyError("'%s' not in header." % k)

class UVFacade(object):
    """
    Provides a simplified interface to an Obit UV object
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

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def close(self):
        """ Closes the wrapped UV file """
        self._uv.Close(obit_err())

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
        handle_obit_err("Error updating UV Descriptor", err)

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

        _sanity_check_header(header)

        err = obit_err()

        ant_table = self._uv.NewTable(Table.READWRITE, "AIPS AN", 1, err)
        handle_obit_err("Error creating AN table.", err)
        ant_table.Open(Table.READWRITE, err)
        handle_obit_err("Error opening AN table.", err)

        # Update header
        ant_table.keys.update(header)
        JD = UVDesc.PDate2JD(header['RefDate'])
        ant_table.keys['GSTIA0'] = UVDesc.GST0(JD)*15.0
        ant_table.keys['DEGPDY'] = UVDesc.ERate(JD)*360.0

        # Mark table as dirty to force header update
        Table.PDirty(ant_table)

        # Write each row to the antenna table
        for ri, row in enumerate(rows, 1):
            ant_table.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in AN table. "
                            "Row data is '%s'" % (ri, row), err)

        # Close table
        ant_table.Close(err)
        handle_obit_err("Error closing AN table.", err)

    def create_frequency_table(self, header, rows):
        """
        Creates an FQ table associated this UV file.

        Parameters
        ----------
        header: dict
            Dictionary containing updates for the antenna table
            header. Should contain number of spectral windows (1):

            .. code-block:: python

                { 'NO_IF' : 1 }

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
        noif = header['NO_IF']

        if not noif == 1:
            raise ValueError("Only handling 1 IF at present. "
                             "'%s' specified in header" % noif)

        if not len(rows) == 1:
            raise ValueError("Only handling 1 IF at present. "
                             "'%s' rows supplied" % len(rows))


        # Create and open a new FQ table
        fqtab = self._uv.NewTable(Table.READWRITE, "AIPS FQ",1, err, numIF=noif)
        handle_obit_err("Error creating FQ table,", err)
        fqtab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening FQ table,", err)

        # Update header
        fqtab.keys.update(header)
        # Force update
        Table.PDirty(fqtab)

        # Write spectral window rows
        for ri, row in enumerate(rows, 1):
            fqtab.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in FQ table. "
                            "Row data is '%s'" % (ri, row), err)

        # Close
        fqtab.Close(err)
        handle_obit_err("Error closing FQ table.")

    def create_index_table(self, header, rows):
        """
        Creates an NX table in this UV file.
        """

        err = obit_err()

        # Create and open the SU table
        nxtab = self._uv.NewTable(Table.READWRITE, "AIPS NX",1,err)
        handle_obit_err("Error creating NX table", err)
        nxtab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening NX table", err)

        # Write index table rows
        for ri, row in enumerate(rows, 1):
            nxtab.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in NX table. "
                            "Row data is '%s'" % (ri, row), err)


        nxtab.Close(err)
        handle_obit_err("Error closing NX table", err)


    def create_source_table(self, header, rows):
        """
        Creates an SU table in this UV file.

        Parameters
        ----------
        header: dict
            Dictionary containing updates for the source table
            header. Should contain:

            .. code-block:: python

                { 'RefDate' : ...,
                  'Freq': ..., }

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

        _sanity_check_header(header)

        err = obit_err()

        # Create and open the SU table
        sutab = self._uv.NewTable(Table.READWRITE, "AIPS SU",1,err)
        handle_obit_err("Error creating SU table", err)
        sutab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening SU table", err)

        # Update header, forcing underlying table update
        sutab.keys.update(header)
        Table.PDirty(sutab)

        # Write rows
        for ri, row in enumerate(rows, 1):
            sutab.WriteRow(ri, row, err)
            handle_obit_err("Error writing SU table", err)

        # Close the table
        sutab.Close(err)
        handle_obit_err("Error closing SU table", err)
