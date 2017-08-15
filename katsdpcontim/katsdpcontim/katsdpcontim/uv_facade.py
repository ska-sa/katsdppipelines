import UVDesc
import Table

from obit_context import obit_err, handle_obit_err

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
        err = obit_err()

        ant_table = self._uv.NewTable(Table.READWRITE, "AIPS AN", 1, err)
        handle_obit_err("Error creating AN table.", err)
        ant_table.Open(Table.READWRITE, err)
        handle_obit_err("Error opening AN table.", err)

        # Update header, forcing underlying table update
        ant_table.keys.update(header)
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
        handle_obit_err("Error creating FQ table,", err)
        fqtab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening FQ table,", err)

        # Update header, forcing underlying table update
        fqtab.keys.update(header)
        pprint("FQTAB Header")
        pprint(fqtab.keys)
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
        Creates an NX table associated with this UV file.
        """

        err = obit_err()

        # Create and open the SU table
        nxtab = self._uv.NewTable(Table.READWRITE, "AIPS NX", 1, err)
        handle_obit_err("Error creating NX table", err)
        nxtab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening NX table", err)

        # Update header
        nxtab.keys.update(header)
        # Force update
        Table.PDirty(nxtab)

        # Write index table rows
        for ri, row in enumerate(rows, 1):
            nxtab.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in NX table. "
                            "Row data is '%s'" % (ri, row), err)


        nxtab.Close(err)
        handle_obit_err("Error closing NX table", err)

    def create_calibration_table(self, header, rows):
        """
        Creates a CL table associated with this UV file.
        """
        err = obit_err()

        cltab = self._uv.NewTable(Table.READWRITE, "AIPS CL", 1, err)
        handle_obit_err("Error creating CL table", err)
        cltab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening CL table", err)

        # Update header, forcing underlying table update
        cltab.keys.update(header)
        Table.PDirty(cltab)

        # Write calibration table rows
        for ri, row in enumerate(rows, 1):
            cltab.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in CL table. "
                            "Row data is '%s'" % (ri, row), err)

        cltab.Close(err)
        handle_obit_err("Error closing CL table", err)

    def create_source_table(self, header, rows):
        """
        Creates an SU table in this UV file.

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
