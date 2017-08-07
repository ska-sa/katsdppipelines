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

    def __del__(self):
        """ Close on garbage collection """
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

        desc = self._uv.Desc.Dict.copy()
        desc.update(descriptor)
        self._uv.Desc.Dict = desc
        self._uv.UpdateDesc(err)
        handle_obit_err("Error updating UV Descriptor", err)

    def create_antenna_table(self, header, rows):
        """
        Creates an AN table in this UV file.

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

        # Read a row to serve as a row template
        row = ant_table.ReadRow(1, err)
        handle_obit_err("Error reading AN table.", err)

        # Write each row to the antenna table
        for ri, row_update in enumerate(rows, 1):
            row.update(row_update)
            ant_table.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in AN table. "
                            "Row data is '%s'" % (ri, row), err)

        # Close table
        ant_table.Close(err)
        handle_obit_err("Error closing AN table.", err)

    def create_frequency_table(self, header, rows):
        """
        Creates an FQ table in this UV file.

        Parameters
        ----------
        header: dict
            Dictionary containing updates for the antenna table
            header. Should contain number of spectral windows (1):

            .. code-block:: python

                { 'nif' : 1 }

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
        noif = header.pop('nif')

        if not noif == 1:
            raise ValueError("Only handling 1 IF at present. "
                             "'%s' supplied" % noif)

        # Create and open a new FQ table
        fqtab = self._uv.NewTable(Table.READWRITE, "AIPS FQ",1, err, numIF=noif)
        handle_obit_err("Error creating FQ table,", err)
        fqtab.Open(Table.READWRITE, err)
        handle_obit_err("Error opening FQ table,", err)

        # Update header
        fqtab.keys['NO_IF'] = noif  # Structural so no effect
        # Force update
        Table.PDirty(fqtab)

        # Basic row definition
        row = { 'NumFields': 7,
                'Table name': 'AIPS FQ',
                '_status': [0] }

        # Write spectral window rows
        for ri, row_update in enumerate(rows, 1):
            row.update(row_update)
            fqtab.WriteRow(ri, row, err)
            handle_obit_err("Error writing row %d in FQ table. "
                            "Row data is '%s'" % (ri, row), err)

        # Close
        fqtab.Close(err)
        handle_obit_err("Error closing FQ table.")

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
