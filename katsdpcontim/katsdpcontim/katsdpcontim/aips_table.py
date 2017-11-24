import attr
import collections
import logging

import six

import InfoList
import Table

from katsdpcontim import handle_obit_err
from katsdpcontim.obit_types import (OBIT_TYPE_ENUM,
                                     OBIT_TYPE,
                                     OBIT_INTS,
                                     OBIT_FLOATS,
                                     OBIT_STRINGS,
                                     OBIT_BOOLS,
                                     OBIT_BITS)

log = logging.getLogger('katsdpcontim')


def _scalarise(value):
    """ Converts length 1 lists to singletons """
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _vectorise(value):
    """ Converts singletons to length 1 lists """
    if not isinstance(value, list):
        return [value]
    return value


class AIPSTableKeywords(object):
    def __init__(self, table, table_name):
        """
        Constructs an :class:`AIPSTableKeywords` object.

        It behave like a dictionary.
        Keywords on the AIPS table configure certain properties
        associated with the table.

        Parameters
        ----------
        table: :class:`Table`
            AIPS table object
        table_name: string
            Table name
        """
        self._dirty = False
        self._table = table
        self._tabname = table_name
        self._schema = {key: (type_, dims) for
                        key, (type_, dims, value) in
                        table.IODesc.List.Dict.items()}

    def keys(self):
        return self._schema.keys()

    def iterkeys(self):
        return iter(self._schema.keys())

    def values(self):
        return [self.__getitem__(key) for key in self.keys()]

    def itervalues(self):
        return iter(self.values())

    def items(self):
        return [(k, self.__getitem__(k)) for k in self.keys()]

    def iteritems(self):
        return iter((k, self.__getitem__(k)) for k in self.keys())

    def __len__(self):
        return len(self._schema)

    def __iter__(self):
        return self.iterkeys()

    def __getitem__(self, key):
        """
        Get keyword value from the IO Descriptor object
        of the associated table.

        Parameter
        ---------
        key: string
            Keyword

        Returns
        -------
        value associated with the keyword
        """

        # Return value out of (code, name, type, dims, value) tuple
        # returned by PGet
        return _scalarise(InfoList.PGet(self._table.IODesc.List, key)[4])

    def __setitem__(self, key, value):
        """
        Set keyword value on the IO Descriptor object
        of the associated table.

        Parameter
        ---------
        key: string
            Keyword
        value:
            value
        """

        try:
            # Look up the type and dimensionality associated
            # with this key value pair
            type_, dims = self._schema[key]
        except KeyError:
            raise ValueError("'%s' is not a valid keyword "
                             "for table '%s' ."
                             "Valid keywords '%s'." % (
                                 key, self._tabname, self._schema.keys()))

        # Convert value into a list
        value = _vectorise(value)

        enum = OBIT_TYPE_ENUM[type_]

        # Coerce types, being sure to pad strings to their full length
        if type_ == OBIT_TYPE.string:
            value = [enum.coerce(v).ljust(dims[0], ' ') for v in value]
        else:
            value = [enum.coerce(v) for v in value]

        InfoList.PSetDict(self._table.IODesc.List,
                          {key: [type_, dims, value]})

        self._dirty = True

    def update(self, other=None, **kwargs):
        """
        Provides a :meth:`dict.update`-like method for
        mass update of keywords.

        Parameters
        ----------
        other: iterable
            Sequence of (k, v) tuples or a {k: v} mapping
        **kwargs (optional):
            key values to set.
        """
        if other is not None:
            is_map = isinstance(other, collections.Mapping)
            for k, v in other.iteritems() if is_map else other:
                self.__setitem__(k, v)

        for k, v in kwargs.iteritems():
            self.__setitem__(k, v)

    def __pretty__(self, p, cycle):
        """ Pretty print this keyword object """

        p.pretty(self._table.IODesc.List.Dict)

    @property
    def dirty(self):
        """
        Returns `True` if keywords have been modified and
        the Table should be updated.
        """
        return self._dirty

    def __str__(self):
        return str({key: _scalarise(value) for
                    key, (type_, dims, value) in
                    self._table.IODesc.List.Dict.items()})

    __repr__ = __str__


AIPSTableField = attr.make_class("AIPSTableField",
                                 ["name", "unit", "dims", "repeat", "type"],
                                 frozen=True, slots=True)


def _default_row_base(row_def, row):
    """
    Returns default row definition updated
    with the contents of `row`.

    Parameters
    ----------
    row_def: dict
        Definition for the row
    row: dict
        Dictionary describing the row

    Returns
    -------
    dict
        default dictionary updated with
        the contents of `row`
    """
    base_row = row_def.copy()
    base_row.update(row)
    return base_row


class AIPSTableRow(object):
    def __init__(self, table, row_nr, row_def, err, row=None):
        """
        Creates a :class:`AIPSTableRow`.

        It behaves like a dictionary.

        table: :class:`Table`
            An Obit table object
        rownr: integer
            Number of this row within the AIPS table
        row_def: dict
            Dictionary defining the type, shape and defaults
            of this row.
        err: :class:`OErr`
            An Obit error object
        row (optional): dict
            If supplied this will be used to define the row,
            rather than loading the row from the Obit table object.
        """

        self._table = table
        self._rownr = row_nr
        self._row_def = row_def
        self._err = err
        self._row = None if row is None else _default_row_base(row_def, row)

    def read(self):
        """ Reads the appropriate row of the AIPS Table """
        self._row = self._table.ReadRow(self._rownr, self._err)
        handle_obit_err("Error loading row '%s'" % self._rownr, self._err)

    def write(self):
        """ Writes the appropriate row of the AIPS Table """
        if self._row is None:
            return

        self._table.WriteRow(self._rownr, self._row, self._err)
        handle_obit_err("Error writing row '%s'" % self._rownr, self._err)

    def __pretty__(self, p, cycle):
        """ Pretty print this row """
        if cycle:
            p.text("{...}")
            return

        if self._row is None:
            self.read()

        p.pretty(self._row)

    def asdict(self):
        if self._row is None:
            self.read()

        return self._row

    def __getitem__(self, key):
        """
        Parameters
        ----------
        key: string
            Row key to return

        Returns
        -------
        value associated with `key` in this row
        """

        if self._row is None:
            self.read()

        return _scalarise(self._row[key])

    def __setitem__(self, key, value):
        """
        Parameters
        ----------
        key: string
            Row key to set
        value:
            value to set in this row
        """

        if self._row is None:
            self.read()

        try:
            type_, dims, defaults = self._row_def[key]
        except KeyError:
            raise ValueError("'%s' does not appear to be a "
                             "valid row key. Valid keys "
                             "include '%s'" % (key, self._row_def.keys()))

        # Pad string values appropriately
        if type_ == OBIT_TYPE.string:
            value = value.ljust(dims[0], ' ')

        self._row[key] = _vectorise(value)

    def __len__(self):
        """ Returns the number of elements in the row """
        return len(self._row)

    def __str__(self):
        """ Returns a string representation of the row """
        if self._row is None:
            self.read()
        return str(self._row)

    __repr__ = __str__


class AIPSTableRows(object):
    def __init__(self, table, nrow, row_def, err, rows=None):
        """
        Constructs a :class:`AIPSTableRows` object.

        It behaves like a list.

        Parameters
        ----------
        table: :class:`Table`
            An Obit table object
        nrow: integer
            Number of rows in the table
        row_def: dict
            A dictionary defining the type, shape and
            default values for this row
        err: :class:`OErr`
            An Obit error stack object
        rows (optional): list of row dictionaries
            If present, these will define the table rows.
            If None, table rows will be lazy loaded from disk
        """
        # No rows provided, lazy loaded from file on first access
        if rows is None:
            rows = nrow * [None]

        self._rows = [AIPSTableRow(table, ri + 1, row_def, err, row=row)
                      for ri, row in zip(range(nrow), rows)]

        self._table = table
        self._row_def = row_def
        self._err = err

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index: integer
            Row index

        Returns
        -------
        dict
            row at the specified index
        """
        return self._rows[index].asdict()

    def __setitem__(self, index, row):
        """
        Parameters
        ----------
        index: integer
            Row index
        row: dict
            Dictionary describing the row
        """

        self._rows[index] = _default_row_base(self._row_def, row)

    def __len__(self):
        """ Returns the number of rows """
        return len(self._rows)

    def append(self, row):
        """ Appends a row to this list """
        obj_row = AIPSTableRow(self._table, len(self) + 1,
                               self._row_def, self._err,
                               row=_default_row_base(self._row_def, row))
        self._rows.append(obj_row)

    def read(self):
        """ Force a read of all rows """
        for r in self._rows:
            r.read()

    def write(self):
        """ Force a write of all rows """
        for r in self._rows:
            r.write()

    def __pretty__(self, p, cycle):
        if cycle:
            p.text('[...]')
        else:
            p.pretty(self._rows)

    def __str__(self):
        return str(self._rows)

    __repr__ = __str__


class AIPSTable(object):
    FIELD_KEYS = ('FieldName', 'FieldUnit',
                  'dim0', 'dim1', 'dim2',
                  'repeat', 'type')

    def __init__(self, uv, name, version, mode, err, **kwargs):
        """
        Creates an AIPS Table object

        Parameters
        ----------
        uv: :class:`UV`
            Obit UV object associated with this table.
        name: string
            Table name. e.g. "AIPS AN"
        version: integer
            Table version
        mode: string
            "r" to read, "w" to write.
        err: :class:`OErr`
            Obit error stack
        **kwargs (optional): integer, usually
            Additional keyword arguments to pass in to
            the table constructor. For example, the "AIPS FQ"
            frequency table takes a `numIF` keyword argument
            specified the number of spectral windows.
        """
        self._err = err

        if 'w' in mode:
            self._clobber_old_tables(uv, name, err)

        self._table = table = uv.NewTable(Table.READWRITE,
                                          name, version,
                                          err, **kwargs)
        handle_obit_err("Error creating table '%s'" % name, err)

        self._table.Open(Table.READWRITE, err)
        handle_obit_err("Error opening table '%s'" % name, err)

        desc = table.Desc.Dict

        nrow = desc['nrow']
        self._name = name = desc['Table name']
        self._version = desc['version']

        self._keywords = AIPSTableKeywords(self._table, name)
        self._fields = fields = self._get_field_defs(desc)
        self._default_row = self._get_row_definitions(self._name, fields)

        self._rows = AIPSTableRows(table, nrow, self._default_row, err)

    def _clobber_old_tables(self, uv, name, err):
        """
        Removes all previously existing versions of table.

        Parameters
        ----------
        uv: :class:`UV`
            Obit UV object
        name: string
            AIPS Table name, "AIPS AN" for instance.
        err: :class:`OErr`
            Obit error stack object
        """
        prev_ver = uv.GetHighVer(name)

        while prev_ver > 0:
            uv.ZapTable(name, prev_ver, err)
            handle_obit_err("Error removing old '%s' table" % name, err)
            prev_ver = uv.GetHighVer(name)

    def read(self):
        pass

    def write(self):
        self._rows.write()
        desc = self._table.Desc.Dict
        desc['nrow'] = len(self._rows)
        self._table.Desc.Dict = desc
        Table.PDirty(self._table)

    @property
    def rows(self):
        """ List of rows on this table """
        return self._rows

    @rows.setter
    def rows(self, rows):
        """ Set the table rows on this table """
        self._rows = AIPSTableRows(self._table, len(rows), self._default_row,
                                   self._err, rows=rows)

    @classmethod
    def _get_field_defs(cls, desc):
        """
        Create a field definition dictionary

        Parameters
        ----------
        desc: dict
            Table Descriptor Dictionary obtained via
            :code:`table.Desc.Dict` for example.

        Returns
        dict
            Field definition dictionary of the form
            { field_name: :class:`AIPSTableField` }
        """
        return collections.OrderedDict(
            (n, AIPSTableField(n, u, [d0, d1, d2], r, t)) for
            n, u, d0, d1, d2, r, t in
            zip(*(desc[k] for k in cls.FIELD_KEYS)))

    @classmethod
    def _get_row_definitions(cls, table_name, fields):
        """
        Return row definition for each field.

        Parameters
        ----------
        table_name: string
            Table name, 'AIPS AN' for example.
        fields: dict
            Dictionary of form { field_name: :class:`AIPSTableField` }

        Returns
        -------
        dict
            Dictionary of { field_name: [type, dims, value]}
            defining a default row definition for each field.
        """

        def _repeat_default(f):
            """ Produce a default row for each field `f` """
            if f.type in OBIT_INTS:
                return f.repeat * [0]
            elif f.type in OBIT_BITS:
                return f.repeat * [0]
            elif f.type in OBIT_FLOATS:
                return f.repeat * [0.0]
            elif f.type in OBIT_STRINGS:
                return [f.repeat * ' ' for _ in range(f.dims[1])]
            elif f.type in OBIT_BOOLS:
                return f.repeat * [False]
            else:
                enum = OBIT_TYPE_ENUM[f.type]
                raise ValueError("Unhandled defaults for field "
                                 "'%s' with type '%s' (%d: %s)" %
                                 (f.name, enum.description,
                                  enum.enum, enum.name))

        defaults = {f.name: [f.type, f.dims + [1, 1], _repeat_default(f)]
                    for f in fields.values()}

        # This works and is pretty hacky since these aren't
        # technically fields, but they're needed
        # for a row definition.
        # TODO: Move this into AIPSTableRow
        defaults.update({"Table name": table_name,
                         "NumFields": len(fields),
                         "_status": [0]})
        return defaults

    @property
    def name(self):
        """ Name of this table """
        return self._name

    @property
    def keywords(self):
        """ Table keyword dictionary """
        return self._keywords

    @property
    def default_row(self):
        """ Default rows for each field of this table """
        return self._default_row

    @property
    def fields(self):
        """ Field definitions for this table """
        return self._fields

    @property
    def nrow(self):
        """ Number of rows in this table """
        return len(self._rows)

    @property
    def version(self):
        """ Table version """
        return self._version

    def close(self):
        """ Close the AIPS table """

        # Flush
        Table.PDirty(self._table)

        self._table.Open(Table.READWRITE, self._err)
        handle_obit_err("Error opening table '%s' for flush" % self._name,
                        self._err)
        # Close
        self._table.Close(self._err)
        handle_obit_err("Error closing '%s' table" % self._name, self._err)

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        return self.close()
