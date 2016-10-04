"""Data loading backend for CASA Measurement Sets."""

import katsdpimager.loader_core
import casacore.tables
import casacore.quanta
import numpy as np
import argparse
import astropy.units as units


def _getcol(table, name, start=0, count=-1,
            casacore_units=None, astropy_units=None,
            measinfo_type=None, measinfo_ref=None,
            virtual=False):
    """Wrap the getcol function to fetch a batch of data from a row, applying
    scaling to the expected units. Scaling is done manually rather than with
    :mod:`casacore.quanta`, because the latter only operates (slowly) on
    scalars while a direct scaling is vectorised.

    Because not all writers specify the units for all quantities, units are
    allowed to be missing.

    Parameters
    ----------
    table : `casacore.tables.table`
        Table from which to load
    name : str
        Column name
    start : int
        First row to load
    count : int
        Number of rows to load
    casacore_units : str, optional
        Expected unit. A column with compatible units will be converted.
        Use ``None`` for unitless quantities.
    astropy_units : :class:`astropy.units.Unit`, optional
        Astropy unit equivalent to `casacore_units`, used for the return
        value. Use ``None`` for unitless quantities (the return value will
        then be a plain numpy array).
    measinfo_type : str, optional
        If specified, requires the ``MEASINFO`` field keyword to have this
        type.
    measinfo_ref : str, optional
        If specified, requires the ``MEASINFO`` field keyword to have this
        reference.
    virtual : bool, optional
        Set to `True` for virtual columns. This is needed because casacore 2.1
        doesn't support fetching a range of rows on virtual columns
        (VirtualScalarColumn doesn't overload getScalarColumnCells).
    """
    if virtual:
        if count < 0:
            count = table.nrows() - start
        data = np.empty((count,), table.coldatatype(name))
        for i in range(count):
            data[i] = table.getcell(name, start + i)
    else:
        data = table.getcol(name, start, count)

    keywords = table.getcolkeywords(name)
    quantum_units = keywords.get('QuantumUnits')
    if quantum_units is not None:
        if casacore_units is None:
            raise ValueError('Found unexpected QuantumUnits for column {}'.format(name))
        quantum_units = np.array(quantum_units)
        if len(data.shape) == 1:
            # Scalar column. Convert quantum_units from a 1D array of 1 element
            # to a numpy scalar
            if quantum_units.shape != (1,):
                raise ValueError('Column {} is scalar, but QuantumUnits has shape {}'.format(
                                 name, quantum_units.shape))
            quantum_units = quantum_units[0]
        # data[0] is included in the iteration so that broadcasting rules are
        # triggered if quantum_units has a smaller shape (e.g. one unit for both
        # receptors on a feed).
        it = np.nditer([quantum_units, data[0]], flags=['multi_index'])
        output_quantity = casacore.quanta.quantity(1, casacore_units)
        while not it.finished:
            input_quantity = casacore.quanta.quantity(1, it[0][()])
            if not input_quantity.conforms(output_quantity):
                raise ValueError('Expected {} in column {} but found {} instead'.format(
                                 casacore_units, name, input_quantity.get_unit()))
            ratio = input_quantity.get_value(casacore_units)
            if ratio != 1:
                # Special-case ratio=1 because it is the common case, and the scaling
                # is potentially expensive.
                data[np.index_exp[:] + it.multi_index] *= ratio
            it.iternext()
    if astropy_units is not None:
        data = units.Quantity(data, astropy_units, copy=False)
    measinfo = keywords.get('MEASINFO')
    if measinfo is not None:
        if (measinfo_ref is not None and measinfo['Ref'] != measinfo_ref) or \
                (measinfo_type is not None and measinfo['type'] != measinfo_type):
            raise ValueError('Unsupported MEASINFO for {}: {}'.format(name, measinfo))
    return data


def _getcell(table, name, row,
             casacore_units=None, astropy_units=None,
             measinfo_type=None, measinfo_ref=None,
             virtual=False):
    """Like :meth:`_getcol`, but for a single cell"""
    data = _getcol(table, name, row, 1, casacore_units, astropy_units,
                   measinfo_type, measinfo_ref, virtual)
    return data[0]


class LoaderMS(katsdpimager.loader_core.LoaderBase):
    def __init__(self, filename, options):
        super(LoaderMS, self).__init__(filename, options)
        parser = argparse.ArgumentParser(
            prog='Measurement set options',
            usage='Measurement set options: [-i data=COLUMN] [-i field=FIELD]')
        parser.add_argument('--data', type=str, metavar='COLUMN', default='DATA', help='Column containing visibilities to image [%(default)s]')
        parser.add_argument('--data-desc', type=int, default=0, help='Data description ID to image [%(default)s]')
        parser.add_argument('--field', type=int, default=0, help='Field to image [%(default)s]')
        parser.add_argument('--pol-frame', choices=['sky', 'feed'], help='Reference frame for polarization [%(default)s]')
        args = parser.parse_args(options)
        self._main = casacore.tables.table(filename, ack=False)
        # Adds virtual columns for parallactic angle, amongst others
        try:
            casacore.tables.addDerivedMSCal(filename)
        except RuntimeError:
            pass  # Above fails if the columns already exist
        self._filename = filename
        self._antenna = casacore.tables.table(self._main.getkeyword('ANTENNA'), ack=False)
        self._data_description = casacore.tables.table(self._main.getkeyword('DATA_DESCRIPTION'), ack=False)
        self._field = casacore.tables.table(self._main.getkeyword('FIELD'), ack=False)
        self._spectral_window = casacore.tables.table(self._main.getkeyword('SPECTRAL_WINDOW'), ack=False)
        self._polarization = casacore.tables.table(self._main.getkeyword('POLARIZATION'), ack=False)
        self._feed = casacore.tables.table(self._main.getkeyword('FEED'), ack=False)
        self._data_col = args.data
        self._field_id = args.field
        self._data_desc_id = args.data_desc
        if self._data_col not in self._main.colnames():
            raise ValueError('{} has no column named {}'.format(
                filename, self._data_col))
        if args.field < 0 or args.field >= self._field.nrows():
            raise ValueError('Field {} is out of range'.format(args.field))
        if args.data_desc < 0 or args.data_desc >= self._data_description.nrows():
            raise ValueError('Data description {} is out of range'.format(args.data_desc))
        self._polarization_id = self._data_description.getcell(
            'POLARIZATION_ID', args.data_desc)
        self._spectral_window_id = self._data_description.getcell(
            'SPECTRAL_WINDOW_ID', args.data_desc)
        self._feed_angle_correction = args.pol_frame == 'feed'
        if self._feed_angle_correction:
            # Load per-antenna feed angles. For now, only a constant value per
            # antenna is supported, rather than per feed or per receptor within a
            # feed. This avoids the need for more per-visibility indexing.
            antenna_id = _getcol(self._feed, 'ANTENNA_ID')
            receptor_angle = _getcol(self._feed, 'RECEPTOR_ANGLE', 0, -1, 'rad', units.rad)
            antenna_angle = [None] * (np.max(antenna_id) + 1)
            for i in range(len(antenna_id)):
                for angle in receptor_angle[i]:
                    if (antenna_angle[antenna_id[i]] is not None
                            and not np.allclose(antenna_angle[antenna_id[i]], angle, atol=1e-8 * units.rad)):
                        raise ValueError('Multiple feed angles for one antenna is not supported')
                    antenna_angle[antenna_id[i]] = angle
            self._antenna_angle = units.Quantity(antenna_angle)
        else:
            self._antenna_angle = None

    @classmethod
    def match(cls, filename):
        return filename.lower().endswith('.ms')

    def antenna_diameters(self):
        return _getcol(self._antenna, 'DISH_DIAMETER', 0, -1, 'm', units.m)

    def antenna_positions(self):
        # We don't actually care about the frame of the positions, because
        # they're only used to compute the baseline lengths. Thus, no
        # restrictions are placed on MEASINFO.
        return _getcol(self._antenna, 'POSITION', 0, -1, 'm', units.m)

    def phase_centre(self):
        value = _getcell(self._field, 'PHASE_DIR', self._field_id,
                        'rad', units.rad, 'direction', 'J2000')
        if tuple(value.shape) != (1, 2):
            raise ValueError('Unsupported shape for PHASE_DIR: {}'.format(value.shape))
        return value[0, :]

    def frequency(self, channel):
        return _getcell(self._spectral_window, 'CHAN_FREQ', self._spectral_window_id,
                        'Hz', units.Hz)[channel]

    def polarizations(self):
        return _getcell(self._polarization, 'CORR_TYPE', self._polarization_id)

    def data_iter(self, channel, max_rows=None):
        if max_rows is None:
            max_rows = self._main.nrows()
        for start in xrange(0, self._main.nrows(), max_rows):
            end = min(self._main.nrows(), start + max_rows)
            flag_row = _getcol(self._main, 'FLAG_ROW', start, end - start)
            field_id = _getcol(self._main, 'FIELD_ID', start, end - start)
            data_desc_id = _getcol(self._main, 'DATA_DESC_ID', start, end - start)
            antenna1 = _getcol(self._main, 'ANTENNA1', start, end - start)
            antenna2 = _getcol(self._main, 'ANTENNA2', start, end - start)
            valid = np.logical_not(flag_row)
            valid = np.logical_and(valid, field_id == self._field_id)
            valid = np.logical_and(valid, data_desc_id == self._data_desc_id)
            valid = np.logical_and(valid, antenna1 != antenna2)
            data = _getcol(self._main, self._data_col, start, end - start, 'Jy', units.Jy)[valid, channel, ...]
            if self._feed_angle_correction:
                pa1 = _getcol(self._main, 'PA1', start, end - start, 'rad', units.rad, virtual=True)[valid, ...]
                pa2 = _getcol(self._main, 'PA2', start, end - start, 'rad', units.rad, virtual=True)[valid, ...]
                feed_angle1 = pa1 + self._antenna_angle[antenna1[valid]]
                feed_angle2 = pa2 + self._antenna_angle[antenna2[valid]]
            else:
                feed_angle1 = units.Quantity(np.zeros(data.shape[0], np.float32), units.rad)
                feed_angle2 = feed_angle1.copy()
            # Note: UVW is negated due to differing sign conventions
            uvw = -_getcol(self._main, 'UVW', start, end - start, 'm', units.m, 'uvw', 'ITRF')[valid, ...]
            if 'WEIGHT_SPECTRUM' in self._main.colnames():
                weight = _getcol(self._main, 'WEIGHT_SPECTRUM', start, end - start)[
                    valid, channel, ...]
            else:
                weight = _getcol(self._main, 'WEIGHT', start, end - start)[valid, ...]
            flag = _getcol(self._main, 'FLAG', start, end - start)[valid, ...]
            weight = weight * np.logical_not(flag[:, channel, :])
            baseline = (antenna1 * self._antenna.nrows() + antenna2)[valid, ...]
            yield dict(uvw=uvw, weights=weight, baselines=baseline, vis=data,
                       feed_angle1=feed_angle1, feed_angle2=feed_angle2,
                       progress=end, total=self._main.nrows())

    def close(self):
        self._main.close()
        self._antenna.close()
        self._data_description.close()
        self._field.close()
        self._spectral_window.close()
        self._polarization.close()
        self._feed.close()
        try:
            casacore.tables.removeDerivedMSCal(self._filename)
        except RuntimeError:
            # Above fails if the columns don't exist
            pass
