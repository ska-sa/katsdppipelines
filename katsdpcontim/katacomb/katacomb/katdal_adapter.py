from collections import Counter
import datetime
import logging
import os
import time

import attr
import six
import numpy as np

import UVDesc

from katacomb import AIPSPath

log = logging.getLogger('katacomb')

ONE_DAY_IN_SECONDS = 24*60*60.0

def aips_timestamps(timestamps, midnight):
    """
    Given katdal timestamps and midnight on the observation date in UTC,
    calculates the Julian Day offset from midnight on the observation date.

    Parameters
    ----------
    timestamps : np.ndarray
        katdal UTC timestamps
    midnight : float
        midnight on day of observation in UTC

    Returns
    -------
    np.ndarray
        AIPS Julian Day timestamps, offset from midnight on the
        observation date.
    """
    return (timestamps - midnight) / ONE_DAY_IN_SECONDS

def katdal_timestamps(timestamps, midnight):
    """
    Given AIPS Julian day timestamps offset from midnight on the day
    of the observation, calculates the katdal UTC timestamp

    Parameters
    ----------
    timestamsp : np.ndarray
        AIPS Julian Day timestamps, offset from midnight on the
        observation date.
    midnight : float
        midnight on day of observation in UTC

    Returns
    -------
    np.ndarray
        katdal UTC timestamps
    """
    return midnight + (timestamps * ONE_DAY_IN_SECONDS)

def katdal_ant_nr(ant_name):
    """
    Given a MeerKAT antenna name of the form 'mnnnp' where
    'm' is a character constant, 'nnn' is the antenna number
    and 'p' is the polarisation, returns 'nnn'.

    Parameters
    ----------
    ant_name : str
        Antenna Name

    Returns
    ------
    integer
        katdal antenna number in MeerKAT antenna name
    """
    try:
        return int(ant_name[1:4])
    except (ValueError, IndexError) as e:
        raise ValueError("Invalid antenna name '%s'" % ant_name)

def aips_ant_nr(ant_name):
    """
    Given a MeerKAT antenna name of the form 'mnnnp' where
    'm' is a character constant, 'nnn' is the antenna number
    and 'p' is the polarisation, returns 'nnn+1'.

    Parameters
    ----------
    ant_name : str
        Antenna Name

    Returns
    ------
    integer
        AIPS antenna number from MeerKAT antenna name
    """
    return katdal_ant_nr(ant_name) + 1

def katdal_ant_name(aips_ant_nr):
    """ Return katdal antenna name, given the AIPS antenna number """
    return "m%03d" % (aips_ant_nr - 1)

def aips_uvw(uvw, refwave):
    """
    Converts katdal UVW coordinates in metres to AIPS UVW coordinates
    in wavelengths *at the reference frequency*.

    Notes
    -----
    Wavelengths at the reference frequency differs from AIPS documentation
    (AIPS Memo 117, Going AIPS, Obitdoc) which state that UVW coordinates
    should be in lightseconds.

    Parameters
    ----------
    uvw : np.ndarray
        katdal UVW coordinates in metres
    refwave : float
        Reference wavelength in metres

    Returns
    -------
    np.ndarray
        AIPS UVW coordinates in wavelengths at the reference frequency
    """
    return uvw / refwave

def katdal_uvw(uvw, refwave):
    """
    Converts AIPS UVW coordinates in wavelengths *at the reference frequency*
    to katdal UVW coordinates in metres. Set :function:`aips_uvw` for
    further discussion.

    Parameters
    ----------
    uvw : np.ndarray
        AIPS UVW coordinates in wavelengths at the reference frequency
    refwave : float
        Reference wavelength in metres

    Returns
    -------
    np.ndarray
        katdal UVW coordinates, in metres
    """
    return refwave * uvw

def aips_source_name(name):
    """ Truncates to length 16, padding with spaces """
    return "{:16.16}".format(name)

def aips_catalogue(katdata):
    """
    Creates a catalogue of AIPS sources from :attribute:`katdata.catalogue`
    It resembles :attribute:`katdal.Dataset.catalogue`.

    Returns a list of dictionaries following the specification
    of the AIPS SU table in AIPS Memo 117.

    Notes
    -----
    At present, the following is set for each source:

    1. AIPS Source ID
    2. Source name
    3. Source position

    Other quantities such as:

    1. Flux
    2. Frequency Offset
    3. Rest Frequency
    4. Bandwidth

    are defaulted to zero for now as these are not strictly required for
    the purposes of the continuum pipeline. See Bill Cotton's description
    of parameter and why it is unnecessary above each row entry in the code.

    Parameters
    ----------
    katdata : :class:`katdal.Dataset`
        katdal object

    Returns
    -------
    list of dicts
        List of dictionaries where each dictionary defines an AIPS source.

    """
    catalogue = []

    zero = np.asarray([0.0, 0.0])

    for aips_i, t in enumerate(katdata.catalogue.targets, 1):
        # Nothings have no position!
        if "Nothing" == t.name:
            radec = zero
            aradec = zero
        else:
            radec = t.radec()
            aradec = t.apparent_radec()

        # Right Ascension and Declination
        ra, dec = np.rad2deg(radec)
        # Apparent position
        raa, deca = np.rad2deg(aradec)

        aips_source_data = {
            # Fill in data derived from katpoint target
            'ID. NO.': [aips_i],
            'SOURCE': [aips_source_name(t.name)],
            'RAEPO': [ra],
            'DECEPO': [dec],
            'RAOBS': [raa],
            'DECOBS': [deca],
            'RAAPP': [raa],
            'DECAPP': [deca],
            'EPOCH': [2000.0],

            # NOTE(bcotton)
            # CALCODE -  is used to distinguish type and usage of calibrator
            # source and is used to carry intent from the scheduling process,
            # e.g. this is a bandpass calibrator.
            # Since this data will likely never be used to derive the
            # external calibration, it is unlikely to ever be used.
            'CALCODE': [' ' * 4],  # 4 spaces for calibrator code

            # Following seven key-values technically vary by
            # spectral window, but we're only handling one SPW
            # at a time so it doesn't matter

            # NOTE(bcotton)
            # I/Q/U/VFLUX are the flux densities, per spectral window,
            # either determined from a standard model or derived in the
            # calibration process from a standard calibrator.
            # Since this data will likely never be used to derive the
            # external calibration, they are unlikely to ever be used.
            'IFLUX': [0.0],
            'QFLUX': [0.0],
            'VFLUX': [0.0],
            'UFLUX': [0.0],

            # NOTE(bcotton)
            # LSRVEL, FREQOFF,RESTFREQ are used in making Doppler corrections
            # for the Earth's motion.  Since MeerKAT data can, in principle
            # include both HI and OH lines (separate spectral windows?)
            # the usage may be complicated.  Look at the documentation for
            # either Obit/CVel or AIPS/CVEL for more details on usage.
            # I'm not sure how the Doppler corrections will be made but
            # likely not on the data in question. In practice, for
            #  NRAO related telescopes this information is not reliable
            # and can be supplied as parameters to the correction software.
            # There are only a handful of transition available to MeerKAT
            # which all have very well known rest frequencies.
            # I don't know if MeerKAT plans online Doppler tracking which
            # I think is a bad idea anyway.
            'LSRVEL': [0.0],    # Velocity
            'FREQOFF': [0.0],   # Frequency Offset
            'RESTFREQ': [0.0],  # Rest Frequency

            # NOTE(bcotton)
            # BANDWIDTH was probably a mistake although, in principle,
            # it can be used to override the value in the FQ table.
            # I'm not sure if anything actually uses this.
            'BANDWIDTH': [0.0], # Bandwidth of the SPW

            # NOTE(bcotton)
            # PMRA, PMDEC are the proper motions of Galactic or extragalactic
            # objects. These are a rather muddled implementation as they also
            # need both equinox and epoch of the standard position, which for
            # Hipparcos positions are different. In practice, these are usually
            # included in the apparent position of date. I can't see these ever
            # being useful for MeerKAT as they are never more than a few mas/yr.
            # There is a separate Planetary Ephemeris (PO) table for solar
            # system objects which may be needed for the Sun or planets
            # (a whole different bucket of worms).
            # I can't see these ever being useful for MeerKAT as they are never
            # more than a few mas/yr. There is a separate
            # Planetary Ephemeris (PO) table for solar system objects which
            # may be needed for the Sun or planets
            # (a whole different bucket of worms).
            'PMRA': [0.0],      # Proper Motion in Right Ascension
            'PMDEC': [0.0],     # Proper Motion in Declination

            # NOTE(bcotton)
            # QUAL can be used to subdivide data for a given "SOURCE"
            # (say for a mosaic).# "SOURCE" plus "QUAL" uniquely define
            # an entry in the table. The MeerKAT field of view is SO HUGE
            # I can't see this being needed.
            'QUAL': [0.0],      # Source Qualifier Number
        }

        catalogue.append(aips_source_data)

    return catalogue

MEERKAT = 'MeerKAT'


class _KatdalTransformer(object):
    """
    Small wrapper around a katdal data attribute.

    Performs two functions

    1. Transforms katdal data into AIPS format
    2. Implements __getitem__ to proxy indexing calls
       to the underlying katdal data attribute.

    Basic idea is as follows:

    .. code-block:: python

        def __init__(self, ...):
            time_xform = lambda idx: (K.timestamps[idx] - midnight) / 86400.0
            self._time_xformer = _KatdalTransformer(time_xform,
                                    shape=lambda: K.timestamps.shape,
                                    dtype=K.timestamps.dtype)

        @property
        def timestamps(self):
            return self._time_xformer
    """
    def __init__(self, transform, shape, dtype):
        self._transform = transform
        self._shape = shape
        self._dtype = dtype

    def __getitem__(self, index):
        return self._transform(index)

    @property
    def shape(self):
        return self._shape()

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self.shape[0]

class KatdalAdapter(object):
    """
    Adapts a :class:`katdal.DataSet` to look
    a bit more like a UV file.

    This is not a true adapter, but perhaps if
    called that enough, it will become one.
    """

    def __init__(self, katds):
        """
        Constructs a KatdalAdapter.

        Parameters
        ----------
        katds : :class:`katdal.DataSet`
            An opened katdal dataset, probably an hdf5file
        """
        self._katds = katds
        self._cache = {}
        self._catalogue = aips_catalogue(katds)

        def _vis_xformer(index):
            """
            Transform katdal visibilities indexed by ``index``
            into AIPS visiblities.
            """
            vis = self._katds.vis[index]
            weights = self._katds.weights[index]
            flags = self._katds.flags[index]

            # Apply flags by negating weights
            weights[np.where(flags)] = -32767.0
            return np.stack([vis.real, vis.imag, weights], axis=3)

        def _time_xformer(index):
            """
            Transform katdal timestamps indexed by ``index``
            into AIPS timestamps. These are the Julian days
            since midnight on the observation date
            """
            return aips_timestamps(self._katds.timestamps[index], self.midnight)

        # Convert katdal UVW into AIPS UVW
        _u_xformer = lambda i: aips_uvw(self._katds.u[i], self.refwave)
        _v_xformer = lambda i: aips_uvw(self._katds.v[i], self.refwave)
        _w_xformer = lambda i: aips_uvw(self._katds.w[i], self.refwave)

        # Set up the actual transformers
        self._vis_xformer = _KatdalTransformer(_vis_xformer,
                                            shape=lambda: self._katds.vis.shape,
                                            dtype=self._katds.weights.dtype)

        self._time_xformer = _KatdalTransformer(_time_xformer,
                                            shape=lambda: self._katds.timestamps.shape,
                                            dtype=self._katds.timestamps.dtype)

        self._u_xformer = _KatdalTransformer(_u_xformer,
                                            shape=lambda: self._katds.u.shape,
                                            dtype=self._katds.u.dtype)
        self._v_xformer = _KatdalTransformer(_v_xformer,
                                            shape=lambda: self._katds.u.shape,
                                            dtype=self._katds.u.dtype)
        self._w_xformer = _KatdalTransformer(_w_xformer,
                                            shape=lambda: self._katds.u.shape,
                                            dtype=self._katds.u.dtype)

    @property
    def uv_timestamps(self):
        """ Returns times in Julian days since midnight on the Observation Date """
        return self._time_xformer

    @property
    def uv_vis(self):
        """ Returns AIPS visibilities """
        return self._vis_xformer

    @property
    def uv_u(self):
        """ U coordinate in seconds """
        return self._u_xformer

    @property
    def uv_v(self):
        """ V coordinate in seconds """
        return self._v_xformer

    @property
    def uv_w(self):
        """ W coordinate in seconds """
        return self._w_xformer

    def scans(self):
        """
        Generator iterating through scans in an observation.
        Proxies :meth:`katdal.Dataset.scans`.

        Yields
        ------
        scan_index : int
            Scan index
        state : str
            State
        aips_source : dict
            AIPS Source

        """
        for si, state, target in self._katds.scans():
            yield si, state, self._catalogue[self.target_indices[0]]

    def aips_path(self, **kwargs):
        """
        Constructs an aips path from a :class:`KatdalAdapter`

        Parameters
        ----------
        **kwargs (optional): :obj:
            See :class:`AIPSPath` for information on
            keyword arguments.

        Returns
        -------
        :class:`AIPSPath`
            AIPS path describing this observation
        """
        name = kwargs.pop('name', None)
        dtype = kwargs.get('dtype', "AIPS")

        if name is None:
            name = self._katds.obs_params.get('capture_block_id',
                                             self._katds.experiment_id)
            if dtype == 'AIPS':
                name = name[-10:]
            if dtype == "FITS":
                name += '.uvfits'

        return AIPSPath(name=name, **kwargs)

    def select(self, **kwargs):
        """ Proxies :meth:`katdal.DataSet.select` """
        return self._katds.select(**kwargs)

    @property
    def size(self):
        return self._katds.size

    @property
    def shape(self):
        """ Proxies :meth:`katdal.DataSet.shape` """
        return self._katds.shape

    @property
    def catalogue(self):
        """
        AIPS source catalogue, resembling
        :attribute:`katdal.Dataset.catalogue`.
        """
        return self._catalogue

    @property
    def scan_indices(self):
        """ Proxies :attr:`katdal.DataSet.scan_indices` """
        return self._katds.scan_indices

    @property
    def target_indices(self):
        """ Proxies :attr:`katdal.DataSet.target_indices` """
        return self._katds.target_indices

    @property
    def name(self):
        """ Proxies :attr:`katdal.DataSet.name` """
        return self._katds.name.encode()

    @property
    def experiment_id(self):
        """ Proxies :attr:`katdal.DataSet.name` """
        return self._katds.experiment_id

    @property
    def observer(self):
        """ Proxies :attr:`katdal.DataSet.observer` """
        return self._katds.observer

    @property
    def description(self):
        """ Proxies :attr:`katdal.DataSet.description` """
        return self._katds.description

    @property
    def version(self):
        """ Proxies :attr:`katdal.DataSet.version` """
        return self._katds.version

    @property
    def katdal(self):
        """ The `katdal.DataSet` adapted by this object """
        return self._katds

    @property
    def obsdat(self):
        """
        Returns
        -------
        str
            The observation date
        """
        start = time.gmtime(self._katds.start_time.secs)
        return time.strftime('%Y-%m-%d', start)

    @property
    def midnight(self):
        """
        Returns
        -------
        float
            Midnight on the observation date in unix seconds
        """
        return time.mktime(time.strptime(self.obsdat, '%Y-%m-%d'))

    @property
    def today(self):
        """
        Returns
        -------
        str
            The current date
        """
        return datetime.date.today().strftime('%Y-%m-%d')

    @property
    def observer(self):
        """
        Returns
        -------
        str
            The observer
        """
        return self._katds.observer

    """ Map correlation characters to correlation id """
    CORR_ID_MAP = {
        ('h', 'h'): 0,
        ('v', 'v'): 1,
        ('h', 'v'): 2,
        ('v', 'h'): 3,
    }

    def correlator_products(self):
        """
        Returns
        -------
        list
            CorrelatorProduct(antenna1, antenna2, correlator_product_id)
            objects, with the correlator_product_id mapped as follows:

            .. code-block:: python

                { ('h','h'): 0,
                  ('v','v'): 1,
                  ('h','v'): 2,
                  ('v','h'): 3 }
        """
        class CorrelatorProduct(object):
            def __init__(self, ant1, ant2, cid):
                self.ant1 = ant1
                self.ant2 = ant2
                self.cid = cid

            @property
            def ant1_ix(self):
                return katdal_ant_nr(self.ant1.name)

            @property
            def ant2_ix(self):
                return katdal_ant_nr(self.ant2.name)

            @property
            def aips_ant1_ix(self):
                return aips_ant_nr(self.ant1.name)

            @property
            def aips_ant2_ix(self):
                return aips_ant_nr(self.ant2.name)

            @property
            def aips_bl_ix(self):
                """ This produces the AIPS baseline index random parameter """
                return self.aips_ant1_ix * 256.0 + self.aips_ant2_ix

        # { name : antenna } mapping
        antenna_map = { a.name : a for a in self._katds.ants }
        products = []

        for a1_corr, a2_corr in self._katds.corr_products:
            # These look like 'm008v', 'm016h' etc.
            # Separate into name 'm008' and type 'v'
            a1_name = a1_corr[:4]
            a1_type = a1_corr[4:].lower()

            a2_name = a2_corr[:4]
            a2_type = a2_corr[4:].lower()

            # Derive the correlation id
            try:
                cid = self.CORR_ID_MAP[(a1_type, a2_type)]
            except KeyError:
                raise ValueError("Invalid Correlator Product "
                                 "['%s, '%s']" % (a1_corr, a2_corr))

            # Look up katdal antenna pair
            a1 = antenna_map[a1_name]
            a2 = antenna_map[a2_name]

            products.append(CorrelatorProduct(a1, a2, cid))

        return products

    @property
    def nstokes(self):
        """
        Returns
        -------
        int
            The number of stokes parameters in this observation,
            derived from the number of times we see a
            pair of antenna names in the correlation products.
        """

        # Count the number of times we see a correlation product
        counts = Counter((cp.ant1_ix, cp.ant2_ix) for cp
                         in self.correlator_products())
        return max(counts.itervalues())

    @property
    def nchan(self):
        """
        Returns
        -------
        int
            The number of channels in this observation.
        """
        return len(self._katds.channel_freqs)

    @property
    def frqsel(self):
        """
        The selected spectral window (FORTRAN-index)

        .. code-block:: python

            KA = KatdalAdapter(katdal.open('...'))
            assert KA.frqsel == 1

        Returns
        -------
        int
            The selected spectral window
        """
        return self._katds.spw + 1

    @property
    def nif(self):
        """
        The number of spectral windows, hard-coded to one,
        since :class:`katdal.DataSet` only supports selecting
        by one spectral window at a time.

        Returns
        -------
        int
            The number of spectral windows
        """
        return 1

    @property
    def channel_freqs(self):
        """
        Returns
        -------
        list or np.ndarray
            List of channel frequencies
        """
        return self._katds.channel_freqs

    @property
    def chinc(self):
        """
        Returns
        -------
        float
            The channel increment, or width.
        """
        return self._katds.channel_width

    @property
    def reffreq(self):
        """
        Returns
        -------
        float
            The first channel frequency as the reference frequency,
            rather than the centre frequency. See `uv_format.rst`.
        """
        return self._katds.channel_freqs[0]

    @property
    def refwave(self):
        """
        Returns
        -------
        float
            Reference wavelength in metres
        """
        return 2.997924562e8 / self.reffreq

    @property
    def uv_antenna_keywords(self):
        """
        Returns
        -------
        dict
            Dictionary containing updates to the AIPS AN
            antenna table keywords.
        """

        julian_date = UVDesc.PDate2JD(self.obsdat)

        return {
            'ARRNAM': MEERKAT,
            'FREQ': self.reffreq,           # Reference frequency
            'FREQID': self.frqsel,          # Frequency setup id
            'RDATE': self.obsdat,           # Reference date
            'NO_IF': self.nif,              # Number of spectral windows
            # GST at 0h on reference data in degrees
            'GSTIA0': UVDesc.GST0(julian_date) * 15.0,
            # Earth's rotation rate (degrees/day)
            'DEGPDY': UVDesc.ERate(julian_date) * 360.0,
        }

    @property
    def uv_antenna_rows(self):
        """
        Returns
        -------
        list
            List of dictionaries describing each antenna.
        """

        return [{
            # MeerKAT antenna information
            'NOSTA': [aips_ant_nr(a.name)],
            'ANNAME': [a.name],
            'STABXYZ': list(a.position_ecef),
            'DIAMETER': [a.diameter],
            'POLAA': [90.0],

            # Defaults for the rest
            'POLAB': [0.0],
            'POLCALA': [0.0, 0.0],
            'POLCALB': [0.0, 0.0],
            'POLTYA': ['X'],
            'POLTYB': ['Y'],
            'STAXOF': [0.0],
            'BEAMFWHM': [0.0],
            'ORBPARM': [],
            'MNTSTA': [0]
        } for a in sorted(self._katds.ants)]

    @property
    def uv_source_keywords(self):
        """
        Returns
        -------
        dict
            Dictionary containing updates to the AIPS SU
            source table keywords.
        """
        return {
            'NO_IF': self.nif,     # Number of spectral windows
            'FREQID': self.frqsel,  # Frequency setup ID
            'VELDEF': 'RADIO',     # Radio/Optical Velocity?
            # Velocity Frame of Reference (LSR is default)
            'VELTYP': 'LSR'
        }

    @property
    def uv_source_rows(self):
        """
        Returns
        -------
        list
            List of dictionaries describing sources.
        """
        return [self._catalogue[ti] for ti in self._katds.target_indices]

    @property
    def uv_spw_keywords(self):
        """
        Returns
        -------
        dict
            Dictionary containing updates to the AIPS FQ
            frequency table keywords.
        """
        return {'NO_IF': self.nif}

    @property
    def max_antenna_number(self):
        """
        Returns
        -------
        integer
            The maximum AIPS antenna number
        """
        return max(r['NOSTA'][0] for r in self.uv_antenna_rows)

    @property
    def uv_calibration_keywords(self):
        """
        Returns
        -------
        dict
            Dictionary containing updates to the AIPS CL
            calibration table keywords.
        """
        return {
            'NO_IF': self.nif,
            'NO_POL': self.nstokes,
            'NO_ANT': self.max_antenna_number,
            'NO_TERM': 1,
            'MFMOD': 1
        }

    @property
    def uv_spw_rows(self):
        """
        Returns
        -------
        list
            List of dictionaries describing each
            spectral window.
        """

        spw = self._katds.spectral_windows[self._katds.spw]
        bandwidth = abs(self.chinc) * self.nchan

        return [{
            # Fill in data from MeerKAT spectral window
            'FRQSEL': [self.frqsel],        # Frequency setup ID
            'IF FREQ': [0.0],
            'CH WIDTH': [self.chinc],
            # Should be 'BANDCODE' according to AIPS MEMO 117!
            'RXCODE': [spw.band],
            'SIDEBAND': [spw.sideband],
            'TOTAL BANDWIDTH': [bandwidth],
        }]

    def fits_descriptor(self):
        """ FITS visibility descriptor setup """

        nstokes = self.nstokes
        # Set STOKES CRVAL according to AIPS Memo 117
        # { RR: -1.0, LL: -2.0, RL: -3.0, LR: -4.0,
        #   XX: -5.0, YY: -6.0, XY: -7.0, YX: -8.0,
        #   I: 1, Q: 2, U: 3, V: 4 }
        stokes_crval = 1.0 if nstokes == 1 else -5.0
        stokes_cdelt = -1.0 # cdelt is always -1.0

        return {
            'naxis': 6,
            'ctype': ['COMPLEX', 'STOKES', 'FREQ', 'IF', 'RA', 'DEC'],
            'inaxes': [3, self.nstokes, self.nchan, self.nif, 1, 1],
            'cdelt': [1.0, stokes_cdelt, self.chinc, 1.0, 0.0, 0.0],
            'crval': [1.0, stokes_crval, self.reffreq, 1.0, 0.0, 0.0],
            'crpix': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'crota': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }

    def default_table_cmds(self):
        """
        Returns
        -------
        dict
        """
        return {
            "AIPS AN" : {
                "attach" : {'version': 1},
                "keywords" : self.uv_antenna_keywords,
                "rows": self.uv_antenna_rows,
                "write": True,
            },
            "AIPS FQ" : {
                "attach" : {'version': 1, 'numIF': 1 },
                "keywords" : self.uv_spw_keywords,
                "rows": self.uv_spw_rows,
                "write": True,
            },
            "AIPS SU" : {
                "attach" : {'version': 1},
                "keywords" : self.uv_source_keywords,
                "rows": self.uv_source_rows,
                "write": True,
            },
            "AIPS NX" : {
                "attach" : {'version': 1},
            },
        }

    def uv_descriptor(self):
        """
        Returns
        -------
        dict
            UV descriptor dictionary, derived from the metadata
            of a katdal file. Suitable for merging into a UVDesc dictionary
            when creating a new AIPS UV data file.
        """

        # FITS descriptor is the base for our uv_descriptor
        desc = self.fits_descriptor()

        desc.update({
            # Observation
            'obsdat': self.obsdat,
            'observer': self.observer,
            'origin': 'katdal export',
            'JDObs': UVDesc.PDate2JD(self.obsdat),
            'date': self.today,
            'epoch': 2000.0,
            'equinox': 2000.0,
            'teles': MEERKAT,
            'instrume': MEERKAT,

            'isort': 'TB',     # Time, Baseline sort order
            'object': 'MULTI',  # Source, this implies multiple sources

            'nvis': 0,
            'firstVis': 0,
            'numVisBuff': 0,

            # These are automatically calculated, but
            # are left here for illustration
            # 'incs' : 3,          # Stokes 1D increment. 3 floats in COMPLEX
            # 'incf' : 12,         # Frequency 1D increment, 12 = 3*4 STOKES
            # 'incif' : 49152,     # Spectral window 1D increment
            #                      # 49152 = 3*4*4096 CHANNELS

            # Regular parameter indices into ctypes/inaxes/cdelt etc.
            'jlocc': 0,   # COMPLEX
            'jlocs': 1,   # SOURCES
            'jlocf': 2,   # FREQ
            'jlocif': 3,  # IF
            'jlocr': 4,   # RA
            'jlocd': 5,   # DEC
        })

        # Random parameter keys, indices and coordinate systems
        # index == -1 indicates its absence in the Visibility Buffer
        RP = attr.make_class("RandomParameters", ["key", "index", "type"])

        random_parameters = [
            RP('ilocu', 0, 'UU-L-SIN'),  # U Coordinate
            RP('ilocv', 1, 'VV-L-SIN'),  # V Coordinate
            RP('ilocw', 2, 'WW-L-SIN'),  # W Coordinate
            RP('ilocb', 3, 'BASELINE'),  # Baseline ID
            RP('iloct', 4, 'TIME1'),     # Timestamp
            RP('iloscu', 5, 'SOURCE'),   # Source Index

            RP('ilocfq', -1, 'FREQSEL'),  # Frequency setup ID
            RP('ilocit', -1, 'INTTIM'),  # Integration Time
            RP('ilocid', -1, 'CORR-ID'),  # VLBA-specific
            RP('ilocws', -1, 'WEIGHT'),  # Ignore

            # The following used when 'BASELINE' id
            # can't be calculated because number of antenna > 255
            RP('iloca1', -1, 'ANTENNA1'),
            RP('iloca2', -1, 'ANTENNA2'),
            RP('ilocsa', -1, 'SUBARRAY'),
        ]

        # Construct parameter types for existent random parameters
        ptype = [rp.type for rp in random_parameters if not rp.index == -1]

        # Update with random parameters
        desc.update({rp.key: rp.index for rp in random_parameters})
        desc.update({'nrparm': len(ptype), 'ptype': ptype})

        return desc

def time_chunked_scans(kat_adapter, time_step=2):
    """
    Generator returning vibility data each scan, chunked
    on the time dimensions in chunks of ``time_step``.
    Internally, this iterates through :code:`kat_adapter.scan()`
    to produce a :code:`(si, state, source, data_gen)` tuple,
    where :code:`si` is the scan index, :code:`state` the state
    of the scan and :code:`source` the AIPS source dictionary.
    :code:`data_gen` is itself a generator that yields
    :code:`time_step` chunks of the data.

    Parameters
    ----------
    kat_adapter : `KatdalAdapter`
        Katdal Adapter
    time_step : integer
        Size of time chunks (Default 2).
        2 timesteps x 32768 channels x 2016 baselines x 4 stokes x 8 bytes
        works out to about ~3.9375 GB

    Yields
    ------
    si : int
        Scan index
    state : str
        Scan state
    aips source : dict
        Dictionary describing AIPS source
    data_gen : generator
        Generator yielding (u, v, w, time, baseline_index, visibilities)
        where :code:`len(time) <= time_step`
    """
    cp = kat_adapter.correlator_products()
    nstokes = kat_adapter.nstokes

    # Lexicographically sort correlation products on (a1, a2, cid)
    sort_fn = lambda x: (cp[x].ant1_ix, cp[x].ant2_ix, cp[x].cid)
    cp_argsort = np.asarray(sorted(range(len(cp)), key=sort_fn))
    corr_products = np.asarray([cp[i] for i in cp_argsort])

    # Use first stokes parameter index of each baseline
    bl_argsort = cp_argsort[::nstokes]

    # Take baseline products so that we don't recompute
    # UVW coordinates for all correlator products
    bl_products = corr_products.reshape(-1, nstokes)[:, 0]
    nbl, = bl_products.shape

    # AIPS baseline IDs
    aips_baselines = np.asarray([bp.aips_bl_ix for bp in bl_products],
                                dtype=np.float32)

    # Get the AIPS visibility data shape (inaxes)
    # reverse to go from FORTRAN to C ordering
    fits_desc = kat_adapter.fits_descriptor()
    inaxes = tuple(reversed(fits_desc['inaxes'][:fits_desc['naxis']]))

    def _get_data(time_start, time_end):
        """
        Retrieve data for the given time index range.

        Parameters
        ----------
        time_start : integer
            Start time index for this scan
        time_end : integer
            Ending time index for this scan

        Returns
        -------
        u : np.ndarray
            AIPS baseline U coordinates
        v : np.ndarray
            AIPS baseline V coordinates
        w : np.ndarray
            AIPS baseline W coordinates
        time : np.ndarray
            AIPS timestamp
        baselines : np.ndarray
            AIPS baselines id's
        vis : np.ndarray
            AIPS visibilities
        """
        _, nchan, ncorrprods = kat_adapter.shape
        ntime = time_end - time_start

        chunk_shape = (ntime,nchan,ncorrprods)
        cplx_size = np.dtype('complex64').itemsize
        vis_size_estimate = np.product(chunk_shape, dtype=np.int64)*cplx_size

        FOUR_GB = 4*1024**3

        if vis_size_estimate > FOUR_GB:
            log.warn("Visibility chunk '%s' is greater than '%s'. "
                    "Check that sufficient memory is available"
                    % (fmt_bytes(vis_size_estimate), fmt_bytes(FOUR_GB)))

        # Retrieve scan data (ntime, nchan, nbl*nstokes)
        # nbl*nstokes is all mixed up at this point
        aips_time = kat_adapter.uv_timestamps[time_start:time_end]
        aips_vis = kat_adapter.uv_vis[time_start:time_end]
        aips_u = kat_adapter.uv_u[time_start:time_end]
        aips_v = kat_adapter.uv_v[time_start:time_end]
        aips_w = kat_adapter.uv_w[time_start:time_end]

        # Check dimension shapes
        assert aips_vis.dtype == np.float32
        assert aips_time.dtype == np.float64
        assert aips_u.dtype == np.float64
        assert aips_v.dtype == np.float64
        assert aips_w.dtype == np.float64
        assert (ntime,) == aips_time.shape
        assert (ntime, nchan, ncorrprods, 3) == aips_vis.shape
        assert (ntime, ncorrprods) == aips_u.shape
        assert (ntime, ncorrprods) == aips_v.shape
        assert (ntime, ncorrprods) == aips_w.shape

        # Reorganise correlation product dim so that
        # correlations are grouped per baseline.
        # Then reshape to separate the two dimensions
        aips_vis = aips_vis[:, :, cp_argsort, :].reshape(
            ntime, nchan, nbl, nstokes, 3)

        # (1) transpose so that we have (ntime, nbl, nchan, nstokes, 3)
        # (2) reshape to include the full inaxes shape,
        #     including singleton nif, ra and dec dimensions
        aips_vis = (aips_vis.transpose(0, 2, 1, 3, 4)
                  .reshape((ntime, nbl,) + inaxes))

        # Select UVW coordinate of each baseline
        aips_u = aips_u[:,bl_argsort]
        aips_v = aips_v[:,bl_argsort]
        aips_w = aips_w[:,bl_argsort]

        assert aips_u.shape == (ntime, nbl)
        assert aips_v.shape == (ntime, nbl)
        assert aips_w.shape == (ntime, nbl)

        # Yield this scan's data
        return (aips_u, aips_v, aips_w,
                   aips_time, aips_baselines,
                   aips_vis)

    # Iterate through scans
    for si, state, target in kat_adapter.scans():
        # Work out the data shape
        ntime, nchan, ncorrprods = kat_adapter.shape

        # Create a generator returning data
        # associated with chunks of time data.
        data_gen = (_get_data(ts, min(ts+time_step, ntime)) for ts
                                in six.moves.range(0, ntime, time_step))

        # Yield scan variables and the generator
        yield si, state, target, data_gen
