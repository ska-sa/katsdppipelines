import datetime
from collections import OrderedDict, Counter
import time

import attr
import boltons.cacheutils

import UVDesc

def _aips_source_name(name):
    """ Truncates to length 16, padding with spaces """
    return "{:16.16}".format(name)

MEERKAT = 'MeerKAT'

class KatdalAdapter(object):
    """
    Adapts a katdal data source to look a bit more like a UV data source.

    This is not a true adapter, but perhaps if
    called that enough, it will become one.
    """
    def __init__(self, katds, spw=0):
        """
        Constructs a KatdalAdapter.

        Parameters
        ----------
        katds : katdal data source object
            An opened katdal data source, probably an hdf5file
        spw : integer
            Index of spectral window to export
        """
        self._katds = katds
        self._cache = {}

        # Set the spectral window we're handling
        self._spw = self._katds.spectral_windows[spw]

    def select(self, **kwargs):
        """ Proxy katdal's select method """
        return self._katds.select(**kwargs)

    @boltons.cacheutils.cachedmethod('_cache')
    def _antenna_map(self):
        """
        Returns
        -------
        dict
            A { antenna_name: (index, antenna) } mapping
        """
        A = attr.make_class("IndexedAntenna", ["index", "antenna"])
        return OrderedDict((a.name, A(i, a)) for i, a
                            in enumerate(sorted(self._katds.ants)))

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
        ('h','h'): 0,
        ('v','v'): 1,
        ('h','v'): 2,
        ('v','h'): 3,
    }

    @boltons.cacheutils.cachedmethod('_cache')
    def correlator_products(self):
        """
        Returns
        -------
        list
            (antenna1, antenna2, data_product_id) tuples, with the
            data_product_id mapped in the following manner.

            .. code-block:: python

                { ('h','h'): 0,
                  ('v','v'): 1,
                  ('h','v'): 2,
                  ('v','h'): 3 }
        """
        attrs = ["ant1", "ant2", "ant1_ix", "ant2_ix", "cid"]
        CorrelatorProductBase = attr.make_class("CorrelatorProductBase", attrs)

        # Add properties onto the base class
        class CorrelatorProduct(CorrelatorProductBase):
            @property
            def aips_ant1_ix(self):
                return self.ant1_ix+1

            @property
            def aips_ant2_ix(self):
                return self.ant2_ix+1

            @property
            def aips_bl_ix(self):
                return self.aips_ant1_ix*256.0 + self.aips_ant2_ix

        antenna_map = self._antenna_map()

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
            except KeyError as e:
                raise ValueError("Invalid Correlator Product "
                                "['{}', '{}']".format(a1_corr, a2_corr))

            # Look up katdal antenna pair
            a1 = antenna_map[a1_name]
            a2 = antenna_map[a2_name]

            products.append(CorrelatorProduct(a1.antenna, a2.antenna,
                                        a1.index, a2.index, cid))

        return products

    @boltons.cacheutils.cachedproperty
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

    def _classify_targets(self):
        """
        Returns
        -------
        tuple
            A tuple, (bandpass, gain, image_targets),
            of dictionaries of the form { name: target }
        """
        bpcals = OrderedDict()
        gaincals = OrderedDict()
        img_targets = OrderedDict()

        for t in self._katds.catalogue.targets:
            if 'bpcal' in t.tags:
                bpcals[t.name] = t
            elif 'gaincal' in t.tags:
                gaincals[t.name] = t
            else: # Assume all other targets are for imaging
                img_targets[t.name] = t

        return bpcals, gaincals, img_targets

    @boltons.cacheutils.cachedproperty
    def _targets(self):
        """
        Returns
        -------
        OrderedDict
            Returns a {target_name : [ra, dec, raa, deca] } mapping,
            where (ra, dec) are right ascension and declination and
            (raa, deca) are their apparent counterparts.
        """
        Target = attr.make_class("Target", ["ra", "dec", "raa", "deca"])
        targets = OrderedDict()

        for i, ti in enumerate(self._katds.target_indices):
            t = self._katds.catalogue.targets[ti]

            # Ignore nothings
            if "Nothing" == t.name:
                continue

            # Get a valid AIPS Source Name
            name = _aips_source_name(t.name)

            # Right Ascension and Declination
            ras, decs = t.radec()
            ra  = UVDesc.PHMS2RA(str(ras).replace(':',' '))
            dec = UVDesc.PDMS2Dec(str(decs).replace(':',' '))

            # Apparent position
            ras, decs = t.apparent_radec()
            deca = UVDesc.PDMS2Dec(str(decs).replace(':',' '))
            raa  = UVDesc.PHMS2RA(str(ras).replace(':',' '))

            targets[name] = Target(ra, dec, raa, deca)


        return targets

    @property
    def nchan(self):
        """
        Returns
        -------
        int
            The number of channels in this observation.
        """
        return len(self._spw.channel_freqs)

    @property
    def nif(self):
        """
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
        return self._spw.channel_freqs

    @property
    def chinc(self):
        """
        Returns
        -------
        float
            The channel increment, or width.
        """
        return self._spw.channel_width

    @property
    def reffreq(self):
        """
        Returns
        -------
        float
            The first channel frequency as the reference frequency,
            rather than the centre frequency. See `uv_format.rst`.
        """
        return self._spw.channel_freqs[0]

    @property
    def refwave(self):
        """
        Returns
        -------
        float
            Reference wavelength
        """
        return 2.997924562e8/self.reffreq

    @boltons.cacheutils.cachedproperty
    def uv_antenna_header(self):
        """
        Returns
        -------
        dict
            Dictionary containing elements necessary
            for constructing an AN table header. Of the form:

            .. code-block:: python

                { 'RefDate': self.obsdat,
                  'Freq': self.channel_freqs[0] }
        """

        julian_date = UVDesc.PDate2JD(self.obsdat)

        return {
            'ArrName': MEERKAT,
            'Freq': self.channel_freqs[0],  # Reference frequency
            'FreqID': 1,                    # Frequency setup id
            'numIF': self.nif,              # Number of spectral windows
            'RefDate': self.obsdat,         # Reference date
            # GST at 0h on reference data in degrees
            'GSTiat0': UVDesc.GST0(julian_date)*15.0,
            # Earth's rotation rate (degrees/day)
            'DegDay': UVDesc.ERate(julian_date)*360.0,
        }

    @boltons.cacheutils.cachedproperty
    def uv_antenna_rows(self):
        """
        Returns
        -------
        list
            List of dictionaries describing each antenna, each
            with the following form:

            .. code-block:: python

                { 'NOSTA': [1],
                  'ANNAME': ['m003'],
                  'STABXYZ': [100.0, 200.0, 300.0],
                  'DIAMETER': [13.4],
                  'POLAA': [90.0] }
        """


        return [{
            # Book-keeping
            'Table name': 'AIPS AN',
            'NumFields': 15,
            '_status': [0],

            # MeerKAT antenna information
            'NOSTA': [i],
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
            'MNTSTA': [0] }
                    for i, a in enumerate(sorted(self._katds.ants), 1)]


    @boltons.cacheutils.cachedproperty
    def uv_source_header(self):
        """
        Returns
        -------
        dict
            Dictionary containing elements necessary
            for constructing a SU table header. Of the form:
        """
        return {
            'numIF': 1,         # Number of spectral windows
            'FreqID': 1,        # Frequency setup ID
            'velDef': 'RADIO',  # Radio/Optical Velocity?
            'velType': 'LSR'    # Velocity Frame of Reference (LSR is default)
        }

    @boltons.cacheutils.cachedproperty
    def uv_source_rows(self):
        """
        Returns
        -------
        list
            List of dictionaries describing sources,
            each with the following form

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
        return self.uv_source_map.values()

    @boltons.cacheutils.cachedproperty
    def uv_source_map(self):
        """
        Returns
        -------
        dict
            A { name: dict } mapping where `name` is a
            :py:class:`katpoint.Target` name and `dict` is a
            dictionary describing a UV source.
        """
        targets = OrderedDict()
        aips_src_index = 0
        bandwidth = self.channel_freqs[-1] - self.channel_freqs[0]

        for target_index in self._katds.target_indices:
            target = self._katds.catalogue.targets[target_index]

            # Ignore nothings
            if "Nothing" == target.name:
                continue

            aips_src_index += 1

            # Get a valid AIPS Source Name
            name = _aips_source_name(target.name)

            # AIPS Right Ascension and Declination
            ras, decs = target.radec()
            ra  = UVDesc.PHMS2RA(str(ras).replace(':',' '))
            dec = UVDesc.PDMS2Dec(str(decs).replace(':',' '))

            # AIPS Apparent Right Ascension and Declination
            ras, decs = target.apparent_radec()
            raa  = UVDesc.PHMS2RA(str(ras).replace(':',' '))
            deca = UVDesc.PDMS2Dec(str(decs).replace(':',' '))

            targets[target.name] = {
                    # Book-keeping
                    'Table name': 'AIPS SU',
                    'NumFields': 22,
                    '_status'  : [0],

                    # Fill in data derived from katpoint targets
                    'ID. NO.'  : [aips_src_index],
                    'SOURCE'   : [name],
                    'RAEPO'    : [ra],
                    'DECEPO'   : [dec],
                    'RAOBS'    : [raa],
                    'DECOBS'   : [deca],
                    'RAAPP'    : [raa],
                    'DECAPP'   : [deca],
                    'EPOCH'    : [2000.0],
                    'BANDWIDTH': [bandwidth],

                    # No calibrator, fill with spaces
                    'CALCODE'  : [' '*4], # 4 spaces for calibrator code

                    # Following seven key-values technically vary by spectral window
                    # Specify zero flux for sources since we don't know them yet
                    'IFLUX'    : [0.0],
                    'QFLUX'    : [0.0],
                    'VFLUX'    : [0.0],
                    'UFLUX'    : [0.0],
                    'LSRVEL'   : [0.0], # Velocity
                    'FREQOFF'  : [0.0], # Frequency Offset
                    'RESTFREQ' : [0.0], # Rest Frequency

                    # Don't have these, zero them
                    'PMRA'     : [0.0], # Proper Motion in Right Ascension
                    'PMDEC'    : [0.0], # Proper Motion in Declination
                    'QUAL'     : [0.0], # Source Qualifier Number
            }

        return targets

    @boltons.cacheutils.cachedproperty
    def uv_spw_header(self):
        """
        Returns
        -------
        dict
            Dictionary used in construction of
            the FQ table. Currently only contains
            :code:`{ 'numIF' : 1 }`, the (singular)
            number of spectral windows.
        """
        return { 'numIF': self.nif }

    @boltons.cacheutils.cachedproperty
    def max_antenna_number(self):
        """
        Returns
        -------
        integer
            The maximum AIPS antenna number
        """
        return max(r['NOSTA'][0] for r in self.uv_antenna_rows)

    @boltons.cacheutils.cachedproperty
    def uv_calibration_header(self):
        """
        Returns
        -------
        dict
            Dictionary used to construct the CL table header.
        """
        return {
            'numIF': self.nif,
            'numPol': self.nstokes,
            'numAnt': self.max_antenna_number,
            'numTerm': 1,
            'mGMod': 1
        }


    @boltons.cacheutils.cachedproperty
    def uv_spw_rows(self):
        """
        Returns
        -------
        list
            List of dictionaries describing each
            spectral window (1 at present),
            each with the following form:

            .. code-block:: python

                {'CH WIDTH': [208984.375],
                  'FRQSEL': [1],
                  'IF FREQ': [0.0],
                  'RXCODE': ['L'],
                  'SIDEBAND': [1],
                  'TOTAL BANDWIDTH': [856000000.0] }
        """
        return [{
            # Book-keeping
            'Table name': 'AIPS FQ',
            'NumFields': 7,
            '_status': [0],

            # Fill in data from MeerKAT spectral window
            'FRQSEL': [i],
            'IF FREQ': [0.0],
            'CH WIDTH': [sw.channel_width],
            'RXCODE': ['L'],
            'SIDEBAND': [1 if sw.channel_width > 0.0 else -1],
            'TOTAL BANDWIDTH': [abs(sw.channel_width)*
                                len(sw.channel_freqs)], }
                for i, sw in enumerate([self._spw], 1)]


    def fits_descriptor(self):
        """ FITS visibility descriptor setup """
        return {
            'naxis': 6,
            'ctype': ['COMPLEX', 'STOKES', 'FREQ', 'IF', 'RA', 'DEC'],
            'inaxes': [3, self.nstokes, self.nchan, self.nif, 1, 1],
            'cdelt': [1.0,-1.0, self.chinc, 1.0, 0.0, 0.0],
            'crval': [1.0, -5.0, self.reffreq, 1.0, 0.0, 0.0],
            'crpix': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'crota': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            'object': 'MULTI', # Source, this implies multiple sources

            'nvis': 0,
            'firstVis': 0,
            'numVisBuff': 0,


            # These are automatically calculated, but
            # are left here for illustration
            # 'incs' : 3,          # Stokes 1D increment. 3 floats in COMPLEX
            # 'incf' : 12,         # Frequency 1D increment, 12 = 3*4 STOKES
            # 'incif' : 49152,     # Spectral window 1D increment = 49152 = 3*4*4096 CHANNELS

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
        RP = attr.make_class("RandomParameters", ["key", "index", "type"] )

        random_parameters = [
            RP('ilocu', 0, 'UU-L-SIN'),  # U Coordinate
            RP('ilocv', 1, 'VV-L-SIN'),  # V Coordinate
            RP('ilovw', 2, 'WW-L-SIN'),  # W Coordinate
            RP('ilocb', 3, 'BASELINE'),  # Baseline ID
            RP('iloct', 4, 'TIME1'),     # Timestamp
            RP('iloscu', 5, 'SOURCE'),   # Source Index

            RP('ilocfq', -1, ''),        # FREQSEL
            RP('ilocit', -1, ''),        # INTTIM
            RP('ilocid', -1, ''),        # CORR-ID
            RP('ilocws', -1, ''),        # WEIGHT

            RP('iloca1', -1, ''),        # ANTENNA 1
            RP('iloca2', -1, ''),        # ANTENNA 2
            RP('ilocsa', -1, ''),        # SUBARRAY
        ]

        # Construct parameter types for existent random parameters
        ptype = [rp.type for rp in random_parameters if not rp.index == -1]

        # Update with random parameters
        desc.update({ rp.key: rp.index for rp in random_parameters })
        desc.update({'nrparm' : len(ptype), 'ptype' : ptype })

        return desc
