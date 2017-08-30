import datetime
import logging
from collections import OrderedDict, Counter
import time

import attr
import boltons.cacheutils
import numpy as np

import UVDesc

log = logging.getLogger('katsdpcontim')

def _aips_source_name(name):
    """ Truncates to length 16, padding with spaces """
    return "{:16.16}".format(name)

MEERKAT = 'MeerKAT'

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

    def select(self, **kwargs):
        """ Proxies :meth:`katdal.DataSet.select` """
        return self._katds.select(**kwargs)

    @property
    def shape(self):
        """ Proxies :meth:`katdal.DataSet.shape` """
        return self._katds.shape

    def uv_scans(self):
        """
        Generator returning vibility data for scan's selected
        via the :meth:`KatdalAdapter.select` method.

        Returns
        -------
        tuple
            (u, v, w, time, baseline_index, source_id, visibilities)
        """
        cp = self.correlator_products()
        nstokes = self.nstokes

        # Lexicographically sort correlation products on (a1, a2, cid)
        sort_fn = lambda x: (cp[x].ant1_ix,cp[x].ant2_ix,cp[x].cid)
        cp_argsort = np.asarray(sorted(range(len(cp)), key=sort_fn))
        corr_products = np.asarray([cp[i] for i in cp_argsort])

        # Take baseline products so that we don't recompute
        # UVW coordinates for all correlator products
        bl_products = corr_products.reshape(-1, nstokes)[:,0]
        nbl, = bl_products.shape

        # AIPS baseline IDs
        aips_baselines = np.asarray([bp.aips_bl_ix for bp in bl_products],
                                                        dtype=np.float32)

        # Get the AIPS visibility data shape (inaxes)
        # reverse to go from FORTRAN to C ordering
        fits_desc = self.fits_descriptor()
        inaxes = tuple(reversed(fits_desc['inaxes'][:fits_desc['naxis']]))

        # Mapping from katdal source name to UV source information
        uv_source_map = self.uv_source_map

        # Useful constants
        refwave = self.refwave
        midnight = self.midnight

        for si, state, target in self._katds.scans():
            # Retrieve UV source information for this scan
            try:
                aips_source = uv_source_map[target.name]
            except KeyError:
                logging.warn("Target '{}' will not be exported"
                                        .format(target.name))
                continue
            else:
                # Retrieve the source ID
                aips_source_id = aips_source['ID. NO.'][0]

            # Retrieve scan data (ntime, nchan, nbl*npol), casting to float32
            # nbl*npol is all mixed up at this point
            times = self._katds.timestamps[:]
            vis = self._katds.vis[:].astype(np.complex64)
            weights = self._katds.weights[:].astype(np.float32)
            flags = self._katds.flags[:]

            # Get dimension shapes
            ntime, nchan, ncorrprods = vis.shape

            log.info("Scan data shape '{}'".format(vis.shape))

            # Apply flags by negating weights
            weights[np.where(flags)] = -32767.0

            # AIPS visibility is [real, imag, weight]
            # Stacking gives us (ntime, nchan, nbl*npol, 3)
            vis = np.stack([vis.real, vis.imag, weights], axis=3)
            assert vis.shape == (ntime, nchan, ncorrprods, 3)

            # Reorganise correlation product dim so that
            # polarisations are grouped per baseline.
            # Then reshape to separate the two dimensions
            vis = vis[:,:,cp_argsort,:].reshape(ntime, nchan, nbl, nstokes, 3)

            # (1) transpose so that we have (ntime, nbl, nchan, npol, 3)
            # (2) reshape to include the full inaxes shape,
            #     including singleton nif, ra and dec dimensions
            vis = (vis.transpose(0,2,1,3,4)
                      .reshape((ntime,nbl,) + inaxes))

            log.info("Read visibilities of shape {} and size {:.2f}MB"
                .format(vis.shape, vis.nbytes / (1024.*1024.)))

            # Compute UVW coordinates from baselines
            # (3, ntimes, nbl)
            u, v, w = np.stack([target.uvw(bp.ant1, antenna=bp.ant2,
                                                     timestamp=times)
                                      for bp in bl_products], axis=2)

            assert u.shape == v.shape == w.shape == (ntime, nbl)

            # UVW coordinates in seconds
            aips_u, aips_v, aips_w = (c / refwave for c in (u, v, w))
            # Convert difference between timestep and
            # midnight on observation date to days.
            # This, combined with (probably) JDObs in the
            # UV descriptor, givens the Julian Date in days
            # of the visibility.
            aips_time = (times - midnight) / 86400.0

            # Yield this scan's data
            yield si, (aips_u, aips_v, aips_w,
                  aips_time, aips_baselines, aips_source_id,
                  vis)

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
            (antenna1, antenna2, correlator_product_id) tuples, with the
            correlator_product_id mapped in the following manner.

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
        return len(self._katds.channel_freqs)

    @property
    def frqsel(self):
        """
        The selected spectral window (FORTRAN-index)

        .. code-block:: python

            KA = KatdalAdapter(katdal.open('...'), spw=0)
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
            Reference wavelength
        """
        return 2.997924562e8/self.reffreq

    @boltons.cacheutils.cachedproperty
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
            'FREQ': self.channel_freqs[0],  # Reference frequency
            'FREQID': self.frqsel,          # Frequency setup id
            'RDATE': self.obsdat,           # Reference date
            'NO_IF': self.nif,              # Number of spectral windows
            # GST at 0h on reference data in degrees
            'GSTIA0': UVDesc.GST0(julian_date)*15.0,
            # Earth's rotation rate (degrees/day)
            'DEGPDY': UVDesc.ERate(julian_date)*360.0,
        }

    @boltons.cacheutils.cachedproperty
    def uv_antenna_rows(self):
        """
        Returns
        -------
        list
            List of dictionaries describing each antenna.
        """


        return [{
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
            'MNTSTA': [0]
        } for i, a in enumerate(sorted(self._katds.ants), 1)]

    @boltons.cacheutils.cachedproperty
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
            'FREQID': self.frqsel, # Frequency setup ID
            'VELDEF': 'RADIO',     # Radio/Optical Velocity?
            'VELTYP': 'LSR'        # Velocity Frame of Reference (LSR is default)
        }

    @boltons.cacheutils.cachedproperty
    def uv_source_rows(self):
        """
        Returns
        -------
        list
            List of dictionaries describing sources.
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
    def uv_spw_keywords(self):
        """
        Returns
        -------
        dict
            Dictionary containing updates to the AIPS FQ
            frequency table keywords.
        """
        return { 'NO_IF': self.nif }

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


    @boltons.cacheutils.cachedproperty
    def uv_spw_rows(self):
        """
        Returns
        -------
        list
            List of dictionaries describing each
            spectral window.
        """
        return [{
            # Fill in data from MeerKAT spectral window
            'FRQSEL': [i],
            'IF FREQ': [0.0],
            'CH WIDTH': [sw.channel_width],
            'RXCODE': ['L'],
            'SIDEBAND': [1 if sw.channel_width > 0.0 else -1],
            'TOTAL BANDWIDTH': [abs(sw.channel_width)*len(sw.channel_freqs)],
        } for i, sw in enumerate(self._katds.spectral_windows, 1)]


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
