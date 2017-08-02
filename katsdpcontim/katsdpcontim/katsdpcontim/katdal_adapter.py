from collections import OrderedDict, Counter
import time

import attr
import boltons.cacheutils

import UVDesc

def _aips_source_name(name):
    """ Truncates to length 16, padding with spaces """
    return "{:16.16}".format(name)

class KatdalAdapter(object):
    """
    Adapts a katdal data source to look a bit more like a UV data source.

    This is not a true adapter, but perhaps if
    called that enough, it will become one.
    """
    def __init__(self, katds):
        """
        Constructs a KatdalAdapter.

        Parameters
        ----------
        katds : katdal data source object
            An opened katdal data source, probably an hdf5file
        """
        self._katds = katds
        self._cache = {}

        nspw = len(self._katds.spectral_windows)

        if nspw != 1:
            raise ValueError("Only handling one Spectral Window for now, "
                            "but found '{}'.".format(nspw))

    @boltons.cacheutils.cachedmethod('_cache')
    def _antenna_map(self):
        """
        Returns
        -------
        dict
            A { antenna_name: antenna } mapping
        """
        return OrderedDict((a.name, a) for a in sorted(self._katds.ants))

    @property
    def obsdat(self):
        """
        Returns
        -------
        str
            The observation date
        """
        start = time.gmtime(self._katds.timestamps[0])
        return time.strftime('%Y-%m-%d', start)

    @property
    def observer(self):
        """
        Returns
        -------
        str
            The observer
        """
        return self._katds.observer


    """ Map correlations to Data Product Id """
    DATA_PRODUCT_ID_MAP = {
        ('h','h'): 0,
        ('v','v'): 1,
        ('h','v'): 2,
        ('v','h'): 3,
    }

    @boltons.cacheutils.cachedmethod('_cache')
    def _data_products(self):
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
        DataProduct = attr.make_class("DataProduct", ["ant1", "ant2", "id"])
        antenna_map = self._antenna_map()

        products = []

        for a1_corr, a2_corr in self._katds.corr_products:
            # These look like 'm008v', 'm016h' etc.
            # Separate into name 'm008' and type 'v'
            a1_name = a1_corr[:4]
            a1_type = a1_corr[4:].lower()

            a2_name = a2_corr[:4]
            a2_type = a2_corr[4:].lower()

            # Look up katdal antenna pair
            a1 = antenna_map[a1_name]
            a2 = antenna_map[a2_name]

            # Derive the data product id
            try:
                dp = self.DATA_PRODUCT_ID_MAP[(a1_type, a2_type)]
            except KeyError as e:
                raise ValueError("Invalid Correlator Product "
                                "['{}', '{}']".format(a1_corr, a2_corr))

            products.append(DataProduct(a1, a2, dp))

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
        counts = Counter((dp.ant1, dp.ant2) for dp in self._data_products())
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
            Presently derived from the first spectral window
        """
        return len(self._katds.spectral_windows[0].channel_freqs)

    @property
    def nif(self):
        """
        Returns
        -------
        int
            The number of spectral windows
        """
        return len(self._katds.spectral_windows)

    @property
    def chinc(self):
        """
        Returns
        -------
        float
            The channel increment, or width.
            Presently derived from the first spectral window.
        """
        return self._katds.spectral_windows[0].channel_width

    @property
    def reffreq(self):
        """
        Returns
        -------
        float
            The centre channel frequency.
            Presently derived from the first spectral window.
        """
        return self._katds.spectral_windows[0].centre_freq

    @boltons.cacheutils.cachedproperty
    def antenna_rows(self):
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
        return [{'NOSTA': [i+1],
                'ANNAME': [a.name],
                'STABXYZ': list(a.position_ecef),
                'DIAMETER': [a.diameter],
                'POLAA': [90.0]}
                    for i, a in enumerate(sorted(self._katds.ants))]

    @boltons.cacheutils.cachedproperty
    def spw_rows(self):
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
                  'IF FREQ': [-428000000.0],
                  'NumFields': 7,
                  'RXCODE': ['L'],
                  'SIDEBAND': [1],
                  'TOTAL BANDWIDTH': [856000000.0],
                  'Table name': 'AIPS FQ',
                  '_status': [0]}
        """
        return [{'FRQSEL': [i+1],
                'IF FREQ': [sw.channel_freqs[0] - sw.centre_freq],
                'CH WIDTH': [sw.channel_width],
                'RXCODE': ['L'],
                'SIDEBAND': [1 if sw.channel_width > 0.0 else -1],
                'TOTAL BANDWIDTH': [abs(sw.channel_width)*
                                    len(sw.channel_freqs)],
                'NumFields': 7,
                'Table name': 'AIPS FQ',
                '_status': [0],

                }
                for i, sw in enumerate(self._katds.spectral_windows)]


    def uv_descriptor(self):
        """
        Returns
        -------
        dict
            Dictionary containing observational and other KAT metadata.
            Suitable for merging into a UVDesc dictionary.
        """
        return {
            'obsdat': self.obsdat,
            'observer': self.observer,
            'JDObs': UVDesc.PDate2JD(self.obsdat),
            'naxis': 6,
            'inaxes': [3, self.nstokes, self.nchan, self.nif, 1, 1, 0],
            'cdelt': [1.0,-1.0, self.chinc, 1.0, 0.0, 0.0, 0.0],
            'crval': [1.0, -5.0, self.reffreq, 1.0, 0.0, 0.0, 0.0],
            'crota': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
