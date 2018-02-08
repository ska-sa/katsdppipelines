import random
import unittest

from ephem.stars import stars
import katpoint
import numpy as np
import six

from katsdptelstate import TelescopeState

from katacomb.mock_dataset import (MockDataSet,
                    ANTENNA_DESCRIPTIONS,
                    DEFAULT_METADATA,
                    DEFAULT_SUBARRAYS,
                    DEFAULT_TIMESTAMPS)

from katacomb import ContinuumPipeline
from katacomb.util import parse_python_assigns


class TestContinuumPipeline(unittest.TestCase):

    def test_continuum_pipeline(self):
        """
        Tests that a run of the Continuum Pipeline executes.
        """

        nchan = 16
        nvispio = 1024

        spws = [{
            'centre_freq' : .856e9 + .856e9 / 2.,
            'num_chans' : nchan,
            'channel_width' : .856e9 / nchan,
            'sideband' : 1,
            'band' : 'L',
        }]

        target_names = random.sample(stars.keys(), 5)

        # Pick 5 random stars as targets
        targets = [katpoint.Target("%s, star" % t) for t in
                                                target_names]

        # Set up varying scans
        scans = [('slew', 1, targets[0]), ('track', 3, targets[0]),
                 ('slew', 2, targets[1]), ('track', 5, targets[1]),
                 ('slew', 1, targets[2]), ('track', 8, targets[2]),
                 ('slew', 2, targets[3]), ('track', 9, targets[3]),
                 ('slew', 1, targets[4]), ('track', 10, targets[4])]

        # Create Mock dataset and wrap it in a KatdalAdapter
        ds = MockDataSet(timestamps=DEFAULT_TIMESTAMPS,
                         subarrays=DEFAULT_SUBARRAYS,
                         spws=spws,
                         dumps=scans)

        # Create a FAKE object
        FAKE = object()

        # Test that metadata agrees
        for k, v in six.iteritems(DEFAULT_METADATA):
            self.assertEqual(v, getattr(ds, k, FAKE))

        # Setup the katdal selection, convert it to a string
        # accepted by our command line parser function, which
        # converts it back to a dict.
        select = {
            'scans': 'track',
            'corrprods': 'cross',
            'targets': target_names,
            'pol': 'HH,VV',
            'channels': slice(0, nchan),}
        assign_str = '; '.join('%s=%s' % (k,repr(v)) for k,v in select.items())
        select = parse_python_assigns(assign_str)

        # Do some baseline averaging for funsies
        uvblavg_params = parse_python_assigns("FOV=1.0; avgFreq=1; "
                                              "chAvg=8; maxInt=2.0")

        # Run with imaging defaults
        mfimage_params = {}

        # Create and run the pipeline
        pipeline = ContinuumPipeline(ds, TelescopeState(),
            katdal_select=select,
            uvblavg_params=uvblavg_params,
            mfimage_params=mfimage_params)

        pipeline.execute()
