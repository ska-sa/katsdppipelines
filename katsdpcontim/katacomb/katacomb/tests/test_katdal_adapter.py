from pprint import pprint
import random
import unittest

from ephem.stars import stars
import katpoint
import numpy as np
import six

from katacomb.mock_dataset import (MockDataSet,
                    ANTENNA_DESCRIPTIONS,
                    DEFAULT_METADATA,
                    DEFAULT_SUBARRAYS,
                    DEFAULT_TIMESTAMPS)

from katacomb import (AIPSPath,
                    KatdalAdapter,
                    obit_context,
                    uv_factory,
                    uv_export)

from katacomb.tests.test_aips_path import file_cleaner
from katacomb.util import parse_python_assigns

import UV

class TestKatdalAdapter(unittest.TestCase):

    def test_katdal_adapter(self):
        """
        Test export to katdal adapter
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

        subarrays = [{'antenna' : ANTENNA_DESCRIPTIONS[:4]}]

        target_names = random.sample(stars.keys(), 5)

        # Pick 5 random stars as targets
        targets = [katpoint.Target("%s, star" % t) for t in
                                                target_names]

        # Slew for 1 dump and track for 4 on each target
        slew_track_dumps = (('slew', 1), ('track', 4))
        scans = [(e, nd, t) for t in targets
                        for e, nd in slew_track_dumps]

        # Create Mock dataset and wrap it in a KatdalAdapter
        ds = MockDataSet(timestamps=DEFAULT_TIMESTAMPS,
                         subarrays=DEFAULT_SUBARRAYS,
                         spws=spws,
                         dumps=scans)

        KA = KatdalAdapter(ds)

        # Create a FAKE object
        FAKE = object()

        # Test that metadata agrees
        for k, v in six.iteritems(DEFAULT_METADATA):
            self.assertEqual(v, getattr(KA, k, FAKE))

        # Setup the katdal selection, convert it to a string
        # accepted by our command line parser function, which
        # converts it back to a dict.
        select = {
            'scans': 'track',
            'targets': target_names,
            'pol': 'HH,VV',
            'channels': slice(0, nchan),}
        assign_str = '; '.join('%s=%s' % (k,repr(v)) for k,v in select.items())
        select = parse_python_assigns(assign_str)

        # Perform the katdal selection
        KA.select(**select)

        # Get data shape after selection
        kat_ndumps, kat_nchans, kat_ncorrprods = KA.shape

        uv_file_path = AIPSPath('test', 1, 'test', 1)

        with obit_context(), file_cleaner([uv_file_path]):
            # Perform export of katdal selection
            with uv_factory(aips_path=uv_file_path, mode="w",
                            nvispio=nvispio,
                            table_cmds=KA.default_table_cmds(),
                            desc=KA.uv_descriptor()) as uvf:

                uv_export(KA, uvf)

            nvispio = 1

            # Now read from the AIPS UV file and sanity check
            with uv_factory(aips_path=uv_file_path,
                            mode="r",
                            nvispio=nvispio) as uvf:

                uv_desc = uvf.Desc.Dict
                inaxes = list(reversed(uv_desc['inaxes'][:6]))
                naips_vis = uv_desc['nvis']
                summed_vis = 0

                # Number of random parameters
                nrparm = uv_desc['nrparm']
                # Length of visibility buffer record
                lrec = uv_desc['lrec']

                # Random parameter indices
                ilocu = uv_desc['ilocu']     # U
                ilocv = uv_desc['ilocv']     # V
                ilocw = uv_desc['ilocw']     # W
                iloct = uv_desc['iloct']     # time
                ilocb = uv_desc['ilocb']     # baseline id
                ilocsu = uv_desc['ilocsu']   # source id

                # Sanity check the UV descriptor inaxes
                uv_nra, uv_ndec, uv_nif, uv_nchans, uv_nstokes, uv_viscomp = inaxes

                self.assertEqual(uv_nchans, kat_nchans,
                                "Number of AIPS and katdal channels differ")
                self.assertEqual(uv_viscomp, 3,
                                "Number of AIPS visibility components")
                self.assertEqual(uv_nra, 1,
                                "RA should be 1")
                self.assertEqual(uv_ndec, 1,
                                "DEC should be 1")
                self.assertEqual(uv_nif, 1,
                                "NIF should be 1")

                # Compare AIPS and katdal scans
                aips_scans = uvf.tables["AIPS NX"].rows
                katdal_scans = list(KA.scans())

                # Must have same number of scans
                self.assertEqual(len(aips_scans), len(katdal_scans))

                # Iterate through the katdal scans
                for i, (si, state, target) in enumerate(KA.scans()):
                    self.assertTrue(state in select['scans'])

                    kat_ndumps, kat_nchans, kat_ncorrprods = KA.shape

                    # Get the expected timestamps for this scan
                    timestamps = KA.uv_timestamps[:].astype(np.float32)

                    # Was is the expected source ID?
                    expected_source = np.float32(target['ID. NO.'][0])

                    # Work out start, end and length of the scan
                    # in visibilities
                    aips_scan = aips_scans[i]
                    scan_source = aips_scan['SOURCE ID'][0]
                    start_vis = aips_scan['START VIS'][0]
                    last_vis = aips_scan['END VIS'][0]
                    naips_scan_vis = last_vis - start_vis + 1

                    summed_vis += naips_scan_vis

                    # Each AIPS visibility has dimension [1,1,1,nchan,nstokes,3]
                    # and one exists for each timestep and baseline
                    # Ensure that the number of visibilities equals
                    # number of dumps times number of baselines
                    self.assertEqual(naips_scan_vis,
                        kat_ndumps*kat_ncorrprods//uv_nstokes,
                        'Mismatch in number of visibilities in scan %d' % si)

                    # Check each visibility in the scan, compare read data
                    # with expected data
                    for firstVis in range(start_vis, last_vis+1, nvispio):
                        numVisBuff = min(last_vis+1-firstVis, nvispio)

                        desc = uvf.Desc.Dict
                        desc.update(numVisBuff=numVisBuff)
                        uvf.Desc.Dict = desc

                        # Read a visibility
                        uvf.Read(firstVis=firstVis)
                        buf = uvf.np_visbuf

                        times = np.unique(buf[iloct:lrec*numVisBuff:lrec])
                        sources = np.unique(buf[ilocsu:lrec*numVisBuff:lrec])

                        # Check that times are in this scan's timestamps
                        self.assertTrue(np.isin(times, timestamps))

                        # Check that we're dealing with the same source
                        self.assertEqual(sources.size, 1)
                        self.assertEqual(sources[0], expected_source)

                # Check that we read the expected number of visibilities
                self.assertEqual(summed_vis, naips_vis)

if __name__ == "__main__":
    unittest.main()

