import unittest
import random

from ephem.stars import stars
import katpoint
import numpy as np

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

class TestAipsFacades(unittest.TestCase):

    def test_uv_facade_read_write(self):
        """
        Test basic reads and writes the AIPS UV Facade
        """
        nvis = 577      # Read/write this many visibilities, total
        nvispio = 20    # Read/write this many visibilities per IO op
        uv_file_path = AIPSPath('test', 1, 'test', 1)

        # Set up the spectral window
        nchan = 4

        spws = [{
            'centre_freq' : .856e9 + .856e9 / 2.,
            'num_chans' : nchan,
            'channel_width' : .856e9 / nchan,
            'sideband' : 1,
            'band' : 'L',
        }]

        # Use first four antenna to create the subarray
        subarrays = [{'antenna' : ANTENNA_DESCRIPTIONS[:4]}]

        # Pick 5 random stars as targets
        targets = [katpoint.Target("%s, star" % t) for t in
                                random.sample(stars.keys(), 5)]

        # track for 5 on each target
        slew_track_dumps = (('track', 5),)
        scans = [(e, nd, t) for t in targets
                        for e, nd in slew_track_dumps]

        # Create Mock dataset and wrap it in a KatdalAdapter
        KA = KatdalAdapter(MockDataSet(timestamps=DEFAULT_TIMESTAMPS,
                subarrays=subarrays, spws=spws, dumps=scans))

        with obit_context(), file_cleaner(uv_file_path):
            # Create the UV file
            with uv_factory(aips_path=uv_file_path,
                            mode="w",
                            nvispio=nvispio,
                            table_cmds=KA.default_table_cmds(),
                            desc=KA.uv_descriptor()) as uvf:

                uv_desc = uvf.Desc.Dict

                # Number of random parameters
                nrparm = uv_desc['nrparm']
                # Length of visibility buffer record
                lrec = uv_desc['lrec']
                # Random parameter indices
                iloct = uv_desc['iloct']     # time

                # Write out visibilities, putting sequential values
                # in the time random parameter
                for firstVis in range(1, nvis+1, nvispio):
                    numVisBuff = min(nvis+1-firstVis, nvispio)

                    uv_desc = uvf.Desc.Dict
                    uv_desc['numVisBuff'] = numVisBuff
                    uvf.Desc.Dict = uv_desc

                    times = np.arange(firstVis, firstVis+numVisBuff, dtype=np.float32)

                    buf = uvf.np_visbuf
                    buf[iloct:lrec*numVisBuff:lrec] = times
                    uvf.Write(firstVis=firstVis)

            # Now re-open in readonly mode and test
            # that we get the same sequential values out
            with uv_factory(aips_path=uv_file_path,
                            mode="r",
                            nvispio=nvispio) as uvf:

                uv_desc = uvf.Desc.Dict

                # Number of random parameters
                nrparm = uv_desc['nrparm']
                # Length of visibility buffer record
                lrec = uv_desc['lrec']
                nvis = uv_desc['nvis']
                # Random parameter indices
                iloct = uv_desc['iloct']     # time

                for firstVis in range(1, nvis+1, nvispio):
                    numVisBuff = min(nvis+1-firstVis, nvispio)

                    uv_desc = uvf.Desc.Dict
                    uv_desc['numVisBuff'] = numVisBuff
                    uvf.Desc.Dict = uv_desc

                    uvf.Read(firstVis=firstVis)
                    buf = uvf.np_visbuf

                    times = np.arange(firstVis, firstVis+numVisBuff, dtype=np.float32)
                    buf_times = buf[iloct:lrec*numVisBuff:lrec]
                    self.assertTrue(np.all(times == buf_times))

if __name__ == "__main__":
    unittest.main()
