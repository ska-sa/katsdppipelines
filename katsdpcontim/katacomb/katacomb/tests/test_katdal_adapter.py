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

class TestKatdalAdapter(unittest.TestCase):

    def test_katdal_adapter(self):
        """
        Tests writing data from a :class:`KatdalAdapter` object
        to an AIPS UV file.

        Then, the AIPS UV file is re-opened and visibility
        data is read per scan and compared with the data
        from each scan of the :class:`KatdalAdapter`
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
            'corrprods': 'cross',
            'targets': target_names,
            'pol': 'HH,VV',
            'channels': slice(0, nchan),}
        assign_str = '; '.join('%s=%s' % (k,repr(v)) for k,v in select.items())
        select = parse_python_assigns(assign_str)

        # Perform the katdal selection
        KA.select(**select)

        # Obtain correlator products and produce argsorts that will
        # order by (a1, a2, stokes)
        cp = KA.correlator_products()
        nstokes = KA.nstokes

        # Lexicographically sort correlation products on (a1, a2, cid)
        sort_fn = lambda x: (cp[x].ant1_ix, cp[x].ant2_ix, cp[x].cid)
        cp_argsort = np.asarray(sorted(range(len(cp)), key=sort_fn))
        corr_products = np.asarray([cp[i] for i in cp_argsort])

        # Use first stokes parameter index of each baseline
        bl_argsort = cp_argsort[::nstokes]

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

                def _strip_strings(aips_keywords):
                    """ AIPS string are padded, strip them """
                    return { k: v.strip()
                            if isinstance(v, six.string_types) else v
                            for k, v in aips_keywords.iteritems() }

                fq_kw =  _strip_strings(uvf.tables["AIPS FQ"].keywords)
                src_kw = _strip_strings(uvf.tables["AIPS SU"].keywords)
                ant_kw = _strip_strings(uvf.tables["AIPS AN"].keywords)

                # Check that the subset of keywords generated
                # by the katdal adapter match those read from the AIPS table
                self.assertDictContainsSubset(KA.uv_spw_keywords, fq_kw)
                self.assertDictContainsSubset(KA.uv_source_keywords, src_kw)
                self.assertDictContainsSubset(KA.uv_antenna_keywords, ant_kw)

                def _strip_metadata(aips_table_rows):
                    """
                    Strip out ``Numfields``, ``_status``, ``Table name``
                    fields from each row entry
                    """
                    STRIP = { 'NumFields', '_status', 'Table name' }
                    return [{k: v for k, v in d.items()
                                if k not in STRIP}
                                        for d in aips_table_rows]

                # Check that frequency, source and antenna rows
                # are correctly exported
                fq_rows = _strip_metadata(uvf.tables["AIPS FQ"].rows)
                src_rows = _strip_metadata(uvf.tables["AIPS SU"].rows)
                ant_rows = _strip_metadata(uvf.tables["AIPS AN"].rows)

                self.assertEqual(fq_rows, KA.uv_spw_rows)
                self.assertEqual(src_rows, KA.uv_source_rows)
                self.assertEqual(ant_rows, KA.uv_antenna_rows)

                uv_desc = uvf.Desc.Dict
                inaxes = tuple(reversed(uv_desc['inaxes'][:6]))
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

                    # Accumulate UVW, time data from the AIPS UV file
                    # By convention uv_export's data in (ntime, nbl)
                    # ordering, so we assume that the AIPS UV data
                    # is ordered the same way
                    u_data = []
                    v_data = []
                    w_data = []
                    time_data = []
                    vis_data = []

                    # For each visibility in the scan, read data and
                    # compare with katdal observation data
                    for firstVis in range(start_vis, last_vis+1, nvispio):
                        # Determine number of visibilities to read
                        numVisBuff = min(last_vis+1-firstVis, nvispio)

                        desc = uvf.Desc.Dict
                        desc.update(numVisBuff=numVisBuff)
                        uvf.Desc.Dict = desc

                        # Read a visibility
                        uvf.Read(firstVis=firstVis)
                        buf = uvf.np_visbuf

                        # Must copy because buf data will change with each read
                        u_data.append(buf[ilocu:lrec*numVisBuff:lrec].copy())
                        v_data.append(buf[ilocv:lrec*numVisBuff:lrec].copy())
                        w_data.append(buf[ilocw:lrec*numVisBuff:lrec].copy())
                        time_data.append(buf[iloct:lrec*numVisBuff:lrec].copy())

                        # Introduce an extra dimension to inaxes
                        # to make later stack work
                        extra_inaxes = (1,) + inaxes

                        for i in range(numVisBuff):
                            base = nrparm + i*lrec
                            data = buf[base:base+lrec-nrparm].copy()
                            data = data.reshape(inaxes)
                            vis_data.append(data)

                        # Check that we're dealing with the same source
                        # within the scan
                        sources = buf[ilocsu:lrec*numVisBuff:lrec].copy()
                        self.assertEqual(sources, expected_source)

                    # Ensure katdal timestamps match AIPS UV file timestamps
                    # and that there are exactly number of baseline counts
                    # for each one
                    times, time_counts = np.unique(time_data, return_counts=True)
                    timestamps = KA.uv_timestamps[:].astype(np.float32)
                    self.assertTrue(np.all(times == timestamps))
                    self.assertTrue(np.all(time_counts == len(bl_argsort)))

                    # Flatten AIPS UVW data, there'll be (ntime*nbl) values
                    u_data = np.concatenate(u_data).ravel()
                    v_data = np.concatenate(v_data).ravel()
                    w_data = np.concatenate(w_data).ravel()

                    # uv_u will have shape (ntime, ncorrprods)
                    # Select katdal stokes 0 UVW coordinates and flatten
                    uv_u = KA.uv_u[:,bl_argsort].astype(np.float32).ravel()
                    uv_v = KA.uv_v[:,bl_argsort].astype(np.float32).ravel()
                    uv_w = KA.uv_w[:,bl_argsort].astype(np.float32).ravel()

                    # Confirm UVW coordinate equality
                    self.assertTrue(np.all(uv_u == u_data))
                    self.assertTrue(np.all(uv_v == v_data))
                    self.assertTrue(np.all(uv_w == w_data))

                    # Number of baselines
                    nbl = len(bl_argsort)

                    # Now compare visibility data

                    # Stacking produces
                    # (ntime*nbl, nra, ndec, nif, nchan, nstokes, 3)
                    aips_vis = np.stack(vis_data, axis=0)
                    kat_vis = KA.uv_vis[:]

                    shape = (kat_ndumps, kat_nchans, nbl, nstokes, 3)
                    # This produces (ntime, nchan, nbl, nstokes, 3)
                    kat_vis = kat_vis[:, :, cp_argsort, :].reshape(shape)


                    # (1) transpose so that we have (ntime, nbl, nchan, nstokes, 3)
                    # (2) reshape to include the full inaxes shape,
                    #     including singleton nif, ra and dec dimensions
                    kat_vis = (kat_vis.transpose(0, 2, 1, 3, 4)
                              .reshape((kat_ndumps, nbl,) + inaxes))

                    aips_vis = aips_vis.reshape((kat_ndumps, nbl) + inaxes)

                    self.assertTrue(np.all(aips_vis == kat_vis))

                # Check that we read the expected number of visibilities
                self.assertEqual(summed_vis, naips_vis)

if __name__ == "__main__":
    unittest.main()

