import os
import shutil
import unittest
from functools import partial


import numpy as np
import six
from scipy import constants

import katpoint
from katsdptelstate import TelescopeState

from katacomb import pipeline_factory, obit_context
from katacomb.aips_export import (fit_flux_model,
                                  obit_flux_model,
                                  _massage_gains,
                                  AIPS_NAN, NP_NAN)
from katacomb.aips_path import AIPSPath
from katacomb.continuum_pipeline import IMG_CLASS
from katacomb.katdal_adapter import CORR_ID_MAP
from katacomb.mock_dataset import (MockDataSet,
                                   DEFAULT_METADATA,
                                   DEFAULT_SUBARRAYS,
                                   DEFAULT_TIMESTAMPS,
                                   ANTENNA_DESCRIPTIONS)
from katacomb.util import (parse_python_assigns,
                           setup_aips_disks)
from katacomb.uv_facade import uv_factory
import katacomb.configuration as kc

CLOBBER = set(['scans', 'avgscans', 'merge', 'clean', 'mfimage'])


def vis(dataset, sources):
    """Compute visibilities for a list of katpoint Targets
    with flux density models. These are can be passed to
    MockDataSet via sources.
    """
    pc = dataset.catalogue.targets[0]
    out_vis = np.zeros(dataset.shape, dtype=np.complex64)
    wl = constants.c / dataset.freqs
    # uvw in wavelengths for each channel
    uvw = np.array([dataset.u, dataset.v, dataset.w])
    uvw_wl = uvw[:, :, np.newaxis, :] / wl[np.newaxis, np.newaxis, :, np.newaxis]
    for target in sources:
        flux_freq = target.flux_density(dataset.freqs/1.e6)
        lmn = np.array(pc.lmn(*target.radec()))
        n = lmn[2]
        lmn[2] -= 1.
        # uvw_wl has shape (uvw, ntimes, nchannels, nbl), move uvw to
        # the last axis before np.dot
        exponent = 2j * np.pi * np.dot(np.moveaxis(uvw_wl, 0, -1), lmn)
        out_vis += flux_freq[np.newaxis, :, np.newaxis] * np.exp(exponent) / n
    return out_vis


def weights(dataset):
    return np.ones(dataset.shape, dtype=np.float32)


def flags(dataset):
    return np.zeros(dataset.shape, dtype=np.bool)


class TestOfflinePipeline(unittest.TestCase):

    def test_offline_pipeline(self):
        """
        Tests that a run of the offline continuum pipeline executes.
        """

        nchan = 16

        spws = [{
            'centre_freq': .856e9 + .856e9 / 2.,
            'num_chans': nchan,
            'channel_width': .856e9 / nchan,
            'sideband': 1,
            'band': 'L',
        }]

        target_name = 'Beckmesser'

        # Construct a target
        target = katpoint.Target("%s, radec, 100.0, -35.0" % target_name)

        # Set up a track
        scans = [('track', 3, target), ('track', 3, target)]

        # Create Mock dataset and wrap it in a KatdalAdapter
        ds = MockDataSet(timestamps=DEFAULT_TIMESTAMPS,
                         subarrays=DEFAULT_SUBARRAYS,
                         spws=spws,
                         dumps=scans)

        # Setup the katdal selection, convert it to a string
        # accepted by our command line parser function, which
        # converts it back to a dict.
        select = {'corrprods': 'cross',
                  'pol': 'HH,VV'}

        # Baseline averaging defaults
        uvblavg_params = parse_python_assigns("FOV=1.0; avgFreq=0; "
                                              "chAvg=1; maxInt=2.0")

        # Run with imaging defaults
        mfimage_params = {'doGPU': False, 'maxFBW': 0.25}

        # Dummy CB_ID and Product ID and temp fits and aips disks
        fd = kc.get_config()['fitsdirs']
        fd += [(None, os.path.join(os.sep, 'tmp', 'FITS'))]
        kc.set_config(output_id='OID', cb_id='CBID', fitsdirs=fd)

        setup_aips_disks()

        # Create and run the pipeline
        pipeline = pipeline_factory('offline', ds,
                                     katdal_select=select,
                                     uvblavg_params=uvblavg_params,
                                     mfimage_params=mfimage_params,
                                     clobber=CLOBBER.difference({'merge'}))

        pipeline.execute()

        # Check that output FITS files exist and have the right names
        # Now check for files
        cfg = kc.get_config()
        cb_id = cfg['cb_id']
        out_id = cfg['output_id']
        fits_area = cfg['fitsdirs'][-1][1]

        out_strings = [cb_id, out_id, target_name, IMG_CLASS]
        filename = '_'.join(filter(None, out_strings)) + '.fits'
        filepath = os.path.join(fits_area, filename)
        assert os.path.isfile(filepath)

        # Remove the tmp/FITS dir
        shutil.rmtree(fits_area)

        ds = MockDataSet(timestamps=DEFAULT_TIMESTAMPS,
                         subarrays=DEFAULT_SUBARRAYS,
                         spws=spws,
                         dumps=scans)

        setup_aips_disks()

        # Create and run the pipeline (Reusing the previous data)
        pipeline = pipeline_factory('offline', ds,
                                    katdal_select=select,
                                    uvblavg_params=uvblavg_params,
                                    mfimage_params=mfimage_params,
                                    reuse=True,
                                    clobber=CLOBBER)

        pipeline.execute()

        assert os.path.isfile(filepath)

        # Remove FITS temporary area
        shutil.rmtree(fits_area)


class TestOnlinePipeline(unittest.TestCase):

    def test_online_pipeline(self):
        """
        Tests that a run of the online continuum pipeline executes.
        """

        nchan = 16

        spws = [{
            'centre_freq': .856e9 + .856e9 / 2.,
            'num_chans': nchan,
            'channel_width': .856e9 / nchan,
            'sideband': 1,
            'band': 'L',
        }]

        target_names = ['Gunther Lord of the Gibichungs',
                        'Gunther\Lord/of%the Gibichungs',
                        'Gutrune', 'Hagen']

        # Construct 4 targets
        targets = [katpoint.Target("%s, radec, 100.0, -35.0" % t) for t in target_names]

        # Construct a 5th target with repeated name
        targets += [katpoint.Target("%s | %s, radec, 100.0, -35.0"
                    % (target_names[2], target_names[3]))]

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
            'pol': 'HH,VV',
            'channels': slice(0, nchan), }
        assign_str = '; '.join('%s=%s' % (k, repr(v)) for k, v in select.items())
        select = parse_python_assigns(assign_str)

        # Do some baseline averaging for funsies
        uvblavg_params = parse_python_assigns("FOV=1.0; avgFreq=1; "
                                              "chAvg=8; maxInt=2.0")

        # Run with imaging defaults
        mfimage_params = {'doGPU': False}

        # Dummy CB_ID and Product ID and temp fits disk
        fd = kc.get_config()['fitsdirs']
        fd += [(None, '/tmp/FITS')]
        kc.set_config(output_id='OID', cb_id='CBID', fitsdirs=fd)

        setup_aips_disks()

        # Create and run the pipeline
        pipeline = pipeline_factory('online', ds, TelescopeState(),
                                     katdal_select=select,
                                     uvblavg_params=uvblavg_params,
                                     mfimage_params=mfimage_params)

        pipeline.execute()

        # Check that output FITS files exist and have the right names
        # Expected target name in file output for the list of targets
        # constructed via normalise_target_name
        sanitised_target_names = ['Gunther_Lord_of_the_Gibichungs',
                                  'Gunther_Lord_of_the_Gibichungs_1',
                                  'Gutrune', 'Hagen', 'Gutrune_1']

        # Now check for files
        cfg = kc.get_config()
        cb_id = cfg['cb_id']
        out_id = cfg['output_id']
        fits_area = cfg['fitsdirs'][-1][1]

        for otarg in sanitised_target_names:
            out_strings = [cb_id, out_id, otarg, IMG_CLASS]
            filename = '_'.join(filter(None, out_strings)) + '.fits'
            filepath = os.path.join(fits_area, filename)
            assert os.path.isfile(filepath)

        # Remove the tmp/FITS dir
        shutil.rmtree(fits_area)

    def test_cc_fitting(self):
        """Check CC fitting with increasing order and conversion to katpoint models
        """
        input_freqs = np.linspace(100.e6, 500.e6, 10)
        lnunu0 = np.log(input_freqs / input_freqs[5])
        input_cc_tab = np.ones((4, 10), dtype=np.float)
        input_cc_tab[0] = obit_flux_model(lnunu0, 10.)
        input_cc_tab[1] = obit_flux_model(lnunu0, -10., -0.7)
        input_cc_tab[2] = obit_flux_model(lnunu0, 2., 0.1, 0.01)
        input_cc_tab[3] = obit_flux_model(lnunu0, -0.5, -0.5, 0.01, 0.05)
        input_sigma = np.ones(10, dtype=np.float) * 0.1
        for order in range(4):
            cc_tab = input_cc_tab[order]
            kp_model = fit_flux_model(input_freqs, cc_tab, input_freqs[0],
                                      input_sigma, cc_tab[0], order=order)
            # Check the flux densities of the input and fitted models
            kp_flux_model = katpoint.FluxDensityModel(100., 500., kp_model)
            np.testing.assert_allclose(kp_flux_model.flux_density(input_freqs/1.e6), cc_tab)
        # Check a target with zero flux
        cc_tab = np.zeros(10)
        kp_model = fit_flux_model(input_freqs, cc_tab, input_freqs[0],
                                  input_sigma, cc_tab[0], order=0)
        kp_flux_model = katpoint.FluxDensityModel(100., 500., kp_model)
        np.testing.assert_array_equal(kp_flux_model.flux_density(input_freqs/1.e6), cc_tab)

    def test_cc_export(self):
        """Check CC models returned by MFImage
        """
        nchan = 128

        spws = [{'centre_freq': .856e9 + .856e9 / 2.,
                 'num_chans': nchan,
                 'channel_width': .856e9 / nchan,
                 'sideband': 1,
                 'band': 'L'}]

        katdal_select = {'pol': 'HH,VV', 'scans': 'track',
                         'corrprods': 'cross'}
        uvblavg_params = {'FOV': 0.2, 'avgFreq': 0,
                          'chAvg': 1, 'maxInt': 2.0}

        cat = katpoint.Catalogue()
        cat.add(katpoint.Target("Amfortas, radec, 0.0, -90.0, (856. 1712. 1. 0. 0.)"))
        cat.add(katpoint.Target("Klingsor, radec, 0.0, 0.0, (856. 1712. 2. -0.7 0.1)"))
        cat.add(katpoint.Target("Kundry, radec, 100.0, -35.0, (856. 1712. -1.0 1. -0.1)"))

        ts = TelescopeState()

        # Set up a scratch space in /tmp
        fd = kc.get_config()['fitsdirs']
        fd += [(None, '/tmp/FITS')]
        kc.set_config(cb_id='CBID', fitsdirs=fd)

        setup_aips_disks()

        # Point sources with various flux models
        for targ in cat:
            scans = [('track', 5, targ)]
            ds = MockDataSet(timestamps={'start_time': 1.0, 'dump_period': 4.0},
                             subarrays=DEFAULT_SUBARRAYS,
                             spws=spws,
                             dumps=scans,
                             vis=partial(vis, sources=[targ]),
                             weights=weights,
                             flags=flags)

            # 100 clean components
            mfimage_params = {'Niter': 100, 'maxFBW': 0.05,
                              'FOV': 0.1, 'xCells': 5.,
                              'yCells': 5., 'doGPU': False}

            pipeline = pipeline_factory('online', ds, ts, katdal_select=katdal_select,
                                         uvblavg_params=uvblavg_params,
                                         mfimage_params=mfimage_params)
            pipeline.execute()

            # Get the fitted CCs from telstate
            fit_cc = ts.get('target0_clean_components')
            ts.delete('target0_clean_components')

            all_ccs = katpoint.Catalogue(fit_cc['components'])
            # Should have one merged and fitted component
            self.assertEqual(len(all_ccs), 1)

            cc = all_ccs.targets[0]
            out_fluxmodel = cc.flux_model
            in_fluxmodel = targ.flux_model

            # Check the flux densities of the flux model in the fitted CC's
            test_freqs = np.linspace(out_fluxmodel.min_freq_MHz, out_fluxmodel.max_freq_MHz, 5)
            in_flux = in_fluxmodel.flux_density(test_freqs)
            out_flux = out_fluxmodel.flux_density(test_freqs)
            np.testing.assert_allclose(out_flux, in_flux, rtol=1.e-3)

        # A field with some off axis sources to check positions
        offax_cat = katpoint.Catalogue()
        offax_cat.add(katpoint.Target("Titurel, radec, 100.1, -35.05, (856. 1712. 1.1 0. 0.)"))
        offax_cat.add(katpoint.Target("Gurmenanz, radec, 99.9, -34.95, (856. 1712. 1. 0. 0.)"))

        scans = [('track', 5, cat.targets[2])]
        ds = MockDataSet(timestamps={'start_time': 1.0, 'dump_period': 4.0},
                         subarrays=DEFAULT_SUBARRAYS,
                         spws=spws,
                         dumps=scans,
                         vis=partial(vis, sources=offax_cat),
                         weights=weights,
                         flags=flags)

        # Small number of CC's and high gain (not checking flux model)
        mfimage_params['Niter'] = 4
        mfimage_params['FOV'] = 0.2
        mfimage_params['Gain'] = 0.5
        mfimage_params['Robust'] = -5

        pipeline = pipeline_factory('online', ds, ts, katdal_select=katdal_select,
                                     uvblavg_params=uvblavg_params,
                                     mfimage_params=mfimage_params)
        pipeline.execute()
        fit_cc = ts.get('target0_clean_components')
        ts.delete('target0_clean_components')
        all_ccs = katpoint.Catalogue(fit_cc['components'])
        # We should have 2 merged clean components for two source positions
        self.assertEqual(len(all_ccs), 2)

        # Check the positions of the clean components
        # These will be ordered by decreasing flux density of the inputs
        # Position should be accurate to within a 5" pixel
        delta_dec = np.deg2rad(5./3600.)
        for model, cc in zip(offax_cat.targets, all_ccs.targets):
            delta_ra = delta_dec/np.cos(model.radec()[1])
            self.assertAlmostEqual(cc.radec()[0], model.radec()[0], delta=delta_ra)
            self.assertAlmostEqual(cc.radec()[1], model.radec()[1], delta=delta_dec)

        # Empty the scratch space
        shutil.rmtree(fd[-1][1])

    def test_SN_to_telstate(self):
        """Check conversion of SN table"""

        def construct_SN_desc(nif, rows):
            dummy_SN = {}
            dummy_SN["AIPS SN"] = {"attach": {'version': 1, 'numIF': nif, 'numPol': 2},
                                   "rows": rows,
                                   "write": True}
            return dummy_SN

        def construct_SN_default_rows(timestamps, ants, nif):
            """
            Construct list of ants dicts for each
            timestamp with REAL, IMAG, WEIGHT = 1.0
            """
            default_nif = [1.0] * nif
            rows = []
            for ts in timestamps:
                rows += [{'TIME': [ts],
                          'TIME INTERVAL': [0.1],
                          'ANTENNA NO.': [antn],
                          'REAL1': default_nif,
                          'REAL2': default_nif,
                          'IMAG1': default_nif,
                          'IMAG2': default_nif,
                          'WEIGHT 1': default_nif,
                          'WEIGHT 2': default_nif}
                         for antn in ants]
            return rows

        ant_ordering = ['m000', 'm001', 'm002', 'm003', 'm004', 'm005']
        # Make a dummy AIPS UV and attach an SN Table
        with obit_context():
            nif = 1
            ap = AIPSPath("Flosshilde")
            rows = construct_SN_default_rows([0.5], [1, 2, 3, 4, 5, 6], nif)
            # Modify some gains
            rows[1]['REAL1'] = [AIPS_NAN]
            rows[2]['REAL1'] = rows[2]['REAL2'] = [AIPS_NAN]
            rows[3]['REAL1'], rows[3]['WEIGHT 1'] = ([AIPS_NAN], [-1.0])
            rows[4]['WEIGHT 1'] = rows[4]['WEIGHT 2'] = [-1.0]
            rows[5]['REAL1'] = rows[5]['REAL2'] = [AIPS_NAN]
            rows[5]['IMAG1'] = rows[5]['IMAG2'] = [AIPS_NAN]
            rows[5]['WEIGHT 1'] = rows[5]['WEIGHT 2'] = [-1.0]
            sn_tab_desc = construct_SN_desc(nif, rows)
            uvf = uv_factory(aips_path=ap, mode="w", table_cmds=sn_tab_desc)
            sntab = uvf.tables["AIPS SN"]
            ts, result = _massage_gains(sntab, ant_ordering)
            # Do the gains and timestamps have the right values/shapes
            self.assertEqual(ts, [0.5])
            self.assertEqual(len(ts), len(result))
            self.assertEqual(result[0].shape, (nif, 2, len(ant_ordering)))
            expected_result = np.full((1, 2, len(ant_ordering)), 1.+1.j, dtype=np.complex64)
            expected_result[0, 0, 1] = AIPS_NAN + 1.j
            expected_result[0, :, 2] = AIPS_NAN + 1.j
            expected_result[0, 0, 3] = NP_NAN
            expected_result[0, :, 5] = NP_NAN
            np.testing.assert_array_equal(result[0], expected_result)

            # Change ntimes, nif and antennas and recheck shapes
            nif = 8
            ntimes = 5
            rows = construct_SN_default_rows(np.linspace(0., 1., ntimes), [1, 3, 5, 6], nif)
            sn_tab_desc = construct_SN_desc(nif, rows)
            uvf = uv_factory(aips_path=ap, mode="w", table_cmds=sn_tab_desc)
            sntab = uvf.tables["AIPS SN"]
            ts, result = _massage_gains(sntab, ant_ordering)
            np.testing.assert_array_equal(ts, np.linspace(0., 1., ntimes))
            self.assertEqual(len(ts), len(result))
            self.assertEqual(result[0].shape, (nif, 2, len(ant_ordering)))
            # Are the missing antennas nans?
            np.testing.assert_array_equal(result[0][:, :, [1, 3]], NP_NAN)

            # Empty SN table should return empty lists
            sn_tab_desc = construct_SN_desc(8, [])
            uvf = uv_factory(aips_path=ap, mode="w", table_cmds=sn_tab_desc)
            sntab = uvf.tables["AIPS SN"]
            ts, result = _massage_gains(sntab, ant_ordering)
            self.assertEqual(ts, [])
            self.assertEqual(result, [])

    def test_gains_export(self):
        """Check l2 export to telstate"""
        nchan = 128
        nif = 4
        dump_period = 1.0
        centre_freq = 1200.e6
        bandwidth = 100.e6
        solPint = dump_period / 2.
        solAint = dump_period
        AP_telstate = 'product_GAMP_PHASE'
        P_telstate = 'product_GPHASE'

        spws = [{'centre_freq': centre_freq,
                 'num_chans': nchan,
                 'channel_width': bandwidth / nchan,
                 'sideband': 1,
                 'band': 'L'}]
        ka_select = {'pol': 'HH,VV', 'scans': 'track',
                     'corrprods': 'cross', 'nif': nif}
        uvblavg_params = {'maxFact': 1.0, 'avgFreq': 0,
                          'FOV': 100.0, 'maxInt': 1.e-6}
        mfimage_params = {'Niter': 50, 'FOV': 0.1,
                          'xCells': 5., 'yCells': 5.,
                          'doGPU': False, 'Robust': -1.5,
                          'minFluxPSC': 0.1, 'solPInt': solPint / 60.,
                          'solPMode': 'P', 'minFluxASC': 0.1,
                          'solAInt': solAint / 60., 'maxFBW': 0.02}

        # Simulate a '10Jy' source at the phase center
        cat = katpoint.Catalogue()
        cat.add(katpoint.Target("Alberich, radec, 20.0, -30.0, (856. 1712. 1. 0. 0.)"))

        telstate = TelescopeState()

        # Set up a scratch space in /tmp
        fd = kc.get_config()['fitsdirs']
        fd += [(None, '/tmp/FITS')]
        kc.set_config(cb_id='CBID', fitsdirs=fd)
        setup_aips_disks()

        scan = [('track', 4, cat.targets[0])]

        # Construct a simulated dataset with our
        # point source at the centre of the field
        ds = MockDataSet(timestamps={'start_time': 0.0, 'dump_period': dump_period},
                         subarrays=DEFAULT_SUBARRAYS,
                         spws=spws,
                         dumps=scan,
                         vis=partial(vis, sources=cat),
                         weights=weights,
                         flags=flags)

        # Try one round of phase only self-cal & Amp+Phase self-cal
        mfimage_params['maxPSCLoop'] = 1
        mfimage_params['maxASCLoop'] = 1

        # Run the pipeline
        pipeline = pipeline_factory('online', ds, telstate, katdal_select=ka_select,
                                     uvblavg_params=uvblavg_params,
                                     mfimage_params=mfimage_params)
        pipeline.execute()

        ts = telstate.view('selfcal')
        # Check what we have in telstate agrees with what we put in
        self.assertEqual(len(ts['antlist']), len(ANTENNA_DESCRIPTIONS))
        self.assertEqual(ts['bandwidth'], bandwidth)
        self.assertEqual(ts['n_chans'], nif)
        pol_ordering = [pol[0] for pol in sorted(CORR_ID_MAP, key=CORR_ID_MAP.get)
                        if pol[0] == pol[1]]
        self.assertEqual(ts['pol_ordering'], pol_ordering)
        if_width = bandwidth / nif
        center_if = nif // 2
        start_freq = centre_freq - (bandwidth / 2.)
        self.assertEqual(ts['center_freq'], start_freq + if_width * (center_if + 0.5))

        self.assertIn(ts.join('selfcal', P_telstate), ts.keys())
        self.assertIn(ts.join('selfcal', AP_telstate), ts.keys())

        def check_gains_timestamps(gains, expect_timestamps):
            timestamps = []
            for gain, timestamp in gains:
                np.testing.assert_array_almost_equal(np.abs(gain), 1.0, decimal=3)
                np.testing.assert_array_almost_equal(np.angle(gain), 0.0)
                timestamps.append(timestamp)
            np.testing.assert_array_almost_equal(timestamps, expect_timestamps, decimal=1)

        # Check phase-only gains and timestamps
        P_times = np.arange(solPint, ds.end_time.secs, 2. * solPint)
        check_gains_timestamps(ts.get_range(P_telstate, st=0), P_times)
        # Check Amp+Phase gains
        AP_times = np.arange(solAint, ds.end_time.secs, 2. * solAint)
        check_gains_timestamps(ts.get_range(AP_telstate, st=0), AP_times)

        # Check with no Amp+Phase self-cal
        mfimage_params['maxASCLoop'] = 0
        telstate.clear()
        pipeline = pipeline_factory('online', ds, telstate, katdal_select=ka_select,
                                     uvblavg_params=uvblavg_params,
                                     mfimage_params=mfimage_params)
        pipeline.execute()
        self.assertIn(telstate.join('selfcal', P_telstate), ts.keys())
        self.assertNotIn(telstate.join('selfcal', AP_telstate), ts.keys())

        # Check with no self-cal
        mfimage_params['maxPSCLoop'] = 0
        telstate.clear()
        pipeline = pipeline_factory('online', ds, telstate, katdal_select=ka_select,
                                     uvblavg_params=uvblavg_params,
                                     mfimage_params=mfimage_params)
        pipeline.execute()
        self.assertNotIn(telstate.join('selfcal', P_telstate), ts.keys())
        self.assertNotIn(telstate.join('selfcal', AP_telstate), ts.keys())

        # Cleanup workspace
        shutil.rmtree(fd[-1][1])
