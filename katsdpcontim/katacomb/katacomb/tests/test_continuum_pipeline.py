import os
import random
import shutil
import unittest
from functools import partial

from ephem.stars import stars
import katpoint
import numpy as np
import six
from scipy import constants

from katsdptelstate import TelescopeState

from katacomb.mock_dataset import (MockDataSet,
                                   DEFAULT_METADATA,
                                   DEFAULT_SUBARRAYS,
                                   DEFAULT_TIMESTAMPS)

from katacomb import ContinuumPipeline
from katacomb.continuum_pipeline import IMG_CLASS
from katacomb.aips_export import fit_flux_model, obit_flux_model
from katacomb.util import (parse_python_assigns,
                           normalise_target_name,
                           setup_aips_disks)
import katacomb.configuration as kc

class TestContinuumPipeline(unittest.TestCase):

    def test_continuum_pipeline(self):
        """
        Tests that a run of the Continuum Pipeline executes.
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
        pipeline = ContinuumPipeline(ds, TelescopeState(),
                                     katdal_select=select,
                                     uvblavg_params=uvblavg_params,
                                     mfimage_params=mfimage_params)

        pipeline.execute()

        # Check that output FITS files exist and have the right names        
        # Output target names have been manipulated via normalise_target_name
        output_targets = []
        for targ in targets:
          new_name = normalise_target_name(targ.name, output_targets)
          output_targets.append(new_name)

        # Now check for files
        cfg = kc.get_config()
        cb_id = cfg['cb_id']
        out_id = cfg['output_id']
        fits_area = cfg['fitsdirs'][-1][1]

        for otarg in output_targets:
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
        def vis(dataset, sources):
            """Compute visibilities for a list of katpoint Targets
            with flux density models. These are passed via sources
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
        kc.set_config(fitsdirs=fd)

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

            pipeline = ContinuumPipeline(ds, ts, katdal_select=katdal_select,
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

        pipeline = ContinuumPipeline(ds, ts, katdal_select=katdal_select,
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