"""Tests for :py:mod:`katsdpimager.preprocess`."""

from __future__ import print_function, division
import astropy.units as units
import numpy as np
import tempfile
import logging
import os
from contextlib import closing
from nose.tools import *
import katsdpimager.parameters as parameters
import katsdpimager.polarization as polarization
import katsdpimager.preprocess as preprocess


def _empty_recarray(dtype):
    return np.rec.array(None, dtype=dtype, shape=(0,))

class BaseTestVisibilityCollector(object):
    def setup(self):
        self.image_parameters = []
        for wavelength in np.array([0.25, 0.125]) * units.m:
            self.image_parameters.append(parameters.ImageParameters(
                q_fov=1.0,
                image_oversample=5.0,
                frequency=wavelength, array=None,
                polarizations=polarization.STOKES_IQUV,
                dtype=np.float32,
                pixel_size=1.0/(4096.0*wavelength.value),
                pixels=2048))
        self.grid_parameters = parameters.GridParameters(
            antialias_width=7.0,
            oversample=8.0,
            image_oversample=4.0,
            w_slices=1,
            w_planes=128,
            max_w=400 * units.m,
            kernel_width=64)

    def check(self, collector, expected):
        reader = collector.reader()
        assert_equal(reader.num_channels, collector.num_channels)
        assert_equal(reader.num_w_slices, collector.num_w_slices)
        for channel, channel_data in enumerate(expected):
            for w_slice, slice_data in enumerate(channel_data):
                assert_equal(len(slice_data), reader.len(channel, w_slice))
                for block_size in [None, 1, 2, 100]:
                    pieces = []
                    for piece in reader.iter_slice(channel, w_slice, block_size):
                        pieces.append(piece.copy())
                    if pieces:
                        actual = np.rec.array(np.hstack(pieces))
                    else:
                        actual = np.rec.recarray(0, collector.store_dtype)
                    # np.testing functions don't seem to handle structured arrays
                    # well, so test one field at a time.
                    np.testing.assert_array_equal(actual.uv, slice_data.uv, "block_size={}".format(block_size))
                    np.testing.assert_array_equal(actual.sub_uv, slice_data.sub_uv)
                    np.testing.assert_allclose(actual.weights, slice_data.weights)
                    np.testing.assert_allclose(actual.vis, slice_data.vis, rtol=1e-5)
                    np.testing.assert_array_equal(actual.w_plane, slice_data.w_plane)

    def test_empty(self):
        with closing(self.factory(self.image_parameters, self.grid_parameters, 2)) as collector:
            pass
        self.check(collector, [[np.rec.recarray(0, collector.store_dtype)] for channel in range(2)])

    def test_simple(self):
        uvw = np.array([
            [12.1, 2.3, 4.7],
            [-3.4, 7.6, 2.5],
            [-5.2, -10.6, 7.2],
            [12.102, 2.299, 4.6],   # Should merge with the first visibility
            ], dtype=np.float32) * units.m
        weights = np.array([
            [1.3, 0.6, 1.2, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.5, 0.6, 0.7, 0.8],
            [1.1, 1.2, 1.3, 1.4],
            ], dtype=np.float32)
        baselines = np.array([
            0,
            -1,    # Auto-correlation: should be removed
            1,
            0,
            ], dtype=np.int16)
        vis = np.array([
            [0.5 - 2.3j, 0.1 + 4.2j, 0.0 - 3j, 1.5 + 0j],
            [0.5 - 2.3j, 0.1 + 4.2j, 0.0 - 3j, 1.5 + 0j],
            [1.5 + 1.3j, 1.1 + 2.7j, 1.0 - 2j, 2.5 + 1j],
            [1.2 + 3.4j, 5.6 + 7.8j, 9.0 + 1.2j, 3.4 + 5.6j],
            ], dtype=np.complex64)
        with closing(self.factory(self.image_parameters, self.grid_parameters, 64)) as collector:
            collector.add(0, uvw, weights, baselines, vis)
        self.check(collector, [[np.rec.fromarrays([
            [[1089, 1011], [951, 908]],
            [[6, 3], [3, 1]],
            [[2.4, 1.8, 2.5, 1.4], [0.5, 0.6, 0.7, 0.8]],
            [[1.97 + 0.75j, 6.78 + 11.88j, 11.7 - 2.04j, 4.76 + 7.84j],
             [0.75 + 0.65j, 0.66 + 1.62j, 0.7 - 1.4j, 2.0 + 0.8j]],
            [64, 65]
            ], dtype=collector.store_dtype)]])


def test_is_prime():
    assert_true(preprocess._is_prime(2))
    assert_true(preprocess._is_prime(3))
    assert_true(preprocess._is_prime(11))
    assert_true(preprocess._is_prime(10007))
    assert_false(preprocess._is_prime(4))
    assert_false(preprocess._is_prime(6))
    assert_false(preprocess._is_prime(18))
    assert_false(preprocess._is_prime(21))


class TestVisibilityCollectorMem(BaseTestVisibilityCollector):
    def factory(self, *args, **kwargs):
        return preprocess.VisibilityCollectorMem(*args, **kwargs)

class TestVisibilityCollectorHDF5(BaseTestVisibilityCollector):
    def setup(self):
        super(TestVisibilityCollectorHDF5, self).setup()
        self._tmpfiles = []

    def factory(self, *args, **kwargs):
        handle, filename = tempfile.mkstemp(suffix='.h5')
        self._tmpfiles.append(filename)
        os.close(handle)
        return preprocess.VisibilityCollectorHDF5(filename, *args, **kwargs)

    def teardown(self):
        for filename in self._tmpfiles:
            try:
                os.remove(filename)
            except OSError as e:
                logging.warning("Failed to remove {}: {}".format(filename, e))
        self._tmpfiles = []
