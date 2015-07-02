"""Assorted equations for computing appropriate imaging parameters. The
functions take physical quantities as Astropy quantities, rather than
specifying any specific units.

Most formulae are taken from SKA-TEL-SDP-0000003.
"""

from __future__ import division
import astropy.units as units
import math
import numpy as np
import katsdpimager.types


def is_smooth(x):
    """Whether x is a good candidate for FFT. We heuristically require
    it to be a multiple of 64 and a product of powers of 2, 3 and 5."""
    if x % 64 != 0:
        return False
    for d in [2, 3, 5]:
        while x % d == 0:
            x = x // d
    return x == 1


class ArrayParameters(object):
    """Physical attributes of an interferometric array."""
    def __init__(self, antenna_diameter, longest_baseline):
        assert antenna_diameter.unit.physical_type == 'length'
        assert longest_baseline.unit.physical_type == 'length'
        self.antenna_diameter = antenna_diameter
        self.longest_baseline = longest_baseline


class ImageParameters(object):
    """Physical properties associated with an image. At present, only
    single-frequency images are supported.

    Parameters
    ----------
    q_fov : float
        Scale factor for field of view. 1.0 specifies the first null of the
        primary beam (computed just from an Airy disk, not a measured beam
        model).
    image_oversample : float
        Number of pixels per beam
    frequency : Quantity
        Representative frequency, for converting UVW between metres and
        wavelengths. It may also be specified as a wavelength.
    array : :class:`ArrayParameters`
        Properties of the array. It is not needed if both `pixel_size` and
        `pixels` are specified.
    polarizations : list
        List of polarizations that will appear in the image
    dtype : {np.float32, np.complex64}
        Floating-point type for image and grid
    pixel_size : Quantity or float, optional
        Angular size of a single pixel, or dimensionless to specify l or m
        size directly. If specified, `image_oversample` is ignored.
    pixels : int, optional
        Number of pixels in the image. If specified, `q_fov` is ignored.
    """
    def __init__(self, q_fov, image_oversample, frequency, array, polarizations,
                 dtype, pixel_size=None, pixels=None):
        self.wavelength = frequency.to(units.m, equivalencies=units.spectral())
        # Compute pixel size
        if pixel_size is None:
            if image_oversample < 3.0:
                raise ValueError('image_oversample is too small to capture all visibilities in the UV plane')
            # Size of the UV image plane
            uv_size = (2.0 / 3.0 * image_oversample) * array.longest_baseline
            self.pixel_size = (self.wavelength / uv_size).decompose()
        else:
            # Ensure pixel_size is a Quantity and not just a float
            pixel_size = units.Quantity(pixel_size)
            if pixel_size.unit.physical_type == 'angle':
                pixel_size = np.sin(pixel_size)
            self.pixel_size = pixel_size
        # Compute number of pixels
        if pixels is None:
            # These are just a preliminary cell and pixel size, to compute pixels
            cell_size = array.antenna_diameter * (math.pi / (7.6634 * q_fov))
            image_size = self.wavelength / cell_size
            # Allow image to be slightly smaller if it makes the Fourier transform easier
            pixels = int(0.98 * image_size / self.pixel_size)
            while not is_smooth(pixels):
                pixels += 1
        else:
            if not is_smooth(pixels):
                recommended = pixels
                while not is_smooth(recommended):
                    recommended += 1
                raise ValueError("Image size {} not supported - try {}".format(pixels, recommended))
        assert pixels % 2 == 0
        self.real_dtype = np.dtype(dtype)
        self.complex_dtype = katsdpimager.types.real_to_complex(dtype)
        self.pixels = pixels
        self.image_size = self.pixel_size * pixels
        self.cell_size = self.wavelength / self.image_size
        self.polarizations = polarizations

    def __str__(self):
        import polarization
        return """\
Pixel size: {:.3f}
Pixels: {}
FOV: {:.3f}
Cell size: {:.3f}
Wavelength: {:.3f}
Polarizations: {}
""".format(
            np.arcsin(self.pixel_size).to(units.arcsec),
            self.pixels,
            np.arcsin(self.pixel_size * self.pixels).to(units.deg),
            self.cell_size, self.wavelength,
            ','.join([polarization.STOKES_NAMES[i] for i in self.polarizations]))


def w_kernel_width(image_parameters, w, eps_w, antialias_width=0):
    """Determine the width (in UV cells) for a W kernel. This is Eq 9 of
    SKA-TEL-SDP-0000003.

    Parameters
    ----------
    image_parameters : :class:`ImageParameters`
        Image parameters, from which wavelength and image size are used
    w : Quantity
        W value for the kernel, as a distance
    eps_w : float
        Fraction of peak at which to truncate the kernel
    antialias_width : float, optional
        If provided, the return value is for a combined W and antialias
        kernel, where the sizes of the individual kernels are combined
        in quadrature.
    """
    fov = image_parameters.image_size
    wl = float(w / image_parameters.wavelength)
    # Squared size of the w part
    wk2 = 4 * fov**2 * (
        (wl * image_parameters.image_size / 2)**2
        + wl**1.5 * fov / (2 * math.pi * eps_w))
    return np.sqrt(wk2 + antialias_width**2)


class GridParameters(object):
    """Parameters affecting gridding algorithm.

    Parameters
    ----------
    antialias_width : float
        Support of the antialiasing kernel
    oversample : int
        Number of UV sub-cells per cell, for sampling kernels
    image_oversample : int
        Oversampling in image plane during kernel generation
    w_slices : int
        Number of slices for w-stacking
    w_planes : int
        Number of samples to take in w within each slice
    max_w : Quantity
        Maximum absolute w value, as a distance quantity
    kernel_width : int, optional
        Number of UV cells corresponding to the combined W+antialias kernel.
    """
    def __init__(self, antialias_width, oversample, image_oversample,
                 w_slices, w_planes, max_w, kernel_width):
        if max_w.unit.physical_type != 'length':
            raise TypeError('max W must be specified as a length')
        self.antialias_width = antialias_width
        self.oversample = oversample
        self.image_oversample = image_oversample
        self.w_slices = w_slices
        self.w_planes = w_planes
        self.max_w = max_w
        self.kernel_width = kernel_width


class CleanParameters(object):
    def __init__(self, minor, loop_gain, mode, psf_patch):
        self.minor = minor
        self.loop_gain = loop_gain
        self.mode = mode
        self.psf_patch = psf_patch
