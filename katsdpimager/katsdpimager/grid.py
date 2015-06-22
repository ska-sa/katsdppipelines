# -*- coding: utf-8 -*-

"""Convolutional gridding"""

from __future__ import division, print_function
from . import parameters
import numpy as np
import math
import pkg_resources
import astropy.units as units
import katsdpsigproc.accel as accel
import katsdpsigproc.tune as tune
import numba


def kaiser_bessel(x, width, beta):
    r"""Evaluate Kaiser-Bessel window function. Refer to
    http://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html
    for details.

    Parameters
    ----------
    x : array-like, float
        Sample positions
    width : float
        The kernel has support :math:`[-\frac{1}{2}W, \frac{1}{2}W]`
    beta : float
        Shape parameter
    """
    param = 1 - (2 * x / width)**2
    # The np.maximum is to protect against runtime warnings for taking
    # sqrt of negative values. The actual values in this situation are
    # irrelvent due to the np.select call.
    values = np.i0(beta * np.sqrt(np.maximum(0, param))) / np.i0(beta)
    return np.select([param >= 0], [values])


def kaiser_bessel_fourier(f, width, beta):
    """
    Evaluate the continuous Fourier transform of :func:`kaiser_bessel`.
    Note that since the function is even and real, this is also the inverse
    Fourier transform.

    Parameters
    ----------
    f : array-like, float
        Sample positions (frequency)
    width : float
        The kernel has support :math:`[-\frac{1}{2}W, \frac{1}{2}W]`
    beta : float
        Shape parameter
    """
    alpha = beta / math.pi
    # np.lib.scimath.sqrt returns complex values for negative inputs, whereas
    # np.sqrt returns NaN. We take the real component because np.sinc returns
    # complex if it has a complex input, even though the imaginary part is 0.
    return width / np.i0(beta) * np.sinc(np.lib.scimath.sqrt((width * f)**2 - alpha * alpha)).real


def antialias_kernel(width, oversample, beta=None):
    r"""Generate 1D anti-aliasing kernel. The return value is 2-dimensional.
    The first dimension selects the subpixel position, the second
    selects the pixel position.

    The returned kernels all have the same size, but it is not necessarily
    the same size as `width` (in fact, `width` may be non-integer). Given
    a real-valued :math:`x` coordinate and a grid point :math:`x_g` to be
    updated, the indices are computed as follows:

    .. math::

       x_0 &= \lfloor x\rfloor\\
       s &= \lfloor (x - x_0)\cdot\text{oversample}\rfloor\\
       u &= x_g - x_0 + \tfrac12 \lvert \text{kernel}\rvert - 1

    Then `s` is the subpixel index and `u` is the pixel index.

    TODO: normalisation so that integral is 1?

    Parameters
    ----------
    width : float
        Kernel support
    oversample : int
        Number of samples per unit step
    beta : float, optional
        Shape parameter for Kaiser-Bessel window
    """
    if beta is None:
        # Puts the first null of the taper function at the edge of the image
        beta = math.pi * math.sqrt(0.25 * width**2 - 1.0)
        # Move the null outside the image, to avoid numerical instabilities.
        # This will cause a small amount of aliasing at the edges, which
        # ideally should be handled by clipping the image.
        beta *= 1.2
    hsize = int(math.ceil(0.5 * width))
    size = 2 * hsize
    # The kernel only contains real values, but we store complex so that we
    # are ready for W projection.
    kernel = np.empty((oversample, size), np.complex64)
    for s in range(oversample):
        bias = (s + 0.5) / oversample + hsize - 1
        kernel[s, :] = kaiser_bessel(np.arange(size) - bias, width, beta)
    return kernel


def kernel_outer(a, b, out=None):
    """Take the outer product of two 1D kernels"""
    return np.multiply(a[:, np.newaxis, :, np.newaxis],
                       b[np.newaxis, :, np.newaxis, :], out)


def antialias_w_kernel(cell_wavelengths, w, width, oversample, antialias_width, image_oversample, beta):
    r"""Computes a combined anti-aliasing and W-projection kernel.

    The format of the returned kernel is similar to :func:`antialias_kernel`.
    In particular, only a 1D kernel is returned.  Note that while the W kernel
    is not truly separable, the small-angle approximation
    :math:`\sqrt{1-l^2-m^2}-1 \approx (\sqrt{1-l^2}-1) + (\sqrt{1-m^2}-1)`
    makes it makes it very close to separable [#]_.

    .. [#] The error in the approximation above is on the order of :math:10^{-8}
       for a 1 degree field of view.

    We multiply the inverse Fourier transform of the (idealised) antialiasing
    kernel with the W correction in image space, to obtain a closed-form
    function we can evaluate in image space. This is then Fourier-transformed
    to UV space. To reduce aliasing, the image-space function is oversampled by
    a factor of `uv_oversample`, and the transform result is then truncated.

    All the continuous functions that we are approximating in UV space are
    expressed as functions of the number of wavelengths.

    The output kernel is sampled at the centres of bins, which are at
    half-subpixel offsets. To create this shift in UV space, we multiply by the
    appropriate complex exponential in image space.

    Parameters
    ----------
    cell_wavelengths : float
        Size of a UV cell in wavelengths
    w : float
        w component of baseline, in wavelengths
    width : int
        Support of combined kernel, in cells
    oversample : int
        Number of samples per unit step in UV space
    antialias_width : float
        Support for anti-alias kernel, in cells
    image_oversample : int
        Oversampling factor in image space. Larger values reduce aliasing in
        the output and increase time and memory required for computation, but
        do not affect the size of the resulting kernel.
    beta : float, optional
        Shape parameter for Kaiser-Bessel window
    """
    def image_func(l):
        # The kaiser_bessel function is designed around units of cells rather
        # than wavelengths. We thus want the Fourier transform of
        # kaiser_bessel(u / cell_wavelengths).
        aa_factor = cell_wavelengths * kaiser_bessel_fourier(l * cell_wavelengths, antialias_width, beta)
        shift_arg = shift_by * l
        w_arg = -w * (np.sqrt(1 - l*l) - 1)
        return aa_factor * np.exp(2j * math.pi * (w_arg + shift_arg))

    out_pixels = oversample * width
    assert out_pixels % 2 == 0, "Odd number of pixels is not tested"
    pixels = out_pixels * image_oversample
    # Convert uv-space width to wavelengths
    uv_width = width * cell_wavelengths * image_oversample
    # Compute other support and step sizes
    image_step = 1 / uv_width
    uv_step = uv_width / pixels
    image_width = image_step * pixels
    # Determine sample points in image space
    l = (np.arange(pixels) - (pixels // 2)) * image_step
    # Evaluate function in image space
    shift_by = -0.5 * cell_wavelengths / oversample
    image_values = image_func(l)
    # Convert to UV space. The multiplication is because we're using a DFT to
    # approximate a continuous FFT.
    uv_values = np.fft.fft(np.fft.ifftshift(image_values)) * image_step
    # Crop to area of interest, and swap halves to put DC in the middle
    uv_values = np.concatenate((uv_values[-(out_pixels // 2):], uv_values[:(out_pixels // 2)]))
    # Split up into subkernels. Since the subpixel index indicates the subpixel
    # position of the visibility, rather than the kernel tap, it runs backwards
    # in the kernel indexing.
    kernel = np.reshape(uv_values, (oversample, width), order='F')[::-1, :]
    # Convert to C memory layout
    return np.copy(kernel)


@numba.jit(nopython=True)
def subpixel_coord(x, oversample):
    """Return pixel and subpixel index, as described in :func:`antialias_kernel`"""
    x0 = np.floor(x)
    return int(x0), int(np.floor((x - x0) * oversample))


class GridderTemplate(object):
    def generate_convolve_kernel(
            self, context, image_parameters, grid_parameters, width):
        cell_wavelengths = float(image_parameters.cell_size / image_parameters.wavelength)
        max_w_wavelengths = float(grid_parameters.max_w / image_parameters.wavelength)
        self.convolve_kernel = accel.SVMArray(
            context,
            (grid_parameters.w_planes, grid_parameters.oversample, grid_parameters.oversample, width, width),
            np.complex64)
        # Puts the first null of the taper function at the edge of the image
        self.beta = math.pi * math.sqrt(0.25 * grid_parameters.antialias_size**2 - 1.0)
        # Move the null outside the image, to avoid numerical instabilities.
        # This will cause a small amount of aliasing at the edges, which
        # ideally should be handled by clipping the image.
        self.beta *= 1.2
        # TODO: use sqrt(w) scaling as in Cornwell, Golap and Bhatnagar (2008)?
        for i, w in enumerate(np.linspace(0.0, max_w_wavelengths, grid_parameters.w_planes)):
            kernel1d = antialias_w_kernel(
                cell_wavelengths, w, width,
                grid_parameters.oversample,
                grid_parameters.antialias_size,
                grid_parameters.image_oversample,
                self.beta)
            self.convolve_kernel[i, ...] = kernel_outer(kernel1d, kernel1d)

    def __init__(self, context, image_parameters, grid_parameters, tuning=None):
        if tuning is None:
            tuning = self.autotune(
                context, grid_parameters.antialias_size, grid_parameters.oversample,
                len(image_parameters.polarizations))
        self.grid_parameters = grid_parameters
        self.image_parameters = image_parameters
        self.dtype = image_parameters.complex_dtype
        self.wgs_x = 16   # TODO: compute based on max_w (or parameter)
        self.wgs_y = 16
        self.multi_x = 1
        self.multi_y = 1
        self.num_polarizations = len(image_parameters.polarizations)
        tile_x = self.wgs_x * self.multi_x
        tile_y = self.wgs_y * self.multi_y
        assert tile_x == tile_y
        self.generate_convolve_kernel(context, image_parameters, grid_parameters, tile_x)
        w_scale = float(units.m / grid_parameters.max_w) * (grid_parameters.w_planes - 1)
        self.program = accel.build(
            context, "imager_kernels/grid.mako",
            {
                'real_type': ('float' if self.dtype == np.complex64 else 'double'),
                'convolve_kernel_row_stride': self.convolve_kernel.padded_shape[4],
                'convolve_kernel_slice_stride':
                    self.convolve_kernel.padded_shape[3] * self.convolve_kernel.padded_shape[4],
                'convolve_kernel_oversample': self.convolve_kernel.shape[1],
                'convolve_kernel_w_stride': np.product(self.convolve_kernel.padded_shape[1:]),
                'convolve_kernel_w_scale': w_scale,
                'convolve_kernel_max_w': float(grid_parameters.max_w / units.m),
                'multi_x': self.multi_x,
                'multi_y': self.multi_y,
                'wgs_x': self.wgs_x,
                'wgs_y': self.wgs_y,
                'num_polarizations': self.num_polarizations
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])

    @classmethod
    @tune.autotuner(test={})
    def autotune(cls, context, antialias_size, oversample, num_polarizations):
        # Nothing to autotune yet
        return {}

    def instantiate(self, *args, **kwargs):
        return Gridder(self, *args, **kwargs)

    def taper(self, N, out=None):
        """Return the Fourier transform of the antialiasing kernel for
        an N×N image.

        Parameters
        ----------
        N : int
            Number of pixels in the image
        out : array-like, optional
            If provided, is used to store the result
        """
        x = np.arange(N) / N - 0.5
        taper1d = kaiser_bessel_fourier(x, self.grid_parameters.antialias_size, self.beta)
        out = np.outer(taper1d, taper1d, out)
        lm = (np.arange(N) - (N // 2)) * float(self.image_parameters.pixel_size)
        # We image T.I/n, so we need to correct for the division by n
        l = lm[:, np.newaxis]
        m = lm[np.newaxis, :]
        out /= np.sqrt(1.0 - (l * l + m * m))
        return out


class Gridder(accel.Operation):
    def __init__(self, template, command_queue, array_parameters,
                 max_vis, allocator=None):
        super(Gridder, self).__init__(command_queue, allocator)
        # Check that longest baseline won't cause an out-of-bounds access
        max_uv_src = float(array_parameters.longest_baseline / template.image_parameters.cell_size)
        convolve_kernel_size = template.convolve_kernel.shape[-1]
        max_uv = max_uv_src + convolve_kernel_size / 2
        if max_uv >= template.image_parameters.pixels // 2 - 1 - 1e-3:
            raise ValueError('image_oversample is too small to capture all visibilities in the UV plane')
        self.template = template
        self.max_vis = max_vis
        self.slots['grid'] = accel.IOSlot(
            (template.image_parameters.pixels, template.image_parameters.pixels,
                accel.Dimension(template.num_polarizations, exact=True)),
            template.dtype)
        self.slots['uvw'] = accel.IOSlot(
            (max_vis, accel.Dimension(3, exact=True)), np.float32)
        self.slots['vis'] = accel.IOSlot(
            (max_vis, accel.Dimension(template.num_polarizations, exact=True)), np.complex64)
        self.kernel = template.program.get_kernel('grid')
        self._num_vis = 0
        cell_size_m = template.image_parameters.cell_size.to(units.m).value
        self.uv_scale = template.grid_parameters.oversample / cell_size_m
        # Offset to bias coordinates such that l,m=0 translates to the first
        # pixel to update in the grid, measured in subpixels
        uv_bias_pixels = template.image_parameters.pixels // 2 - (convolve_kernel_size - 1) // 2
        self.uv_bias = float(uv_bias_pixels) * template.grid_parameters.oversample

    def set_num_vis(self, n):
        if n < 0 or n > self.max_vis:
            raise ValueError('Number of visibilities {} is out of range 0..{}'.format(
                n, self.max_vis))
        self._num_vis = n

    def clear(self):
        """TODO: implement on GPU"""
        self.buffer('grid').fill(0)

    def grid(self, uvw, vis):
        """Add visibilities to the grid, with convolutional gridding using the
        anti-aliasing filter."""
        self.set_num_vis(len(uvw))
        self.buffer('uvw')[:len(uvw)] = uvw
        self.buffer('vis')[:len(vis)] = vis
        self()

    def _run(self):
        if self._num_vis == 0:
            return
        grid = self.buffer('grid')
        workgroups = 256  # TODO: tune this in some way
        vis_per_workgroup = accel.divup(self._num_vis, workgroups)
        self.command_queue.enqueue_kernel(
            self.kernel,
            [
                grid.buffer,
                np.int32(grid.padded_shape[1]),
                self.buffer('uvw').buffer,
                self.buffer('vis').buffer,
                self.template.convolve_kernel.buffer,
                np.float32(self.uv_scale),
                np.float32(self.uv_bias),
                np.int32(vis_per_workgroup),
                np.int32(self._num_vis),
            ],
            global_size=(self.template.wgs_x * workgroups, self.template.wgs_y),
            local_size=(self.template.wgs_x, self.template.wgs_y)
        )

    def parameters(self):
        return {
            'grid_parameters': self.template.grid_parameters,
            'image_parameters': self.template.image_parameters
        }


@numba.jit(nopython=True)
def _grid(kernel, values, uvw, vis, pixels, cell_size, oversample):
    """Internal implementation of :meth:`GridderHost.grid`, split out so that
    Numba can JIT it.
    """
    ksize = kernel.shape[2]
    # Offset to bias coordinates such that l,m=0 translates to the first
    # pixel to update in the grid.
    offset = pixels // 2 - (ksize - 1) // 2
    for row in range(uvw.shape[0]):
        # l and m are measured in cells
        l = np.float32(uvw[row, 0] / cell_size) + offset
        m = np.float32(uvw[row, 1] / cell_size) + offset
        l0, sub_l = subpixel_coord(l, oversample)
        m0, sub_m = subpixel_coord(m, oversample)
        sample = vis[row, :]
        sub_kernel = kernel[sub_m, sub_l, :, :]
        for j in range(ksize):
            for k in range(ksize):
                for pol in range(values.shape[2]):
                    values[int(m0 + j), int(l0 + k), pol] += sample[pol] * sub_kernel[j, k]


class GridderHost(object):
    def __init__(self, image_parameters, grid_parameters):
        self.image_parameters = image_parameters
        self.grid_parameters = grid_parameters
        self.kernel1d = antialias_kernel(grid_parameters.antialias_size,
                                         grid_parameters.oversample)
        self.kernel = kernel_outer(self.kernel1d, self.kernel1d)
        pixels = image_parameters.pixels
        shape = (pixels, pixels, len(image_parameters.polarizations))
        self.values = np.empty(shape, image_parameters.complex_dtype)

    def clear(self):
        self.values.fill(0)

    def taper(self, N, out=None):
        """Return the Fourier transform of the convolution kernel.
        See :func:`fourier_kernel` for details.
        """
        taper1d = fourier_kernel(self.kernel1d, N)
        return np.outer(taper1d, taper1d, out)

    def grid(self, uvw, vis):
        """Add visibilities to the grid, with convolutional gridding using the
        anti-aliasing filter.

        Parameters
        ----------
        uvw : 2D Quantity array
            UVW coordinates for visibilities, indexed by sample then u/v/w
        vis : 2D ndarray of complex
            Visibility data, indexed by sample and polarization, and
            pre-multiplied by all weights
        """
        assert uvw.unit.physical_type == 'length'
        pixels = self.image_parameters.pixels
        _grid(self.kernel, self.values,
              uvw.to(units.m).value, vis,
              self.image_parameters.pixels,
              self.image_parameters.cell_size.to(units.m).value,
              self.grid_parameters.oversample)
