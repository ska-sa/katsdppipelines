#!/usr/bin/env python
from __future__ import print_function, division
import math
import sys
import argparse
import casacore.tables
import astropy.io.fits as fits
import astropy.wcs as wcs
import astropy.units as units
import numpy as np
import katsdpsigproc.accel as accel
import katsdpimager.loader as loader
import katsdpimager.parameters as parameters
import katsdpimager.grid as grid
from contextlib import closing

def write_fits(dataset, image, image_parameters, filename):
    header = fits.Header()
    header['BUNIT'] = 'JY/BEAM'
    header['ORIGIN'] = 'katsdpimager'
    # Transformation from pixel coordinates to intermediate world coordinates,
    # which are taken to be l, m coordinates. The reference point is current
    # taken to be the centre of the image (actually half a pixel beyond the
    # centre, because of the way fftshift works).  Note that astropy.io.fits
    # reverses the axis order. The X coordinate is computed differently
    # because the X axis is flipped to allow RA to increase right-to-left.
    header['CRPIX1'] = image.shape[1] * 0.5
    header['CRPIX2'] = image.shape[0] * 0.5 + 1.0
    # FITS uses degrees; and RA increases right-to-left
    delt = np.arcsin(image_parameters.pixel_size).to(units.deg).value
    header['CDELT1'] = -delt
    header['CDELT2'] = delt

    # Transformation from intermediate world coordinates to world
    # coordinates (celestial coordinates in this case).
    # TODO: get equinox from input
    header['EQUINOX'] = 2000.0
    header['RADESYS'] = 'FK5' # Julian equinox
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CTYPE1'] = 'RA---SIN'
    header['CTYPE2'] = 'DEC--SIN'
    phase_centre = dataset.phase_centre()
    header['CRVAL1'] = phase_centre[0].to(units.deg).value
    header['CRVAL2'] = phase_centre[1].to(units.deg).value

    hdu = fits.PrimaryHDU(image[:, ::-1], header)
    hdu.writeto(filename, clobber=True)

def parse_quantity(str_value):
    """Parse a string into an astropy Quantity. Rather than trying to guess
    where the split occurs, we try every position from the back until we
    succeed."""
    for i in range(len(str_value), -1, 0):
        try:
            value = float(str_value[:i])
            unit = units.Unit(str_value[i:])
            return units.Quantity(value, unit)
        except ValueError:
            pass
    raise ValueError('Could not parse {} as a quantity'.format(str_value))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT', help='Input measurement set')
    parser.add_argument('output_file', type=str, metavar='OUTPUT', help='Output FITS file')
    group = parser.add_argument_group('Input selection')
    group.add_argument('--input-option', '-i', action='append', default=[], metavar='KEY=VALUE', help='Backend-specific input parsing option')
    group.add_argument('--channel', '-c', type=int, default=0, help='Channel number')
    group.add_argument('--image-oversample', type=float, default=5, help='Pixels per beam')
    group = parser.add_argument_group('Image options')
    group.add_argument('--q-fov', type=float, default=1.0, help='Field of view to image, relative to main lobe of beam')
    group.add_argument('--pixel-size', type=parse_quantity, help='Size of each image pixel [computed from array]')
    group.add_argument('--pixels', type=int, help='Number of pixels in image')
    group = parser.add_argument_group('Gridding options')
    group.add_argument('--grid-oversample', type=int, default=8, help='Oversampling factor for convolution kernels')
    group.add_argument('--aa-size', type=int, default=7, help='Support of anti-aliasing kernel')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.input_option = ['--' + opt for opt in args.input_option]

    context = accel.create_some_context()
    if not context.device.is_cuda:
        print("Only CUDA is supported at present. Please select a CUDA device.", file=sys.stderr)
        sys.exit(1)
    queue = context.create_command_queue()

    print("Converting {} to {}".format(args.input_file, args.output_file))
    with closing(loader.load(args.input_file, args.input_option)) as dataset:
        array_p = dataset.array_parameters()
        image_p = parameters.ImageParameters(
            args.q_fov, args.image_oversample,
            dataset.frequency(args.channel), array_p,
            args.pixel_size, args.pixels)
        grid_p = parameters.GridParameters(args.aa_size, args.grid_oversample)
        block_size = 65536 # TODO: make tunable
        pols = 1 # TODO: get from the dataset and/or command line
        gridder_template = grid.GridderTemplate(context, grid_p, pols)
        gridder = gridder_template.instantiate(queue, image_p, block_size, accel.SVMAllocator(context))
        gridder.ensure_all_bound()
        gridder.buffer('grid').fill(0)

        for chunk in dataset.data_iter(args.channel, block_size):
            n = len(chunk['uvw'])
            gridder.buffer('uvw')[:n] = chunk['uvw'].to(units.m).value
            gridder.buffer('weights')[:n] = chunk['weights']
            gridder.buffer('vis')[:n] = chunk['vis']
            gridder.set_num_vis(n)
            gridder()
            queue.finish()
        image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gridder.buffer('grid')[:, :, 0]))).real
        write_fits(dataset, image, image_p, args.output_file)

if __name__ == '__main__':
    main()
