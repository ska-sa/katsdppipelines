#!/usr/bin/env python
from __future__ import print_function, division
import math
import argparse
import casacore.tables
import astropy.io.fits as fits
import astropy.wcs as wcs
import astropy.units as units
import numpy as np
import katsdpsigproc
import katsdpimager.loader as loader
import katsdpimager.parameters as parameters
from contextlib import closing

def write_fits(dataset, image, image_parameters, filename):
    header = fits.Header()
    header['BUNIT'] = 'JY/BEAM'
    header['ORIGIN'] = 'katsdpimager'
    # Transformation from pixel coordinates to intermediate world
    # coordinates, which are taken to be l, m coordinates. The reference
    # point is current taken to be the centre of the image.
    # Note that astropy.io.fits reverses the axis order.
    header['CRPIX1'] = image.shape[1] * 0.5 + 0.5
    header['CRPIX2'] = image.shape[0] * 0.5 + 0.5
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

    hdu = fits.PrimaryHDU(image, header)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='INPUT', help='Input measurement set')
    parser.add_argument('output_file', type=str, metavar='OUTPUT', help='Output FITS file')
    parser.add_argument('--input-option', '-i', action='append', default=[], metavar='KEY=VALUE', help='Backend-specific input parsing option')
    parser.add_argument('--channel', '-c', type=int, default=0, help='Channel number')
    parser.add_argument('--image-oversample', type=float, default=5, help='Pixels per beam')
    parser.add_argument('--q-fov', type=float, default=1.0, help='Field of view to image, relative to main lobe of beam')
    parser.add_argument('--pixel-size', type=parse_quantity, help='Size of each image pixel [computed from array]')
    parser.add_argument('--pixels', type=int, help='Number of pixels in image')

    args = parser.parse_args()
    args.input_option = ['--' + opt for opt in args.input_option]

    print("Converting {} to {}".format(args.input_file, args.output_file))
    with closing(loader.load(args.input_file, args.input_option)) as dataset:
        print(dataset.antenna_diameter())
        print(dataset.longest_baseline())
        for chunk in dataset.data_iter(args.channel, 65536):
            print(chunk['vis'].shape, chunk['weights'].shape, chunk['uvw'].shape)

        array_p = dataset.array_parameters()
        # TODO: get frequency from data
        image_p = parameters.ImageParameters(
            args.q_fov, args.image_oversample, 0.21 * units.m, array_p,
            args.pixel_size, args.pixels)
        print(image_p)
        image = np.zeros((1024, 1024), dtype=np.float32)
        write_fits(dataset, image, image_p, args.output_file)

if __name__ == '__main__':
    main()
