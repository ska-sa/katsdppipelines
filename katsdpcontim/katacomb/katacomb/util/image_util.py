import numpy as np
from matplotlib import use
use('Agg', warn=False)
from matplotlib import pylab as plt

def zscale(image, maxsamples=100000, contrast=0.02, stretch=5.0,
           sigma_rej=3.0, max_iter=500, max_reject=0.1, min_npix=5):
    """
    A python implementation of the IRAF/ds9 zscale algorithm
    for determining maximum and minumum image values for a given
    contrast. This is used to scale a colourmaps applied to FITS 
    images when saving to alternative image formats or for on
    screen display.

    A description of the zscale algorithm can be found here:
    https://iraf.net/forum/viewtopic.php?showtopic=134139

    This implementation has a few minor changes from IRAF:
        1: Don't default to the full data range if iterations
           reject `max_reject` samples- use the range determined
           on the final iteration instead.
        2: Use the masked set of samples to work out the image
           median. The masked data should give an answer without
           including source/artefact contributions to the pixel
           distribution.
        3: Addition of a stretch factor to scale the derived
           minimum and maximum. This is applicable to radio
           images, where negative pixel values can cause the
           vanilla IRAF algorithm to display too much of the
           background rumble.

    Parameters
    ----------
    image : array 
        Input array of pixel values. shape(numxpix, numypix)
    maxsamples : int
        Number of elements to subsample the input image into.
        This improves the speed of the algorithm.
    contrast : float
        The scaling factor (between 0 and 1) for determining
        the returnes minimum and maximum value. Larger values
        decrease the difference between them.
    stretch : float
        Scale factor by which to scale the contrasted gradient.
        The minimum value is derived from gradient / stretch
        and the maximum value from gradient * stretch.
    sigma_rej: float
        Multiple of standard deviation to reject outliers
        while iterating.
    max_iter: int
        Maximum number of iterations.
    max_reject: float
        Stop iterating if number of pixels left is less than
        `max_reject`*npix.
    min_npix: int
        Hard limit on the minimum number of pixels allowed in
        the sampled input and after rejection.
    """


    # Figure out which pixels to use for the zscale algorithm
    # Returns the 1-d array samples
    nx, ny = image.shape
    stride = int(max(1., np.sqrt(nx * ny / float(maxsamples))))
    samples = image[::stride, ::stride].flatten()
    # Remove blanked pixels
    samples = samples[np.isfinite(samples)]
    # Force the maximum number of samples
    samples = samples[:maxsamples]

    samples.sort()
    npix = len(samples)
    zmin = samples[0]
    zmax = samples[-1]

    # Minimum number of pixels allowed
    minpix = max(min_npix, int(npix * max_reject))

    # Grow rejected pixels in each iteration
    ngrow = max(1, int(npix * 0.01))

    if npix <= minpix:
        return zmin, zmax

    # Re-map indices from -1.0 to 1.0
    # to improve fitting.
    xscale = 2.0 / (npix - 1)
    xnorm = np.arange(npix)
    xnorm = xnorm * xscale - 1.0

    # Set up for iteration
    ngoodpix = npix
    last_ngoodpix = npix + 1

    # Mask used in k-sigma clipping.
    badpix = np.zeros(npix, dtype=np.bool)

    # Iterate, until maximum iteration, too many pixels
    # rejected or until no change in number of good pixels
    niter = 0
    while niter < max_iter and ngoodpix > minpix and ngoodpix < last_ngoodpix:
        # Fit a line to the remaining pixels
        good_xnorm = xnorm[~badpix]
        A = np.vstack((good_xnorm, np.ones(len(good_xnorm)))).T
        slope, intercept = np.linalg.lstsq(A, samples[~badpix], rcond=None)[0]

        # Subtract fitted line from the full data array
        fitted = xnorm * slope + intercept
        flat = samples - fitted

        # Compute the k-sigma rejection threshold
        sigma = np.std(flat[~badpix])
        threshold = sigma * sigma_rej

        # Detect and reject pixels further than k*sigma from the fitted line
        badpix = np.abs(flat) > threshold
        
        # Compute number of remaining pixels before convolution
        last_ngoodpix = ngoodpix
        ngoodpix = np.sum(~badpix)

        # Convolve with a kernel of length ngrow
        kernel = np.ones(ngrow, dtype=np.bool)
        badpix = np.convolve(badpix, kernel, mode='same')

        niter += 1
    
    # Transform the slope back to the X range [0:npix-1]
    slope = slope * xscale

    # Apply contrast scaling
    if contrast > 0:
        slope = slope / contrast

    # Stretch the slope
    slope_hi = slope * stretch
    slope_low = slope / stretch

    # Median of the remaining samples is close to true
    # pixel median sans sources/artefacts
    good_samples = samples[~badpix]
    median = np.median(good_samples)

    # Find the indices of the median, low and high pixels
    median_pixel = np.searchsorted(samples, median)
    low_pixel = np.searchsorted(samples, good_samples[0])
    hi_pixel = np.searchsorted(samples, good_samples[-1])

    # Derive scale limits from slope_low and slope_hi
    z1 = max(zmin, median - (median_pixel - low_pixel) * slope_low)
    z2 = min(zmax, median + (hi_pixel - median_pixel) * slope_hi)

    return z1, z2

def save_image(image, filename, plane=1, image_fraction=0.75, cmap='afmhot', dpi=1000, **kwargs):
    """
    Write an image plane to a file using imshow with contrast scaling.
    Input kwargs are passed to the zscale function for the contrast
    scaling.

    Parameters
    ----------
    image : :class:`ImageFacade`
        The image to plot
    filename : str
        Disk location of output. Filename extension determins
        format of the output as for `matplotlib.pylab.savefig`
    image_fraction : float
        Fraction of pixels in x,y image dimensions to plot
    cmap : str or :class:`matplotlib.colors.Colormap`
        Matplotlib style colormap to use (passed to `imshow`)
    dpi : int
        Dots per inch desired for figure.
        (Passed to `matplotlib.pylab.savefig`)
    """

    # Get the image array for the desired plane
    image.GetPlane(None, [plane, 1, 1, 1, 1])
    imagedata = image.np_farray

    xpix, ypix = imagedata.shape 
    
    # Determine amount of image edge to remove
    x_off = int(xpix * 0.5 * (1 - image_fraction))
    y_off = int(ypix * 0.5 * (1 - image_fraction))

    # Cut the image to the desired display size
    x_keep = slice(x_off, xpix - x_off)
    y_keep = slice(y_off, ypix - y_off)
    cutimagedata = imagedata[x_keep, y_keep]

    # Get low and high pixel values to stretch the comormap
    lowcut, highcut = zscale(cutimagedata, **kwargs)

    im=plt.figure(figsize=(10,10))

    # Remove axes and whitespace around edges
    plt.axis('off')
    plt.axes([0, 0, 1, 1])

    # Plot the image with the desired colormap and contrast
    plt.imshow(cutimagedata, cmap=cmap, vmin=lowcut, vmax=highcut,
               aspect='equal', origin='lower', interpolation='nearest')
    
    # Save to filename at the desired dpi
    plt.savefig(filename, dpi=dpi)
    plt.close(im)
