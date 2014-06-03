"""Convert fits file to .jpeg file with useful contrast scaling options.
Optional input is contrast percentage.
"""

import pyfits
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import os, sys
from optparse import OptionParser
import numpy as np

def get_background_variance(data,sigma_clip=5.0,tolerance=0.01):
    """Compute the variance by iteratively removing outliers greater than a given sigma
    until the mean changes by no more than tolerance.

    Inputs
    ------
    data - 1d numpy array of data to compute variance
    sigma_clip - the amount of sigma to clip the data before the next iteration
    tolerance - the fractional change in the mean to stop iterating

    Outputs
    -------
    variance - the final background variance in the sigma clipped image
    """

    #Initialise diff and data_clip and mean and std
    diff = 1
    mean = data.mean()
    data_clip = data
    while diff > tolerance:
        data_clip = data_clip[np.abs(data_clip)<mean+sigma_clip*data_clip.std()]
        newmean = data_clip.mean()
        diff = np.abs(mean-newmean)/(mean+newmean)
        mean = newmean
    return np.var(data_clip)


def writejpeg(data, filename, contrast=99.5, cmap='gray'):
    """Write a 2d array of data to a jpeg with contrast scaling.

    Inputs
    ------
    data - 2d array of data values
    filename - name of output jpeg file (.jpeg will be appended to the name)
    contrast - float between 0 and 100 that selects the fraction of the pixel distrubution to scale into the colormap
    cmap - the desired colourmap to be input to matplotlib
    """

    # Get grid size
    # Make a grid for the x,y coordinate values (just an integer grid)- again this should be reworked one day for a full WCS treatment
    y,x = np.mgrid[slice(1, data.shape[0]+1, 1), slice(1, data.shape[1]+1, 1)]
    # Get the values fot the contrast scaling        
    datasort = np.sort(data[np.isfinite(data)])         #Remone NaNs often found in fits images.
    percentile = contrast/100.0
    arrayremove = int(len(datasort)*(1.0 - percentile)/2.0)
    lowcut,highcut = datasort[arrayremove],datasort[-(arrayremove+1)]
    # make a lookup table for colour scaling (todo- change the colour scaling from linear to other functions))
    levels = np.linspace(lowcut,highcut, num=150)
    # Colormap selection (todo- make the colormap a kwarg)
    cmapin = plt.get_cmap(cmap)
    norm = BoundaryNorm(levels, ncolors=cmapin.N, clip=True)
    im=plt.figure(figsize=(5,5))
    plt.axes([0,0,1,1])
    plt.pcolormesh(x,y,data, cmap=cmapin, norm=norm)
    #Axis size
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.axis('off')
    plt.savefig(filename+'.jpeg')



def fits2jpeg(fitsfilename,contrast=99.5,cmap='gray',chans=None,imchans=False,forceaverage=False,weightaverage=False):
    """Convert FITS files to jpegs using matplotlib.

    Inputs
    ------
    fitsfilenames - String
                    The name of ths input fits file.
    contrast - float between 0 and 1 that selects the fraction of the pixel distrubution to scale into the colormap
    cmap - the desired colourmap to be input to matplotlib
    chans - the desired channel range to use
    imchans - produce an individual jpeg for each channel in the input file
    forceaverage - produce an average of the range of channels in chans
    weightaverage - weight the averages
    """

    #Only work if the image exists
    if not os.path.exists(fitsfilename):
        print 'Specified fits file does not exist'
        sys.exit(-1)
    outname = os.path.splitext(fitsfilename)[0]
    #open file in pyfits. ANd get the image plane(s) and the header
    datahdu = pyfits.open(fitsfilename)
    imageheader = datahdu[0].header
    allimagedata = datahdu[0].data[0]
    #make a masked array to remove nans
    allimagedata = np.ma.masked_array(allimagedata, np.isnan(allimagedata))
    chan_range = chans
    if not chan_range: 
        imagedata=allimagedata
        chan_range='1,'+str(imagedata.shape[0])
    chan_range = chan_range.split(',')
    # Get the desired subset of the fits file to converty to jpeg
    if len(chan_range)==1: imagedata = allimagedata[int(chan_range[0])-1:int(chan_range[0])]
    else: imagedata = allimagedata[int(chan_range[0])-1:int(chan_range[1])-1]    
    # This will work on pipeline images- but needs to be reworked to work on any image you want
    # Loop over every channel and produce a jpeg for each one if the user asks
    if len(chan_range)>1:
    	if imchans==True or imageheader['CTYPE3'] != 'FREQ':
 	       	[writejpeg(imageplane,outname+'_'+str(num+int(chan_range[0])),float(contrast),cmap) for num,imageplane in enumerate(imagedata)]             
    if len(chan_range)==1:
    	if imchans==True or imageheader['CTYPE3'] != 'FREQ':
    		writejpeg(imagedata[0],outname,float(contrast),cmap)                 
    # Do a weighted or straight average image only if image cube or forced average 
    if forceaverage==True or imageheader['CTYPE3'] == 'FREQ':
        # Set a dummy weight array if not doing weights
        weightarray = np.ones(imagedata.shape[0])
        # Recompute weights if the user askes for variance weights
        if weightaverage==True : weightarray = [get_background_variance(imageplane.flatten()) for imageplane in imagedata]
        #Compute the average
        avdata = np.average(imagedata,axis=0,weights=weightarray)
        #Write out the averaged image
        writejpeg(avdata,outname,contrast,cmap)

