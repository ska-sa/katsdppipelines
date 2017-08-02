from os.path import join as pjoin

import numpy as np
import pyfits

import ObitTalkUtil
import OTObit

from katsdpcontim import obit_err

def fitsdisk_path(fitsdisk, filename):
    """
    Parameters
    ----------
    fitsdisk: integer
        FITS disk on which `filename` is located
    filename: str
        filename on the FITS disk

    Returns
    -------
    str
        Path of `filename` on the FITS disk.
        e.g. `/usr/local/AIPS/FITS/MKATTemplate.uvtab.gz`
    """
    return pjoin(ObitTalkUtil.FITSDir.FITSdisks[fitsdisk], filename)

def open_fits_template(fitsdisk, template_filename=None):
    """
    Parameters
    ----------
    fitsdisk: integer
        FITS Disk on which the template resides
    template_filename (optional): str
        Name of the FITS template file.
        Defaults to `MKATTemplate.uvtab.gz` if
        not specified.

    Returns
    -------
    file object
        The opened FITS template file.
    """
    if template_filename is None:
        template_filename = "MKATTemplate.uvtab.gz"

    return pyfits.open(fitsdisk_path(fitsdisk, template_filename))

def create_fits_from_template(fitsdisk, basefilename, template_filename=None):
    """
    Creates a FITS file from the given template.

    At present this supports reconfiguring the FITS visibility table
    to support the correct number of channels in the MeerKAT observation.

    Parameters
    ----------
    fitsdisk: integer
        FITS Disk on which the template resides
    basefilename: str
        Base filename for the new FITS file.
        `.uvfits` extension is appended to this.
    template_filename (optional): str
        Name of the FITS template file.
        Defaults to `MKATTemplate.uvtab.gz` if
        not specified.

    Returns
    -------
    str
        Full path of the newly created FITS file.

    """
    with open_fits_template(fitsdisk, template_filename) as uvfits:
        # Visibility HDU is at position 1
        vis_hdu = uvfits[1]

        # Get columns describing the visibility table
        vis_table = vis_hdu.columns

        # TODO: Stop hardcoding this.
        nchans = 4096

        # Remove the existing visibility column, replacing
        # with one matching our number of channels.
        vis_table.del_col('VISIBILITIES')

        # Create a new visibility column with the
        # new channel configuration
        newvis = pyfits.Column(name='VISIBILITIES',
                                format='%dE'%(3*4*nchans),
                                dim='(3,4,%d,1,1,1)'%(nchans,),
                                array=np.zeros((1,1,1,nchans,4,3,),
                                dtype=np.float32))

        # Add to table
        vis_table.add_col(newvis)

        new_vis_hdu = pyfits.BinTableHDU.from_columns(vis_table)

        # Transfer
        for key in vis_hdu.header.keys():
            if not key in new_vis_hdu.header.keys() and not key == "HISTORY":
                new_vis_hdu.header[key] = vis_hdu.header[key]

        # Create a HDU header containing the new visibility HDU
        newuvfits = pyfits.HDUList([uvfits[0],
                                    new_vis_hdu,  # Visibilities
                                    uvfits[2],
                                    uvfits[3],
                                    uvfits[4],    # Antenna
                                    uvfits[5],
                                    uvfits[6]])

        # Create the new FITS file
        filename = basefilename + '.uvfits'
        newuvfits.writeto(filename, clobber=True)

        return filename

def open_aips_file_from_fits_template(fitsdisk, basefilename, template_filename=None):
    """
    """

    # Create a FITS file from the given template
    fits_filename = create_fits_from_template(fitsdisk, basefilename, template_filename)

    # AIPS class, disk and sequence
    Aclass = "Raw"
    Adisk = 1
    Aseq = 1         # Implies file creation

    return OTObit.uvlod(fits_filename, fitsdisk, basefilename,
                        Aclass, Adisk, Aseq, obit_err())

