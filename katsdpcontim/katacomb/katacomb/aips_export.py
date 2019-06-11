from __future__ import with_statement

import datetime
import json
import logging
from os.path import join as pjoin

from katacomb import (uv_factory,
                      img_factory,
                      katdal_timestamps,
                      katdal_ant_name,
                      obit_image_mf_rms,
                      normalise_target_name,
                      save_image)
import katacomb.configuration as kc

import katpoint
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import binom

log = logging.getLogger('katacomb')

# AIPS Table entries follow this kind of schema where data
# is stored in singleton lists, while book-keeping entries
# are not.
# { 'DELTAX' : [1.4], 'NumFields' : 4, 'Table name' : 'AIPS CC' }
# Strip out book-keeping keys and flatten lists
_DROP = {"Table name", "NumFields", "_status"}

OFILE_SEPARATOR = '_'
# Default image plane to write as FITS.
IMG_PLANE = 1
# File extensions for FITS, PNG and thumbnails
FITS_EXT = '.fits'
PNG_EXT = '.png'
TNAIL_EXT = OFILE_SEPARATOR + 'tnail.png'
METADATA_JSON = 'metadata.json'


def _condition(row):
    """ Flatten singleton lists, drop book-keeping keys
        and convert non-singleton lists to np.ndarray
    """
    return {k: v[0] if len(v) == 1 else np.array(v)
            for k, v in row.items() if k not in _DROP}


def _integration_time(target, katds):
    """ Work out integration time (in hours) on a target,
        preserving katdal selection.
        NOTE:
        This is a plceholder until we can manipulate katdal
        target selection as described in JIRA ticket SR-1732:
        https://skaafrica.atlassian.net/browse/SR-1732
    """
    n_dumps = 0
    selection = katds._selection
    katds.select(reset='T')
    for _, _, scan_targ in katds.scans():
        if scan_targ == target:
            n_dumps += len(katds.dumps)
    katds.select(**selection)
    return n_dumps * katds.dump_period / 3600.


def _update_target_metadata(target_metadata, image, target, tn, katds, filebase):
    """ Append target metadata to target_metadata lists. """
    target_metadata.setdefault('FITSImageFilename', []).append(filebase + FITS_EXT)
    target_metadata.setdefault('PNGImageFileName', []).append(filebase + PNG_EXT)
    target_metadata.setdefault('PNGThumbNailFileName', []).append(filebase + TNAIL_EXT)
    target_metadata.setdefault('IntegrationTime', []).append(_integration_time(target, katds))
    image.GetPlane(None, [IMG_PLANE, 1, 1, 1, 1])
    target_metadata.setdefault('RMSNoise', []).append(str(image.FArray.RMS))
    target_metadata.setdefault('RightAscension', []).append(str(target.radec()[0]))
    target_metadata.setdefault('Declination', []).append(str(target.radec()[1]))
    radec = np.rad2deg(target.radec())
    target_metadata.setdefault('DecRa', []).append(','.join(map(str, radec[::-1])))
    target_metadata.setdefault('Targets', []).append(tn)
    target_metadata.setdefault('KatpointTargets', []).append(target.description)
    return target_metadata


def _metadata(katds, cb_id, target_metadata):
    """ Construct metadata dictionary """
    obs_params = katds.obs_params
    metadata = {}
    product_type = {}
    product_type['ProductTypeName'] = 'FITSImageProduct'
    product_type['ReductionName'] = 'Continuum Image'
    metadata['ProductType'] = product_type
    metadata['Run'] = str(katds.target_indices[0])
    # Format time as required
    start_time = datetime.datetime.utcfromtimestamp(katds.start_time)
    metadata['StartTime'] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata['CaptureBlockId'] = cb_id
    metadata['ScheduleBlockIdCode'] = obs_params.get('sb_id_code', 'UNKNOWN')
    metadata['Description'] = obs_params.get('description', 'UNKNOWN') + ': Continuum image'
    metadata['ProposalId'] = obs_params.get('proposal_id', 'UNKNOWN')
    metadata['Observer'] = obs_params.get('observer', 'UNKNOWN')
    # Add per-target metadata lists
    metadata.update(target_metadata)
    return metadata


def export_images(clean_files, target_indices, disk, kat_adapter):
    """
    Write out FITS, PNG and metadata.json files for each image in clean_files.

    Parameters
    ----------
    clean_files : list
        List of :class:`katacomb.ImageFacade` objects
    target_indices : list of integers
        List of target indices associated with each CLEAN file
    disk : int
        FITS disk number to write to
    kat_adapter : :class:`KatdalAdapter`
        Katdal Adapter
    """

    target_metadata = {}
    targets = kat_adapter.katdal.catalogue.targets
    for clean_file, ti in zip(clean_files, target_indices):
        try:
            with img_factory(aips_path=clean_file, mode='r') as cf:
                # Derive output product label and image class from AIPSPath
                ap = cf.aips_path

                # Get disk location of chosen FITS disk
                # and capture block ID from configuration
                cfg = kc.get_config()
                out_dir = cfg['fitsdirs'][disk - 1][1]
                cb_id = cfg['cb_id']

                # Get and sanitise target name
                targ = targets[ti]
                tn = normalise_target_name(targ.name, target_metadata.get('Targets', []))

                # Output file name
                out_strings = [cb_id, ap.label, tn, ap.aclass]
                out_filebase = OFILE_SEPARATOR.join(filter(None, out_strings))

                log.info('Write FITS image output: %s' % (out_filebase + FITS_EXT))
                cf.writefits(disk, out_filebase + FITS_EXT)

                # Export PNG and a thumbnail PNG
                log.info('Write PNG image output: %s' % (out_filebase + PNG_EXT))
                out_pngfile = pjoin(out_dir, out_filebase + PNG_EXT)
                save_image(cf, out_pngfile, plane=IMG_PLANE)
                out_pngthumbnail = pjoin(out_dir, out_filebase + TNAIL_EXT)
                save_image(cf, out_pngthumbnail, plane=IMG_PLANE, display_size=5., dpi=100)

                # Set up metadata for this target
                _update_target_metadata(target_metadata, cf, targ, tn,
                                        kat_adapter.katdal, out_filebase)

        except Exception as e:
            log.warn("Export of FITS and PNG images from %s failed.\n%s",
                     clean_file, str(e))

    # Export metadata json
    try:
        metadata = _metadata(kat_adapter.katdal, cb_id, target_metadata)
        metadata_file = pjoin(out_dir, METADATA_JSON)
        log.info('Write metadata JSON: %s' % (METADATA_JSON))
        with open(metadata_file, 'w') as meta:
            json.dump(metadata, meta)
    except Exception as e:
        log.warn("Creation of %s failed.\n%s", METADATA_JSON, str(e))


def export_calibration_solutions(uv_files, kat_adapter, telstate):
    """
    Exports calibration solutions from each file
    in ``uv_files`` into ``telstate``.

    Parameters
    ----------
    uv_files : list
        List of :class:`katacomb.UVFacade` objects
    kat_adapter : :class:`KatdalAdapter`
        Katdal Adapter
    telstate : :class:`katsdptelstate.Telescope`
        telstate object
    """

    # MFImage outputs a UV file per source.  Iterate through each source:
    # (1) Extract complex gains from attached "AIPS SN" table
    # (2) Write them to telstate
    for si, uv_file in enumerate(uv_files):
        try:
            with uv_factory(aips_path=uv_file, mode='r') as uvf:
                try:
                    sntab = uvf.tables["AIPS SN"]
                except KeyError:
                    log.warn("No calibration solutions in '%s'", uv_file)
                else:
                    # Handle cases for single/dual pol gains
                    if "REAL2" in sntab.rows[0]:
                        def _extract_gains(row):
                            return np.array([row["REAL1"] + 1j*row["IMAG1"],
                                            row["REAL2"] + 1j*row["IMAG2"]],
                                            dtype=np.complex64)
                    else:
                        def _extract_gains(row):
                            return np.array([row["REAL1"] + 1j*row["IMAG1"],
                                            row["REAL1"] + 1j*row["IMAG1"]],
                                            dtype=np.complex64)

                    # Write each complex gain out per antenna
                    for row in (_condition(r) for r in sntab.rows):
                        # Convert time back from AIPS to katdal UTC
                        time = katdal_timestamps(row["TIME"], kat_adapter.midnight)
                        # Convert from AIPS FORTRAN indexing to katdal C indexing
                        ant = "%s_gains" % katdal_ant_name(row["ANTENNA NO."])

                        # Store complex gain for this antenna
                        # in telstate at this timestamp
                        telstate.add(ant, _extract_gains(row), ts=time)
        except Exception as e:
            log.warn("Export of calibration solutions from '%s' failed.\n%s",
                     uv_file, str(e))


NUM_KATPOINT_PARMS = 10


def export_clean_components(clean_files, target_indices, kat_adapter, telstate):
    """
    Exports clean components from each file in ``clean_files`` into ``telstate``.

    Parameters
    ----------
    clean_files : list
        List of :class:`katacomb.ImageFacade` objects
    target_indices : list of integers
        List of target indices associated with each CLEAN file
    kat_adapter : :class:`KatdalAdapter`
        Katdal Adapter
    telstate : :class:`katsdptelstate.Telescope`
        telstate object
    """

    # MFImage outputs a CLEAN image per source.  Iterate through each source:
    # (1) Extract clean components from attached "AIPS CC" table
    # (2) Merge positionally coincident clean components
    # (3) Convert CC table to list of katpoint Target strings
    # (4) Write them to telstate

    targets = kat_adapter.katdal.catalogue.targets
    it = enumerate(zip(clean_files, target_indices))
    for si, (clean_file, ti) in it:
        try:
            with img_factory(aips_path=clean_file, mode='r') as cf:
                if "AIPS CC" not in cf.tables:
                    log.warn("No clean components in '%s'", clean_file)
                else:
                    target = "target%d" % si

                    cf.MergeCC()

                    # Convert cctab to katpoint strings
                    katpoint_rows = cc_to_katpoint(cf)

                    # Extract description
                    description = targets[ti].description
                    data = {'description': description, 'components': katpoint_rows}

                    # Store them in telstate
                    key = telstate.join(target, "clean_components")
                    telstate.add(key, data, immutable=True)

        except Exception as e:
            log.warn("Export of clean components from '%s' failed.\n%s",
                     clean_file, str(e))


def cc_to_katpoint(img, order=2):
    """
    Convert the AIPS CC table attached to img to a
    list of katpoint Target strings.
    The CC table must be in tabulated form (with PARMS[3] = 20.).

    Parameters
    ----------
    img : :class:`ImageFacade`
        Obit MFImage with attached tabulated CC table
    order : int
        The desired polynomial order of the katpoint FluxDensityModel

    Returns
    -------
    list of strings
        One katpoint Target string per element
    """

    cctab = img.tables["AIPS CC"]
    # Condition all rows up front
    ccrows = [_condition(r) for r in cctab.rows]

    imlistdict = img.List.Dict
    imdescdict = img.Desc.Dict
    jlocr = imdescdict["jlocr"]
    jlocd = imdescdict["jlocd"]
    jlocf = imdescdict["jlocf"]
    jlocs = imdescdict["jlocs"]
    nspec = imlistdict["NSPEC"][2][0]
    nimterms = imlistdict["NTERM"][2][0]
    reffreq = imdescdict["crval"][jlocf]
    refra = np.deg2rad(imdescdict["crval"][jlocr])
    refdec = np.deg2rad(imdescdict["crval"][jlocd])
    # Center frequencies of the image planes
    planefreqs = np.array([imlistdict["FREQ%04d" % (freqid + 1)][2][0]
                           for freqid in range(nspec)])
    # Start and end frequencies of the frequency range
    startfreq = imlistdict["FREL0001"][2][0]
    endfreq = imlistdict["FREH%04d" % (nspec)][2][0]
    # Assume projection can be found from ctype 'RA--XXX' where XXX is the projection
    improj = imdescdict["ctype"][jlocr].strip()[-3:]
    stok = int(imdescdict["crval"][jlocs])

    # RMS of coarse frequency plane images for fitting sigma
    planerms = obit_image_mf_rms(img)
    # Only one stokes per image for MFImage output so can assume Stokes has length 1
    # Drop first NTERM planes as these are spectral fit images
    planerms = planerms[nimterms:, 0]
    # Mask planes with zero RMS (these are completely flagged frequency ranges in the images)
    fitmask = planerms > 0.
    planerms = planerms[fitmask]
    planefreqs = planefreqs[fitmask]
    katpoint_rows = []
    for ccnum, cc in enumerate(ccrows):
        # PARMS[3] must be 20. for tabulated CCs
        if cc["PARMS"][3] != 20.:
            raise ValueError("Clean Components are not in tabulated form for %s" % (img.aips_path))
        # PARMS[4:] are the tabulated CC flux densities in each image plane.
        ccflux = cc["PARMS"][4:][fitmask]
        kp_coeffs = fit_flux_model(planefreqs, ccflux, reffreq,
                                   planerms, cc["FLUX"], stokes=stok, order=order)
        kp_coeffs_str = " ".join([str(coeff) for coeff in kp_coeffs])
        kp_flux_model = "(%r %r %s)" % \
                        (startfreq/1.e6, endfreq/1.e6, kp_coeffs_str)
        l, m = np.deg2rad([cc["DELTAX"], cc["DELTAY"]])
        posn = katpoint.plane_to_sphere[improj](refra, refdec, l, m)
        ra_d, dec_d = np.rad2deg(posn)
        katpoint_rows.append("CC_%06d, radec, %s, %s, %s" %
                             (ccnum, ra_d, dec_d, kp_flux_model))
    return katpoint_rows


def fit_flux_model(nu, s, nu0, sigma, sref, stokes=1, order=2):
    """
    Fit a flux model of given order from Eqn 2. of
    Obit Development Memo #38, Cotton (2014)
    (ftp.cv.nrao.edu/NRAO-staff/bcotton/Obit/CalModel.pdf):
    s_nu = s_nu0 * exp(a0*ln(nu/nu0) + a1*ln(nu/nu0)**2 + ...)

    Very rarely, the requested fit fails, in which case fall
    back to a lower order, iterating until zeroth order. If all
    else fails return the weighted mean of the components.

    Finally convert the fitted parameters to a
    katpoint FluxDensityModel:
    log10(S) = a + b*log10(nu) + c*log10(nu)**2 + ...

    Parameters
    ----------
    nu : np.ndarray
        Frequencies to fit in Hz
    s : np.ndarray
        Flux densities to fit in Jy
    nu0 : float
        Reference frequency in Hz
    sigma : np.ndarray
        Errors of s
    sref : float
        Initial guess for the value of s at nu0
    stokes : int (optional)
        Stokes of image (in AIPSish 1=I, 2=Q, 3=U, 4=V)
    order : int (optional)
        The desired order of the fitted flux model (1: SI, 2: SI + Curvature ...)
    """

    if order > 3:
        raise ValueError("katpoint flux density models are only supported up to 3rd order.")
    init = [sref, -0.7] + [0] * (order - 1)
    lnunu0 = np.log(nu/nu0)
    for fitorder in range(order, -1, -1):
        try:
            popt, _ = curve_fit(obit_flux_model, lnunu0, s, p0=init[:fitorder + 1], sigma=sigma)
        except RuntimeError:
            log.warn("Fitting flux model of order %d to CC failed. Trying lower order fit." %
                     (fitorder,))
        else:
            coeffs = np.pad(popt, ((0, order - fitorder),), "constant")
            return obit_flux_model_to_katpoint(nu0, stokes, *coeffs)
    # Give up and return the weighted mean
    coeffs = [np.average(s, weights=1./(sigma**2))] + [0] * order
    return obit_flux_model_to_katpoint(nu0, stokes, *coeffs)


def obit_flux_model(lnunu0, iref, *args):
    """
    Compute model:
    (iref*exp(args[0]*lnunu0 + args[1]*lnunu0**2) ...)
    """
    exponent = np.sum([arg * (lnunu0 ** (power + 1))
                       for power, arg in enumerate(args)], axis=0)
    return iref * np.exp(exponent)


def obit_flux_model_to_katpoint(nu0, stokes, iref, *args):
    """ Convert model from Obit flux_model to katpoint FluxDensityModel.
    """
    kpmodel = [0.] * NUM_KATPOINT_PARMS
    # +/- component?
    sign = np.sign(iref)
    nu1 = 1.e6
    r = np.log(nu1 / nu0)
    p = np.log(10.)
    exponent = np.sum([arg * (r ** (power + 1))
                       for power, arg in enumerate(args)])
    # Compute log of flux_model directly to avoid
    # exp of extreme values when extrapolating to 1MHz
    lsnu1 = np.log(sign * iref) + exponent
    a0 = lsnu1 / p
    kpmodel[0] = a0
    n = len(args)
    for idx in range(1, n + 1):
        coeff = np.poly1d([binom(j, idx) * args[j - 1]
                           for j in range(n, idx - 1, -1)])
        betai = coeff(r)
        ai = betai * p ** (idx - 1)
        kpmodel[idx] = ai
    # Set Stokes +/- based on sign of iref
    # or zero in the unlikely event that iref is zero
    # I, Q, U, V are last 4 elements of kpmodel
    kpmodel[stokes - 5] = sign
    return kpmodel
