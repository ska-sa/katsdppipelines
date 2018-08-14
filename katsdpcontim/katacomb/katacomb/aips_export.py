from __future__ import with_statement

import logging

from katacomb import (uv_factory,
                    img_factory,
                    katdal_timestamps,
                    katdal_ant_name,
                    obit_image_mf_rms)

import katpoint
import numpy as np
from scipy.optimize import curve_fit

log = logging.getLogger('katacomb')

# AIPS Table entries follow this kind of schema where data
# is stored in singleton lists, while book-keeping entries
# are not.
# { 'DELTAX' : [1.4], 'NumFields' : 4, 'Table name' : 'AIPS CC' }
# Strip out book-keeping keys and flatten lists
_DROP = { "Table name", "NumFields", "_status" }
def _condition(row):
    """ Flatten singleton lists, drop book-keeping keys
        and convert non-singleton lists to np.ndarray
    """
    return { k: v[0] if len(v) == 1 else np.array(v) for k, v in row.items() if k not in _DROP }

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
                try:
                    cf.tables["AIPS CC"]
                except KeyError:
                    log.warn("No clean components in '%s'", clean_file)
                else:
                    target = "target%d" % si

                    cctab = cf.MergeCC()

                    # Convert cctab to katpoint strings
                    katpoint_rows = cc_to_katpoint(cf)

                    # Extract description
                    description = targets[ti].description
                    data = { 'description': description, 'components': katpoint_rows }

                    # Store them in telstate
                    key = telstate.SEPARATOR.join((target, "clean_components"))
                    telstate.add(key, data, immutable=True)

        except Exception as e:
            log.warn("Export of clean components from '%s' failed.\n%s",
                        clean_file, str(e))

def cc_to_katpoint(img):
    """
    Convert the AIPS CC table attached to img to a
    list of katpoint Target strings.
    The CC table must be in tabulated form (with PARMS[3] = 20.)

    Parameters
    ----------
    img : :class:`ImageFacade`
        Obit MFImage with attached tabulated CC table

    Returns
    -------
    list
        One katpoint Target string per element
    """

    def get_metadata(img):
        """ Extract relevant metadata from img for CC conversion. """
        imlistdict = img.List.Dict
        imdescdict = img.Desc.Dict
        jlocr = imdescdict['jlocr']
        jlocd = imdescdict['jlocd']
        jlocf = imdescdict['jlocf']
        meta = {}
        meta['nimplanes'] = imlistdict['NSPEC'][2][0]
        meta['nimterms'] = imlistdict['NTERM'][2][0]
        meta['reffreq'] = imdescdict['crval'][jlocf]
        meta['refra'] = np.deg2rad(imdescdict['crval'][jlocr])
        meta['refdec'] = np.deg2rad(imdescdict['crval'][jlocd])
        # Center frequencies of the image planes
        meta['planefreqs'] = np.array([imlistdict['FREQ%04d'%(freqid + 1)][2][0] for freqid in range(meta['nimplanes'])])
        # Start and end frequencies of the frequency range
        meta['startfreq'] = imlistdict['FREL0001'][2][0]/1.e6
        meta['endfreq'] = imlistdict['FREH%04d'%(meta['nimplanes'])][2][0]/1.e6
        # Assume projection can be found from ctype 'RA--XXX' where XXX is the projection
        meta['improj'] = imdescdict['ctype'][jlocr].strip()[-3:]
        return meta


    cctab = img.tables["AIPS CC"]
    # Condition all rows up front
    ccrows = [_condition(r) for r in cctab.rows]
    mt = get_metadata(img)
    # RMS of coarse frequency planes for fitting sigma
    planerms = obit_image_mf_rms(img)
    # Only one stokes per image for MFImage output so can assume Stokes has length 1
    # Drop first NTERM planes as these are spectral fit images
    planerms = planerms[mt['nimterms']:, 0]
    # Mask planes with zero RMS (these are completely flagged frequency ranges in the images)
    fitmask = planerms > 0.
    planerms = planerms[fitmask]
    planefreqs = mt['planefreqs'][fitmask]
    katpoint_rows = []
    for ccnum, cc in enumerate(ccrows):
        # PARMS[3] must be 20. for tabulated CCs
        if cc['PARMS'][3] != 20.:
            raise ValueError("Clean Components are not in tabulated form for %s"%(img.aips_path))
        # PARMS[4:] are the tabulated CC flux densities in each image plane.
        ccflux = cc['PARMS'][4:][fitmask]
        # Condition the clean components:
        # 1) Skip any clean component whose mean flux density is < 0.
        # 2) Make sure the flux of the clean components is positive in every plane
        # This will ensure the fitted model is positive which is a requirement of the katpoint
        # FluxDensityModel. It also removes weak negative components which are generally just
        # fits to the noise in individual coarse frequency planes.
        # This conditioning should be revisited after further testing. Perhaps it
        # will be necessary to allow negative components in katpoint models or
        # models from here in flux_model format if there are problems with continuum subtraction.
        if np.mean(ccflux) <= 0.:
            continue
        ccflux = np.abs(ccflux)
        fm_a, fm_b, fm_c = fit_flux_model(planefreqs, ccflux, mt['reffreq'], planerms, cc['FLUX'])
        flux_model = '(%7.2f %7.2f %.6f %.6f %.6f)'%(mt['startfreq'], mt['endfreq'], fm_a, fm_b, fm_c)
        l, m = np.deg2rad([cc['DELTAX'], cc['DELTAY']])
        posn = katpoint.plane_to_sphere[mt['improj']](mt['refra'], mt['refdec'], l, m)
        ra_d, dec_d = np.rad2deg(posn)
        name = 'CC_%06d'%(ccnum)
        katpoint_rows.append(', '.join([name, 'radec', '%.6f'%(ra_d), '%.6f'%(dec_d), flux_model]))

    return katpoint_rows

def fit_flux_model(nu, s, nu0, sigma, sref):
    """
    Fit the 2nd order flux model (ie. with SI + curvature) from Eqn 2. of
    the Obit Development Memo #38, Cotton (2014)
    (ftp.cv.nrao.edu/NRAO-staff/bcotton/Obit/CalModel.pdf):
    s_nu = s_nu0 * exp(alpha*ln(nu/nu0) + beta*ln(nu/nu0)**2)

    Very rarely, the second order fit fails, in which case fall back to a 1st,
    then 0th order fit. If all else fails use the weighted mean of the components.

    Finally convert the fitted parameters to the 2nd order
    katpoint FluxDensityModel:
    log10(S) = a + b*log10(nu) + c*log10(nu)**2

    Parameters
    ----------
    nu : np.ndarray
        Frequencies to fit
    s : np.ndarray
        Flux densities to fit
    nu0 : float
        Reference frequency in same units as nu
    sigma : np.ndarray
        Errors of s
    sref : float
        Initial guess for the value of s at nu0
    """

    def flux_model(lnunu0, *args):
        """
        Compute model:
        args[0]*exp(args[1]*lnunu0 + args[2]*lnunu0**2) ...).
        """
        if args is None:
            return 0.
        iref = args[0]
        exponent = 0.
        for power, arg in enumerate(args[1:]):
            exponent += arg * (lnunu0 ** (power + 1))
        return iref * np.exp(exponent)

    def si_to_katpoint(nu0, iref, alpha1, alpha2):
        """ Convert 2nd order model from flux_model to katpoint flux model.
        """
        nu1 = 1.e6
        r = np.log(nu1 / nu0)
        p = np.log(10.)
        snu1 = iref * np.exp(alpha1 * r + alpha2 * (r ** 2.))
        a0 = np.log10(iref)
        beta1 = alpha1 + 2 * r * alpha2
        beta2 = alpha2
        a1 = beta1
        a2 = beta2 * p
        return a0, a1, a2

    init_si = np.log(nu[0]/nu[-1])/np.log(s[0]/s[-1])
    init = (sref, init_si, 0.,)
    lnunu0 = np.log(nu/nu0)
    for order in range(2, 0, -1):
        try:
            popt, _ = curve_fit(flux_model, lnunu0, s, p0=init[:order + 1], sigma=sigma)
        except RuntimeError:
            log.warn("Fitting flux model of order %d to CC failed. Trying lower order fit."%(order,))
        else:
            iref, alpha1, alpha2 = np.pad(popt, (0, 2 - order), 'constant')
            return si_to_katpoint(nu0, iref, alpha1, alpha2)
    # Give up and return the weighted mean
    iref, alpha1, alpha2 = np.average(s, weights=1./(sigma**2)), 0., 0.
    return si_to_katpoint(nu0, iref, alpha1, alpha2)
