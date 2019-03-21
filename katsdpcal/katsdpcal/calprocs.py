"""
Calibration procedures for MeerKAT calibration pipeline
=======================================================

Solvers and averagers for use in the MeerKAT calibration pipeline.
"""

from __future__ import print_function
import logging

import numpy as np
import scipy.fftpack
import numba

import katpoint


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# --- Modelling procedures
# --------------------------------------------------------------------------------------------------


def radec_to_lm(ra, dec, ra0, dec0):
    """
    Parameteters
    ------------
    ra : float
        Right Ascension, radians
    dec : float
        Declination, radians
    ra0 : float
        Right Ascention of centre, radians
    dec0 : float
        Declination of centre, radians

    Returns
    -------
    l, m : float
        direction cosines
    """
    l = np.cos(dec)*np.sin(ra - ra0)    # noqa: E741
    m = np.sin(dec)*np.cos(dec0) - np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)
    return l, m


def list_to_model(model_list, centre_position):
    lmI_list = []

    ra0, dec0 = centre_position

    for ra, dec, I in model_list:
        l, m = radec_to_lm(ra, dec, ra0, dec0)
        lmI_list.append([l, m, I])

    return lmI_list


def to_ut(t):
    """
    Converts MJD seconds into Unix time in seconds

    Parameters
    ----------
    t : float
        time in MJD seconds

    Returns
    -------
    float
        Unix time in seconds
    """
    return (t/86400. - 2440587.5 + 2400000.5)*86400.


def calc_uvw_wave(phase_centre, timestamps, corrprod_lookup, antennas,
                  wavelengths=None, array_centre=None):
    """
    Calculate uvw coordinates

    Parameters
    ----------
    phase_centre : :class:`katpoint.Target`
        phase centre position
    timestamps : array of float
        times, shape(nrows)
    corrprod_lookup : array
        lookup table of antenna indices for each baseline, shape(nant,2)
    antennas : list of :class:`katpoint.Antenna`
        the antennas, same order as antlist
    wavelengths : array or scalar
        wavelengths, single value or array shape(nchans)
    array_centre : :class:`katpoint.Antenna`
        array centre position

    Returns
    -------
    uvw_wave
        uvw coordinates, normalised by wavelength
    """
    uvw = calc_uvw(phase_centre, timestamps, corrprod_lookup, antennas, array_centre)
    if wavelengths is None:
        return uvw
    elif np.isscalar(wavelengths):
        return uvw/wavelengths
    else:
        return uvw[:, :, np.newaxis, :]/wavelengths[:, np.newaxis]


def calc_uvw(phase_centre, timestamps, corrprod_lookup, antennas, array_centre=None):
    """
    Calculate uvw coordinates

    Parameters
    ----------
    phase_centre : :class:`katpoint.Target`
        phase centre position
    timestamps : array of float
        times, shape(nrows)
    corrprod_lookup : array
        lookup table of antenna indices for each baseline, shape(nant,2)
    antennas : list of :class:`katpoint.Antenna`
        the antennas, same order as antlist
    array_centre : str
        array centre position

    Returns
    -------
    uvw_wave
        uvw coordinates
    """
    if array_centre is None:
        # if no array centre position is given, use lat-long-alt of first
        # antenna in the antenna list
        array_centre = katpoint.Antenna('array_position', *antennas[0].ref_position_wgs84)

    # use the array reference position for the basis
    basis = phase_centre.uvw_basis(timestamp=timestamps, antenna=array_centre)
    antenna_uvw = np.empty([len(antennas), 3, len(timestamps)])

    for i, ant in enumerate(antennas):
        enu = np.array(ant.baseline_toward(array_centre))
        antenna_uvw[i, ...] = np.tensordot(basis, enu, ([1], [0]))

    baseline_uvw = np.empty([3, len(timestamps), len(corrprod_lookup)])
    for i, [a1, a2] in enumerate(corrprod_lookup):
        baseline_uvw[..., i] = antenna_uvw[a2] - antenna_uvw[a1]

    return baseline_uvw

# --------------------------------------------------------------------------------------------------
# --- Solvers
# --------------------------------------------------------------------------------------------------


def get_bl_ant_pairs(corrprod_lookup):
    """
    Get antenna lists in solver format, from `corrprod_lookup`

    Inputs:
    -------
    corrprod_lookup : array
        lookup table of antenna indices for each baseline, array shape(nant,2)

    Returns:
    --------
    antlist1, antlist2 : list
        lists of antennas matching the correlation_product lookup table,
        appended with their conjugates (format required by stefcal)
    """

    # NOTE: no longer need hh and vv masks as we re-ordered the data to be
    # ntime x nchan x nbl x npol

    # get antenna number lists for stefcal - need vis then vis.conj (assume
    # constant over an observation)
    # assume same for hh and vv
    antlist1 = np.concatenate((corrprod_lookup[:, 0], corrprod_lookup[:, 1]))
    antlist2 = np.concatenate((corrprod_lookup[:, 1], corrprod_lookup[:, 0]))

    return antlist1, antlist2


@numba.guvectorize(['c8[:], int_[:], int_[:], f4[:], int_[:], c8[:], int_[:], f4[:], c8[:]',
                    'c16[:], int_[:], int_[:], f8[:], int_[:], c16[:], int_[:], f8[:], c16[:]'],
                   '(n),(n),(n),(n),(),(a),(),()->(a)', nopython=True, cache=True)
def _stefcal_gufunc(rawvis, ant1, ant2, weights, ref_ant, init_gain, num_iters, conv_thresh, g):
    ref_ant2 = max(ref_ant[0], 0)
    num_ants = init_gain.shape[0]
    R = np.zeros((num_ants, num_ants), rawvis.dtype)    # Weighted visibility matrix
    M = np.zeros((num_ants, num_ants), weights.dtype)   # Weighted model
    # Convert list of baselines into a covariance matrix
    for i in range(len(ant1)):
        a = ant1[i]
        b = ant2[i]
        if a != b:
            weighted_vis = rawvis[i] * weights[i]
            R[a, b] = weighted_vis
            R[b, a] = np.conj(weighted_vis)
            M[a, b] = weights[i]
            M[b, a] = weights[i]
    g_old = init_gain.copy()

    # Remove completely flagged antennas,
    # where flags are indicated by a weighted visibility of zero
    good_ant = np.ones(num_ants, np.bool_)
    for p in range(num_ants):
        good_ant[p] = ~np.all(R[p] == 0j)

    antlist = np.arange(num_ants)[good_ant]  # list of good antennas to iterate over.
    for n in range(num_iters[0]):
        for p in antlist:
            gn = g.dtype.type(0)
            gd = g.real.dtype.type(0)
            for i in antlist:
                z = g_old[i] * M[p, i]
                gn += R[p, i] * z       # exploiting R[p, i] = R[i, p].conj()
                gd += z.real * z.real + z.imag * z.imag
            g[p] = gn / gd
        # Salvini & Wijnholds tweak gains every *even* iteration but their counter starts at 1
        if n % 2:
            # Check for convergence of relative l_2 norm of change in gain vector
            dnorm2 = g.real.dtype.type(0)
            gnorm2 = g.real.dtype.type(0)
            for i in antlist:
                delta = g[i] - g_old[i]
                dnorm2 += delta.real * delta.real + delta.imag * delta.imag
                gnorm2 += g[i].real * g[i].real + g[i].imag * g[i].imag
            # The sense of this if test is carefully chosen so that if
            # g contains nans, the loop will exit (there is no point
            # continuing, since the nans can only spread.
            if dnorm2 >= conv_thresh[0] * conv_thresh[0] * gnorm2:
                # Avoid getting stuck bouncing between two gain vectors by
                # going halfway in between
                for i in antlist:
                    g[i] = (g[i] + g_old[i]) / 2
            else:
                break
        g_old[:] = g

    # replace gains of completely flagged antennas with NaNs to indicate
    # that no gain was found by stefcal.
    g[~good_ant] = np.nan*1j

    ref = g[ref_ant2]
    g *= np.conj(ref) / np.abs(ref)

    if ref_ant[0] < 0:
        middle_angle = np.median(g.real) - np.median(g.imag) * g.dtype.type(1j)
        middle_angle /= np.abs(middle_angle)
        g *= middle_angle


def stefcal(rawvis, num_ants, corrprod_lookup, weights=None, ref_ant=0,
            init_gain=None, num_iters=30, conv_thresh=0.0001):
    """Solve for antenna gains using StEFCal.

    The observed visibilities are provided in a NumPy array of any shape and
    dimension, as long as the last dimension represents baselines. The gains
    are then solved in parallel for the rest of the dimensions. For example,
    if the *vis* array has shape (T, F, B) containing *T* dumps / timestamps,
    *F* frequency channels and *B* baselines, the resulting gain array will be
    of shape (T, F, num_ants), where *num_ants* is the number of antennas.

    Parameters
    ----------
    rawvis : array of complex, shape (M, ..., N)
        Complex cross-correlations between antennas A and B, assuming *N*
        baselines or antenna pairs on the last dimension
    num_ants : int
        Number of antennas
    corrprod_lookup : numpy array of int, shape (N, 2)
        First and second antenna indices associated with visibilities
    weights : float or array of float, shape (M, ..., N), optional
        Visibility weights (positive real numbers)
    ref_ant : int, optional
        Reference antenna for which phase will be forced to 0.0. Alternatively,
        if *ref_ant* is -1, the median gain phase will be 0.
    init_gain : array of complex, shape(num_ants,), optional
        Initial gain vector (all equal to 1.0 by default)
    num_iters : int, optional
        Number of iterations
    conv_thresh : float, optional
        Convergence threshold (max relative l_2 norm of change in gain vector)

    Returns
    -------
    gains : array of complex, shape (M, ..., num_ants)
        Complex gains per antenna

    Notes
    -----
    The model visibilities are assumed to be 1, implying a point source model.
    The algorithm is iterative but should converge in a small number of
    iterations (10 to 30).

    References
    ----------
    .. [1] Salvini, Wijnholds, "Fast gain calibration in radio astronomy using
       alternating direction implicit methods: Analysis and applications," 2014,
       preprint at `<http://arxiv.org/abs/1410.2101>`_
    """
    if init_gain is None:
        init_gain = np.ones(num_ants, rawvis.dtype)
    if init_gain.shape[-1] != num_ants:
        raise ValueError('initial gains have wrong length {} for number of antennas {}'.format(
            init_gain.shape[-1], num_ants))
    if weights is None or np.isscalar(weights):
        # Any scalar weight is the same as an unweighted problem, and we don't
        # want a double-precision scalar weight to cause the result to be
        # upgraded to double precision.
        weights = rawvis.real.dtype.type(1.0)
    if not all(0 <= x < num_ants for x in corrprod_lookup.flat):
        raise ValueError('invalid antenna index in corrprod_lookup')
    if ref_ant >= num_ants:
        raise ValueError('invalid reference antenna')
    weights = np.broadcast_to(weights, rawvis.shape)
    return _stefcal_gufunc(rawvis, corrprod_lookup[:, 0], corrprod_lookup[:, 1], weights,
                           ref_ant, init_gain, num_iters, conv_thresh)


# --------------------------------------------------------------------------------------------------
# --- Calibration helper functions
# --------------------------------------------------------------------------------------------------

def g_from_K(chans, K):
    g_array = np.ones(K.shape+(len(chans),), dtype=np.complex)
    for i, c in enumerate(chans):
        g_array[:, :, i] = np.exp(1.0j * 2. * np.pi * K * c)
    return g_array


def ants_from_bllist(bllist):
    return len(set([item for sublist in bllist for item in sublist]))


def best_refant(data, corrprod_lookup, chans):
    """
    Determine antenna whose FFT has the maximum peak to noise ratio (pnr) by taking the median
    pnr of the FFT over all baselines to each antenna

    Parameters:
    -----------
    data : :class:`np.ndarray`, array of complex, shape(n_chans, n_pols, n_bls)
        visibility data
    corrprod_lookup : list of int, shape (2, n_bls)
        list of antenna pairs for selected baselines
    chans : :class:`np.ndarray`
        channel frequencies in Hz
    Returns:
    --------
    int : index of antenna with the maximum sum of PNR over all baselines
    """
    # Detect position of fft peak
    ft_vis = scipy.fftpack.fft(data, axis=0)
    k_arg = np.argmax(np.abs(ft_vis), axis=0)
    # Shift data so that peak of fft is always positioned at the first index of the array
    n_chans, n_pols, n_bls = data.shape
    index = np.array([np.roll(range(n_chans), -n) for n in k_arg.ravel()])
    index = index.T.reshape(ft_vis.shape)
    ft_vis = np.take_along_axis(ft_vis, index, axis=0)

    # Calculate the height of the peak and the standard deviation away from the peak
    peak = np.max(np.abs(ft_vis), axis=0)
    chan_slice = np.s_[len(chans)//2-len(chans)//4:len(chans)//2+len(chans)//4]
    mean = np.mean(np.abs(ft_vis[chan_slice]), axis=0)
    # Add 1e-9 to avoid divide by zero errors
    std = np.std(np.abs(ft_vis[chan_slice]), axis=0) + 1e-9

    # Calculate the median of the pnr for all baselines per antenna
    num_ants = ants_from_bllist(corrprod_lookup)
    med_pnr_ants = np.zeros(num_ants)
    try:
        for a in range(num_ants):
            mask = [np.any(b == a) for b in corrprod_lookup]
            pnr = (peak[..., mask] - mean[..., mask]) / std[..., mask]
            med_pnr_ants[a] = np.median(pnr)
    except IndexError:
        logger.warn('Failed to calculate pnr for antenna %r. Ref ant selection may be compromised.',
		    a, exc_info=True)
    return np.argmax(med_pnr_ants)


def g_fit(data, weights, corrprod_lookup,  g0=None, refant=0, **kwargs):
    """
    Fit complex gains to visibility data.

    Parameters
    ----------
    data : visibility data, array of complex, shape(num_sol, num_chans, baseline)
    weights : weight data, array of real, shape(num_sol, num_chans, baseline)
    g0 : array of complex, shape(num_ants) or None
    corrprod_lookup : antenna mappings, for first then second antennas in bl pair
    refant : reference antenna

    Returns
    -------
    g_array : Array of gain solutions, shape(num_sol, num_ants)
    """
    num_ants = ants_from_bllist(corrprod_lookup)
    return stefcal(data, num_ants, corrprod_lookup, weights,
                   ref_ant=refant, init_gain=g0, **kwargs)


def k_fit(data, weights, corrprod_lookup, chans, refant=0, cross=True, chan_sample=1):
    """
    Fit delay (phase slope across frequency) to visibility data.
    If corrprod_lookup is xc, i.e. all baselines have two different antenna indices,
    it will decompose the delays into solutions per antenna.
    Else it solves for a delay per element in final axis.

    Parameters
    ----------
    data : array of complex, shape (num_chans, num_pols, num_baselines)
        Visibility data (may contain NaNs indicating completely flagged data)
    weights : array of real, shape (num_sol, num_chans, baseline)
        Weight data, must be zero for flagged data and NaNed data
    corrprod_lookup : array of int, shape (num_baselines/num_ant, 2)
        Pairs of antenna indices associated with each baseline
    chans : sequence of float, length num_chans
        Channel frequencies in Hz
    refant : int, optional
        Reference antenna index
    cross: bool, optional
        default assume data are cross-correlations per baseline, else assume data are
        auto-correlations per antenna
    chan_sample : int, optional
        Subsample channels by this amount, i.e. use every n'th channel in fit

    Returns
    -------
    kdelay : array of float, shape (num_pols, num_ants/data.shape[-1])
        Delay solutions per antenna, in seconds (NaN if fit failed)
    """
    # OVER THE TOP NOTE:
    # This solver was written to deal with extreme gains, which can wrap many
    # many times across the solution window, to the extent that solving for a
    # per-channel per-antenna bandpass, then fitting a line to the per-antenna
    # bandpass breaks down. It breaks down because the phase unwrapping is
    # unreliable when the phases (a) wrap so extremely, and (b) are also
    # noisy.
    #
    # This situation was adressed by first doing a coarse delay fit through an
    # FFT, for each antenna to the reference antenna. Then the course delay is
    # applied, and a secondary linear phase fit is performed with the corrected
    # data. With the coarse delays removed there should be little, or minimal
    # wrapping, and the unwrapping prior to the linear fit usually works.
    # BUT:
    # * This algorithm is most effective if the delays are of similar scales.
    #   How to ensure this is to choose a reference antenna that has high delays
    #   on as many baselines as possible.  Perhaps this could be done by looking
    #   through all of the FFT results, grouped by antenna, to find the highest
    #   set of delays?
    # * In the future world where not every delay fit needs to deal with
    #   extreme delay values, maybe we could separate out the
    #   bandpass-linear-phase-fit component of this algorithm, and use just that
    #   for delay solutions we know won't be too extreme. Then this extreme delay
    #  solver could solve for the coarse delay then apply it and call the fine
    #  delay bandpass-linear-phase-fit solver.

    # -----------------------------------------------------
    chans = np.asarray(chans, dtype=np.float32)
    # if channel sampling is specified, thin down the data and channel list
    if chan_sample != 1:
        data = data[::chan_sample, ...]
        weights = weights[::chan_sample, ...]
        chans = chans[::chan_sample]

    # set up parameters
    num_pol = data.shape[-2] if len(data.shape) > 2 else 1
    chan_spacing = chans[1] - chans[0]

    # -----------------------------------------------------
    # NOTE: I know that iterating over polarisation is horrible, but I ran out
    # of time to fix this up I initially just assumed we would always have two
    # polarisations, but shouldn't make that assumption especially for use of
    # calprocs offline on an h5 file
    kdelay = []
    for p in range(num_pol):
        pol_data = data[:, p, :] if len(data.shape) > 2 else data
        pol_weights = weights[:, p, :] if len(weights.shape) > 2 else weights
        # Suppress NaNs by setting them to zero. This masking step broadens
        # the delay peak in Fourier space and potentially introduces spurious
        # sidelobes if severe, like a dirty image suffering from poor uv coverage.
        good_pol_data = np.nan_to_num(pol_data)
        # -----------------------------------------------------
        # FT to find visibility space delays
        # NOTE: This is a bit inefficient at the moment as the FFT for al
        # baselines is calculated but only the baselines to the reference
        # antenna are used for the coarse delay
        # Also note that scipy.fftpack will do a single-precision FFT given
        # single-precision input, unlike np.fft.fft which converts it to
        # double precision.
        # NB: The coarse delay part assumes regularly spaced frequencies in chans
        ft_vis = scipy.fftpack.fft(good_pol_data, axis=0)
        # get index of FT maximum
        k_arg = np.argmax(np.abs(ft_vis), axis=0)
        # calculate vis space K from FT sample frequencies
        vis_k = np.float32(np.fft.fftfreq(ft_vis.shape[0], chan_spacing)[k_arg])

        # test whether data is xc
        if cross:
            # now determine per-antenna K values
            num_ants = ants_from_bllist(corrprod_lookup)
            coarse_k = np.zeros(num_ants, np.float32)
            for ai in range(num_ants):
                k = vis_k[..., (corrprod_lookup == (ai, refant)).all(axis=1)]
                if k.size > 0:
                    coarse_k[ai] = np.squeeze(k)
                k = vis_k[..., (corrprod_lookup == (refant, ai)).all(axis=1)]
                if k.size > 0:
                    coarse_k[ai] = np.squeeze(-1.0 * k)

            # apply coarse K values to the data and solve for bandpass
            # The baseline delay is calculated as delay(ant2) - delay(ant1)
            bl_delays = np.diff(coarse_k[corrprod_lookup])
            good_pol_data *= np.exp(2j * np.pi * np.outer(chans, bl_delays))
            bpass = stefcal(good_pol_data, num_ants, corrprod_lookup, pol_weights,
                            num_iters=100, ref_ant=refant, init_gain=None)

            # set weight for NaNed stefcal outputs to zero
            bpass_weights = (~np.isnan(bpass)).astype(np.float32)

        # else assume solution is ac
        else:
            coarse_k = vis_k
            bpass = pol_data * np.exp(-2j * np.pi * np.outer(chans, coarse_k))
            bpass_weights = pol_weights

        # Find slope of the residual bandpass, per antenna (defaults to NaN)
        delta_k = np.full_like(coarse_k, np.nan)
        for i, (bp, bp_weights) in enumerate(zip(bpass.T, bpass_weights.T)):
            # np.unwrap falls over in the case of bad RFI - robustify this later
            # np.unwrap might not unwrap correctly with zeroed data, thus mask
            # zero values in these routines (but skip if everything is masked)
            valid = bp_weights > 0
            if any(valid):
                bp_phase = np.unwrap(np.angle(bp[valid]), discont=1.9 * np.pi)
                freqs = chans[valid]
                A = np.array([freqs, np.ones(len(freqs))])
                # trick np.linalg.lstsq into performing a weighted least squares (WLS) fit
                # multiply each term in the fit by w to minimize w**2(bp_phase-A.T)**2
                # for a WLS fit set w**2 = bp_weights
                w = np.sqrt(bp_weights[valid])
                delta_k[i] = np.linalg.lstsq((w * A).T, w * bp_phase, None)[0][0] / (2. * np.pi)

        kdelay.append(coarse_k + delta_k)

    return np.atleast_2d(kdelay)


def normalise_complex(x):
    """
    Calculate normalisation factor for a complex array across the 0-axis to center
    phase of data on zero and scale the average amplitude to one. If the 0-axis
    is all NaN or all zero, the normalisation factor is given as 1+0j.

    Parameters
    ----------
    x : :class:`np.ndarray`
        data to be normalised

    Returns
    -------
    :class:`np.ndarray`
        normalisation factor, complex
    """
    # suppress warnings related to all-NaN and all-zero values on the 0-axis
    # by replacing columns which consist of all NaNs and/or zeros with all ones.
    all_nan = np.all((np.isnan(x)) ^ (x == 0), axis=0)
    all_nan = np.broadcast_to(all_nan, x.shape)
    valid_x = np.where(all_nan, 1.0, x)

    angle = np.angle(valid_x)
    base_angle = np.nanmin(angle, axis=0) - np.pi
    # angle relative to base_angle, wrapped to range [0, 2pi], with
    # some data point sitting at pi.
    rel_angle = np.fmod(angle - base_angle, 2 * np.pi)
    mid_angle = np.nanmean(rel_angle, axis=0) + base_angle
    centre_rotation = np.exp(-1.0j * mid_angle)
    n_valid = np.sum(~np.isnan(valid_x), axis=0, dtype=np.float32)
    average_amplitude = np.nansum(np.absolute(valid_x), axis=0) / n_valid
    norm_factor = centre_rotation / average_amplitude

    return norm_factor


def asbool(arr):
    """View an array as boolean.

    If possible it simply returns a view, otherwise a copy. It works on both
    dask and numpy arrays.
    """
    if arr.dtype in (np.uint8, np.int8, np.bool_):
        return arr.view(np.bool_)
    else:
        return arr.astype(np.bool_)


def solint_from_nominal(solint, dump_period, num_times):
    """
    Given nominal solint, modify it by up to 20percent to optimally fit the scan length
    and dump period. Times are assumed to be contiguous.

    Parameters
    ----------
    solint      : nominal solint
    dump_period : dump period of the data
    num_times   : number of time dumps in the scan

    Returns
    -------
    nsolint     : new optimal solint
    dumps_per_solint : number of dump periods in a solution interval
    """

    if dump_period > solint:
        # case of solution interval that is shorter than dump time
        dumps_per_solint = 1
    elif dump_period * num_times < solint:
        # case of solution interval that is longer than scan
        dumps_per_solint = num_times
    else:
        # requested number of dumps per nominal solution interval
        req_dumps_per_solint = solint / dump_period

        # range for searching: nominal solint +-20%
        dumps_per_solint_low = int(np.round(req_dumps_per_solint * 0.8))
        dumps_per_solint_high = int(np.round(req_dumps_per_solint * 1.2))
        solint_check_range = np.arange(dumps_per_solint_low,
                                       dumps_per_solint_high + 1).astype(np.int)

        # compute size of final partial interval (in dumps)
        tail = (num_times - 1) % solint_check_range + 1
        delta = solint_check_range - tail

        # choose a solint to minimise the change in the final fractional solution interval
        solint_index = np.argmin(delta)
        logger.debug('solint index: {0}'.format(solint_index,))
        dumps_per_solint = solint_check_range[solint_index]
    return dumps_per_solint * dump_period, dumps_per_solint


# --------------------------------------------------------------------------------------------------
# --- Baseline ordering
# --------------------------------------------------------------------------------------------------

def get_ant_bls(pol_bls_ordering):
    """
    Given baseline list with polarisation information, return pure antenna list
    e.g. from
    [['ant0h','ant0h'],['ant0v','ant0v'],['ant0h','ant0v'],['ant0v','ant0h'],['ant1h','ant1h']...]
    to [['ant0,ant0'],['ant1','ant1']...]
    """

    # get antenna only names from pol-included bls orderding
    ant_bls = np.array([[a1[:-1], a2[:-1]] for a1, a2 in pol_bls_ordering])
    ant_dtype = ant_bls[0, 0].dtype
    # get list without repeats (ie no repeats for pol)
    #     start with array of empty strings, shape (num_baselines x 2)
    bls_ordering = np.empty([len(ant_bls) / 4, 2], dtype=ant_dtype)

    # iterate through baseline list and only include non-repeats
    #   I know this is horribly non-pythonic. Fix later.
    bls_ordering[0] = ant_bls[0]
    bl = 0
    for c in ant_bls:
        if not np.all(bls_ordering[bl] == c):
            bl += 1
            bls_ordering[bl] = c

    return bls_ordering


def get_pol_bls(bls_ordering, pol):
    """
    Given baseline ordering and polarisation ordering, return full baseline-pol ordering array

    Inputs:
    -------
    bls_ordering
        list of correlation products without polarisation information, string shape(nbl,2)
    pol
        list of polarisation pairs, string shape (npol_pair, 2)

    Returns:
    --------
    pol_bls_ordering : :class:`np.ndarray`
        correlation products without polarisation information, shape(nbl*4, 2)
    """
    pol_ant_dtype = np.array(bls_ordering[0][0] + 'h').dtype
    nbl = len(bls_ordering)
    pol_bls_ordering = np.empty([nbl * len(pol), 2], dtype=pol_ant_dtype)
    for i, p in enumerate(pol):
        for b, bls in enumerate(bls_ordering):
            pol_bls_ordering[nbl * i + b] = bls[0] + p[0], bls[1] + p[1]
    return pol_bls_ordering


def get_reordering(antlist, bls_ordering, output_order_bls=None, output_order_pol=None):
    """
    Determine reordering necessary to change given bls_ordering into desired ordering

    Inputs:
    -------
    antlist : list of antennas, string shape(nants), or string of csv antenna names
    bls_ordering : list of correlation products, string shape(nbl,2)
    output_order_bls : desired baseline output order, string shape(nbl,2), optional
    output_order_pol : desired polarisation output order, string shape(npolcorr,2), optional

    Returns:
    --------
    ordering
        ordering array necessary to change given bls_ordering into desired
        ordering, numpy array shape(nbl*4, 2)
    bls_wanted
        ordering of baselines, without polarisation, list shape(nbl, 2)
    pol_order
        ordering of polarisations, list shape (4, 2)

    """
    # convert antlist to list, if it is a csv string
    if isinstance(antlist, str):
        antlist = antlist.split(',')
    # convert bls_ordering to a list, if it is not a list (e.g. np.ndarray)
    if not isinstance(bls_ordering, list):
        bls_ordering = bls_ordering.tolist()

    # get current antenna list without polarisation
    bls_ordering_nopol = [[b[0][0:4], b[1][0:4]] for b in bls_ordering]
    # find unique elements
    unique_bls = []
    for b in bls_ordering_nopol:
        if b not in unique_bls:
            unique_bls.append(b)

    # determine polarisation ordering
    if output_order_pol is not None:
        pol_order = output_order_pol
    else:
        # get polarisation products from current antenna list
        bls_ordering_polonly = [[b[0][4], b[1][4]] for b in bls_ordering]
        # find unique pol combinations
        unique_pol = []
        for pbl in bls_ordering_polonly:
            if pbl not in unique_pol:
                unique_pol.append(pbl)
        # put parallel polarisations first
        pol_order = []
        if len(bls_ordering_polonly) > 2:
            for pp in unique_pol:
                if pp[0] == pp[1]:
                    pol_order.insert(0, pp)
            pol_order.append([pol_order[0][0], pol_order[1][0]])
            pol_order.append([pol_order[1][0], pol_order[0][0]])
    npol = len(pol_order)

    if output_order_bls is None:
        # default output order is XC then AC
        bls_wanted = [b for b in unique_bls if b[0] != b[1]]

        # number of baselines including autocorrelations
        nants = len(antlist)
        nbl_ac = nants * (nants + 1) / 2

        # add AC to bls list, if necessary
        # assume that AC are either included or not (no partial states)
        if len(bls_ordering) == (nbl_ac * npol):
            # data include XC and AC
            bls_wanted.extend([b for b in unique_bls if b[0] == b[1]])

        bls_pol_wanted = get_pol_bls(bls_wanted, pol_order)
    else:
        bls_pol_wanted = get_pol_bls(output_order_bls, pol_order)
    # note: bls_pol_wanted must be an np array for list equivalence below

    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering = np.array([np.all(bls_ordering == bls, axis=1).nonzero()[0][0]
                         for bls in bls_pol_wanted])

    # how to use this:
    # print bls_ordering[ordering]
    # print bls_ordering[ordering].reshape([4,nbl,2])
    return ordering, bls_wanted, pol_order


def get_reordering_nopol(antlist, bls_ordering, output_order_bls=None):
    """
    Determine reordering necessary to change given bls_ordering into desired ordering

    Inputs:
    -------
    antlist : list of antennas, string shape(nants), or string of csv antenna names
    bls_ordering : list of correlation products, string shape(nbl,2)
    output_order_bls : desired baseline output order, string shape(nbl,2), optional

    Returns:
    --------
    ordering
        ordering array necessary to change given bls_ordering into desired
        ordering, numpy array, shape(nbl*4, 2)
    bls_wanted
        ordering of baselines, without polarisation, numpy array, shape(nbl, 2)

    """
    # convert antlist to list, if it is a csv string
    if isinstance(antlist, str):
        antlist = antlist.split(',')
    # convert bls_ordering to a list, if it is not a list (e.g. np.ndarray)
    if not isinstance(bls_ordering, list):
        bls_ordering = bls_ordering.tolist()

    # find unique elements
    unique_bls = []
    for b in bls_ordering:
        if b not in unique_bls:
            unique_bls.append(b)

    # defermine output ordering:
    if output_order_bls is None:
        # default output order is XC then AC
        bls_wanted = [b for b in unique_bls if b[0] != b[1]]
        # add AC to bls list
        bls_wanted.extend([b for b in unique_bls if b[0] == b[1]])
    else:
        # convert to list in case it is not a list
        bls_wanted = output_order_bls
    # note: bls_wanted must be an np array for list equivalence below
    bls_wanted = np.array(bls_wanted)

    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering = np.array([np.all(bls_ordering == bls, axis=1).nonzero()[0][0]
                        for bls in bls_wanted])
    # how to use this:
    # print bls_ordering[ordering]
    # print bls_ordering[ordering].reshape([4,nbl,2])
    return ordering, bls_wanted


def get_bls_lookup(antlist, bls_ordering):
    """
    Get correlation product antenna mapping

    Inputs:
    -------
    antlist : list of antenna names, string shape (nant)
    bls_ordering : list of correlation products, string shape(nbl,2)

    Returns:
    --------
    corrprod_lookup : lookup table of antenna indices for each baseline, shape(nbl,2)
    """

    # make polarisation and corr_prod lookup tables (assume this doesn't change
    # over the course of an observaton)
    antlist_index = dict([(antlist[i], i) for i in range(len(antlist))])
    return np.array([[antlist_index[a1[0:4]], antlist_index[a2[0:4]]] for a1, a2 in bls_ordering])


# --------------------------------------------------------------------------------------------------
# --- Simulation
# --------------------------------------------------------------------------------------------------

def fake_vis(shape=(7,), gains=None, noise=None, random_state=None):
    """Create fake point source visibilities, corrupted by given or random gains. The
    final dimension of `shape` corresponds to the number of antennas.
    """
    # create antenna lists
    if isinstance(shape, (int, long)):
        shape = (shape,)
    shape = tuple(shape)
    nants = shape[-1]

    antlist = range(nants)
    list1 = np.array([])
    for a, i in enumerate(range(nants - 1, 0, -1)):
        list1 = np.hstack([list1, np.ones(i) * a])
    list1 = np.hstack([list1, antlist])
    list1 = np.int_(list1)

    list2 = np.array([], dtype=np.int)
    mod_antlist = antlist[1:]
    for i in range(0, len(mod_antlist)):
        list2 = np.hstack([list2, mod_antlist[:]])
        mod_antlist.pop(0)
    list2 = np.hstack([list2, antlist])

    # create fake gains, if gains are not given as input
    if random_state is None:
        random_state = np.random
    if gains is None:
        gains = random_state.random_sample(shape) + 1j * random_state.random_sample(shape)
    else:
        assert shape == gains.shape

    # create fake corrupted visibilities
    nbl = nants * (nants + 1) // 2
    vis = np.ones(tuple(shape[:-1]) + (nbl,), gains.dtype)
    # corrupt vis with gains
    for i, (a, b) in enumerate(zip(list1, list2)):
        vis[..., i] *= gains[..., a] * gains[..., b].conj()
    # if requested, corrupt vis with noise
    if noise is not None:
        vis_noise = random_state.standard_normal(vis.shape) * noise
        vis = vis + vis_noise

    # return useful info
    bl_pair_list = np.column_stack([list1, list2])
    return vis, bl_pair_list, gains

# --------------------------------------------------------------------------------------------------
# --- Averaging
# --------------------------------------------------------------------------------------------------


def wavg_full_f(data, flags, weights, chanav, threshold=0.8):
    """
    Perform weighted average of data, flags and weights,
    applying flags, over axis -3, for specified number of channels

    Parameters
    ----------
    data : :class:`np.ndarray`
        complex (ntimes, nchans, npols, nbls)
    flags : :class:`np.ndarray`
        int (ntimes, nchans, npols, bls)
    weights : :class:`np.ndarray`
        real (ntimes, nchans, npols, bls)
    chanav : int
        number of channels over which to average, integer
    threshold : float
        if fraction of flags in the input data
        exceeds threshold then set output flag to True, else False

    Returns
    -------
    av_data : :class:`np.ndarray`
        complex (..., av_chans, npols, nbls), weighted average of data
    av_flags : :class:`np.ndarray`
        bool (..., av_chans, npols, nbls), weighted average of flags
    av_weights : :class:`np.ndarray`
        real (..., av_chans, npols, nbls), weighted average of weights
    """
    # ensure chanav is an integer
    chanav = np.int(chanav)
    inc_array = range(0, data.shape[-3], chanav)

    flagged_weights = np.where(flags, weights.dtype.type(0), weights)
    weighted_data = data * flagged_weights

    # Clear the elements that have a nan anywhere
    isnan = np.isnan(weighted_data)
    weighted_data = np.where(isnan, weighted_data.dtype.type(0), weighted_data)
    flagged_weights = np.where(isnan, flagged_weights.dtype.type(0), flagged_weights)
    av_weights = np.add.reduceat(flagged_weights, inc_array, axis=-3)
    with np.errstate(divide='ignore', invalid='ignore'):
        av_data = np.add.reduceat(weighted_data, inc_array, axis=-3) / av_weights
    n_flags = np.add.reduceat(asbool(flags), inc_array, axis=-3)
    n_samples = np.add.reduceat(np.ones(flagged_weights.shape), inc_array, axis=-3)
    av_flags = n_flags >= n_samples * threshold

    return av_data, av_flags, av_weights


@numba.jit(nopython=True)
def _wavg_flags_f(flags, chanav, excise):
    """Implementation of wavg_flags_f.

    This numba-accelerated version handles only the case where there are
    4 dimensions and we're averaging along axis 1.
    """
    n_time, n_channels, n_pol, n_baselines = flags.shape
    good = np.empty((n_pol, n_baselines), np.bool_)
    merge_good = np.empty((n_pol, n_baselines), flags.dtype)
    merge_all = np.empty((n_pol, n_baselines), flags.dtype)
    out_channels = n_channels // chanav
    out = np.empty((n_time, out_channels, n_pol, n_baselines), flags.dtype)
    for t in range(n_time):
        for oc in range(out_channels):
            good.fill(False)
            merge_good.fill(0)
            merge_all.fill(0)
            start_channel = oc * chanav
            end_channel = start_channel + chanav
            for ic in range(start_channel, end_channel):
                for p in range(n_pol):
                    for b in range(n_baselines):
                        flag = flags[t, ic, p, b]
                        is_good = not excise[t, ic, p, b]
                        merge_all[p, b] |= flag
                        merge_good[p, b] |= is_good * flag
                        good[p, b] |= is_good
            for p in range(n_pol):
                for b in range(n_baselines):
                    out[t, oc, p, b] = merge_good[p, b] if good[p, b] else merge_all[p, b]
    return out


def wavg_flags_f(flags, chanav, excise, axis):
    """Combine flags across frequencies.

    This function is designed to mirror flag merging done by ingest. For each
    visibility, there is a flag bitmask in `flags` and a corresponding boolean
    in `excise`. The combined flags for a set of visibilities is
    - The bitwise OR of the flags for which the excise bit is clear, if there
      is at least one such; otherwise
    - The bitwise OR of all the flags.

    The implementation currently supports at most one axis before `axis` and at
    most two after it (which caters for the standard time, frequency, pol,
    baseline axis ordering).

    Parameters
    ----------
    flags : :class:`np.ndarray`
        uint8 flag masks
    chanav : int
        number of channels over which to average, integer. It must exactly
        divide into the number of channels.
    excise : class:`np.ndarray`
        boolean corresponding to each element of `flags`
    axis : int
        Axis along which to apply

    Returns
    -------
    av_flags : :class:`np.ndarray`
        uint8 combined flags

    Raises
    ------
    ValueError
        if `flags` and `excise` have different shapes
    ValueError
        if chanav doesn't divide into the length of the chosen axis
    np.AxisError
        if `axis` is invalid
    NotImplementedError
        if there are more than two axes after `axis` or more than one axis
        before it
    """
    flags = np.asarray(flags)
    excise = np.asarray(excise)
    if excise.shape != flags.shape:
        raise ValueError('excise must have the same shape as flags')
    if axis < 0:
        axis += flags.ndim
    if axis < 0 or axis >= flags.ndim:
        raise np.AxisError('axis {} is out of bounds for array of dimension {}'
                           .format(axis, flags.ndim))
    if axis > 1 or flags.ndim - axis > 3:
        raise NotImplementedError('axis {} is not currently supported'.format(axis))
    if flags.shape[axis] % chanav != 0:
        raise ValueError('chanav {} does not divide length {}'
                         .format(chanav, flags.shape[axis]))

    pre = 1 - axis
    post = axis + 3 - flags.ndim
    for i in range(pre):
        flags = flags[np.newaxis]
        excise = excise[np.newaxis]
    for i in range(post):
        # Insert the new axes immediately after channels, rather than at the
        # end, because we want the inner loops to run over longer axes.
        flags = flags[:, :, np.newaxis]
        excise = excise[:, :, np.newaxis]
    out_flags = _wavg_flags_f(flags, chanav, excise)
    # Now reverse the axis removals
    for i in range(post):
        out_flags = out_flags[:, :, 0]
    for i in range(pre):
        out_flags = out_flags[0]
    return out_flags


# --------------------------------------------------------------------------------------------------
# --- General helper functions
# --------------------------------------------------------------------------------------------------

def arcsec_to_rad(angle):
    """
    Convert angle in arcseconds to angle in radians
    """
    return np.deg2rad(angle / 60. / 60.)
