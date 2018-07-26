import os
import numpy as np
from . import plotting

def derive_metrics(vis,weights,flags,s,ts,parameters,deg=3,plot=False,metric_func=np.mean,metric_tol=10):

    """Derive quality metrics for the calibrated data of a bandpass calibrator. This compares low-order polynomial fits
    to the calibrated bandpass (amplitude and phase), comparing each timestamp of a single baseline to a reference timestamp
    (the first timestamp), and derives a reduced chi squared value for each timestamp.

    Paramaters:
    -----------
    vis : class:`np.ndarray`
        The visibility data of the bandpass calibrator.
    weights : class:`np.ndarray`
        The visbilitiy weights of the bandpass calibrator.
    flags : class:`np.ndarray`
        The flags of the bandpass calibrator.
    s : class: `Scan`
        The scan.
    ts : :class:`katsdptelstate.TelescopeState`
        Telescope state, scoped to the cal namespace within the capture block.
    parameters : dict
        The pipeline parameters
    deg : int, optional
        Degree of polynomials to fit.
    plot : bool, optional
        Plot the data and fitting?
    metric_func : function, optional
        Function to apply to the distribution of residuals, which is np.mean by default.
    metric_tol : float, optional
        If the value of metric_func is less than this tolerance, flag the metric as bad."""

    # get good axis limits from data for plotting amplitude
    inc = vis.shape[1] // 4
    high = int(np.nanmedian(vis[:,0:inc,:,:].real)*1.3)
    low = int(np.nanmedian(vis[:,-inc:-1,:,:].real)*0.75)
    aylim=(low,high)

    # compute bandpass data quality metrics
    amp_resid,amp_polys = bandpass_metrics(vis.compute(),weights.compute(),flags.compute(),dtype='Amp',plot=plot,plot_poly=True,plot_ref=True,deg=deg,ylim=aylim)
    phase_resid,phase_polys = bandpass_metrics(vis.compute(),weights.compute(),flags.compute(),dtype='Phase',plot=plot,plot_poly=True,plot_ref=True,deg=deg,ylim=(-10,10))

    # take single bandpass metric as worst
    BP_amp_metric = metric_func(amp_resid[:,:,:,3])
    BP_amp_metric_loc = np.where(amp_resid == BP_amp_metric)
    description = metric_description(s, ts, parameters, BP_amp_metric_loc)
    add_telstate_metric('bp',BP_amp_metric,'amp',description,tolerance)

    BP_phase_metric = metric_func(phase_resid[:,:,:,3])
    BP_phase_metric_loc = np.where(phase_resid == BP_phase_metric)
    description = metric_description(s, ts, parameters, BP_phase_metric_loc)
    add_telstate_metric('bp',BP_phase_metric,'phase',description,tolerance)


def poly_residual(xdata,ydata,weights=None,flags=None,deg=3,plot=True,fig=None,xlab=None,ylab=None,xlim=None,ylim=None,logy=False):

    """Return residual statistics based on fitting a polynomial to a distribution.

    Paramaters:
    -----------
    xdata : class:`np.ndarray`
        The x axis data
    ydata : class:`np.ndarray`
        The y axis data

    weights : class:`np.ndarray`, optional
        The x axis weights.
    flags : class:`np.ndarray`, optional
        The x axis flags.
    deg : int, optional
        The degree of polynomial to fit
    plot : bool, optional
        Plot figure?
    fig : class:`plt.figure`, optional
        Use this existing figure.
    xlab : string, optional
        The x axis label
    ylab : string, optional
        The y axis label
    xlim : tuple, optional
        Limits to put on the x axis
    ylim : tuple, optional
        Limits to put on the y axis
    logy : bool
        Log the y axis?

    Returns:
    --------
    red_chi_sq,nmad,nmad_model : class:`np.array`
        red_chi_sq : float
            The reduced chi squared
        nmad : float
            The normalised median absolute deviation
        nmad_model : float
            The normalised median absolute deviation from the model
        None : NoneType
            Place holder for reduced chi squared metric"""

    indices = ~np.isnan(xdata) & ~np.isnan(ydata) & (flags == 0)
    if weights is not None:
        indices = indices & ~(weights == 0)
        w = 1.0/weights[indices]

    x = xdata[indices]
    y = ydata[indices]

    xlin = np.arange(np.min(x),np.max(x))
    poly_paras = np.polyfit(x,y,deg,w=w)
    poly = np.poly1d(poly_paras)
    fit = poly(xlin)

    chi_sq=np.sum((y-poly(x))**2)
    DOF=len(x)-(deg+1) #polynomial has deg+1 free parameters

    red_chi_sq = chi_sq/DOF
    nmad = np.median(np.abs(y-np.median(y)))/0.6745
    nmad_model = np.median(np.abs(y-poly(x)))/0.6745

    if plot:
        plotting.plot_bandpass(x,y,xlab,ylab,fig=fig,logy=logy)

    return np.array([red_chi_sq,nmad,nmad_model,None]),poly_paras


def compare_poly(xdata,poly1,poly2,plot=True,fig=None,plot_poly=True,plot_ref=True):

    """Compare two polynomials fits to x data by returning the reduced chi squared.

    Paramaters:
    -----------
    xdata : class:`np.ndarray`
        The x axis data
    poly1 : class `np.polyfit`
        The first polynomial.
    poly2 : class `np.polyfit`
        The reference polynomial.
    plot : bool, optional
        Plot figure?
    fig : class:`plt.figure`, optional
        Use this existing figure.
    plot_poly : bool, optional
        Overlay the polynomial fit on the figure?
    plot_ref : bool, optional
        Overlay the reference polynomial fit on the figure?"""

    xdata = xdata[~np.isnan(xdata)]
    xlin = np.arange(np.min(xdata),np.max(xdata))

    fit1 = np.poly1d(poly1)
    fit2 = np.poly1d(poly2)

    chi_sq=np.sum((fit1(xlin)-fit2(xlin))**2)
    DOF=len(xdata)-len(poly1)-len(poly2)
    red_chi_sq = chi_sq/DOF

    if plot:
        plotting.plot_polys(xlin,fit1,fit2,fig=fig)

    return red_chi_sq

def bandpass_metrics(vis,weights,flags,dtype='Amplitude',freqs=None,deg=3,plot=False,plot_poly=True,plot_ref=True,infig=None,ylim=None):

    """Calculate the bandpass metrics.

    Paramaters:
    -----------
    vis : class:`np.ndarray`
        The visibilities. Assumed to have shape (ntimes, nchannels, npols, nbaselines).
    weights : class:`np.ndarray`
        The visibility weights. Assumed to have shape (ntimes, nchannels, npols, nbaselines).
    flags : class:`np.ndarray`
        The visibility flags. Assumed to have shape (ntimes, nchannels, npols, nbaselines).
    dtype : str, optional
        'phase' | 'amplitude' (case-insensitive).
    freqs : class:`np.ndarray`, optional
        A list of frequencies for each channel.
    deg : int, optional
        Degree of polynomial to fit.
    plot : bool, optional
        Write a matplotlib figure?
    plot_poly : bool, optional
        Overlay the polynomial on the plot?
    plot_ref : bool, optional
        Overlay the reference polynomial on the plot?
    infig : class:`plt.figure`, optional
        Use this existing figure.
    ylim : tuple, optional
        Limits to put on the y axis

    Returns:
    --------
    residuals : class `np.array`
        The residual metrics of the fit polynomials, with shape
        (ntimestamps, npols, nbaselines, 4 [red_chi_sq,nmad,nmad_model,red_chi_sq_poly])
    polys : class `np.array`
        Array of `np.polyfit` objects, representing the polynomials fit to each element of input data, with shape
        (ntimestamps, npols, nbaselines, deg+1)"""

    #make smaller for testing / plotting
    vis = vis[:,:,0:1,0:5]
    weights = weights[:,:,0:1,0:5]
    flags = flags[:,:,0:1,0:5]

    residuals = np.ndarray(shape=(vis.shape[0],vis.shape[2],vis.shape[3],4))
    polys = np.ndarray(shape=(vis.shape[0],vis.shape[2],vis.shape[3],deg+1))

    if dtype.lower() in ['amplitude','amp']:
        vis = vis.real #np.absolute(data)
        logy = False
    elif dtype.lower() == 'phase':
        vis = vis.imag #np.angle(data,deg=True)
        logy = False
    else:
        print "'{0}' not recognised type. Use 'amplitude' or 'phase'.".format(dtype)
        return

    for time in range(vis.shape[0]):
        for pol in range(vis.shape[2]):
            for bline in range(vis.shape[3]):

                data = vis[time,:,pol,bline]
                data_w = weights[time,:,pol,bline]
                data_f = flags[time,:,pol,bline]

                if freqs is None:
                    all_chan = np.arange(vis.shape[1])
                    xlab = 'Channel'
                else:
                    all_chan = np.arange(freq[0],freq[-1],1) / 1.0e6
                    xlab = 'Frequency (MHz)'

                if plot and infig is None:
                    fig = plt.figure(figsize=(10,5))
                else:
                    fig = infig

                residuals[time,pol,bline],polys[time,pol,bline] = poly_residual(all_chan,data,weights=data_w,flags=data_f,xlab=xlab,ylab=dtype,deg=deg,ylim=ylim,logy=logy,plot=plot,fig=fig)

                red_chi_sq = compare_poly(all_chan,polys[time,pol,bline],polys[0,pol,bline],plot_poly=plot_poly,plot=plot,fig=fig)
                residuals[time,pol,bline,3] = red_chi_sq

                if plot:
                    if not os.path.exists('plots'):
                        os.mkdir('plots')
                    txt = 'Baseline {0}, Time {1:02d}, '.format(bline,time)
                    if red_chi_sq == 0.0:
                        txt += 'Reference fit'
                    else:
                        txt += r'$\chi_{\rm red}^2 = %s$' % '{0:.2f}'.format(red_chi_sq)

                    plt.title(txt)
                    if infig is None:
                        plt.savefig('plots/{0}-b{1}-p{2}-t{3}.png'.format(dtype,bline,pol,time))
                        plt.close()

    return residuals,polys

def location_specs(s,ts,parameters,loc):

    """Input a visibilitiy location, and get the timestamp, (auto-)polarisation and baseline at this location.

    Paramaters:
    -----------
    s : class: `Scan`
        The scan.
    ts : class: `katsdptelstate.TelescopeState`
        Telescope state, scoped to the cal namespace within the capture block.
    parameters : dict
        The pipeline parameters
    loc : class: `np.array`
        A location within a numpy ndarray.

    Returns:
    --------
    time : float
        Timestamp at location.
    pol : str
        Polarisation at location.
    bls_str : str
        Baseline string, in the format 'm001-m002'""""

    time = s.timestamps[loc[0][0]]
    pol = parameters['pol_ordering'][loc[1][0]] * 2
    bline = s.cross_ant.bls_lookup[loc[2][0]]
    bls_str = '{0}-{1}'.format(s.antennas[metric_bls[0]].name,s.antennas[metric_bls[1]].name)

    return time,pol,bls_str

def metric_description(s,ts,parameters,loc):

    """Derive metric description from its location.

    Paramaters:
    -----------
    s : class: `Scan`
        The scan.
    ts : class: `katsdptelstate.TelescopeState`
        Telescope state, scoped to the cal namespace within the capture block.
    parameters : dict
        The pipeline parameters
    loc : class: `np.array`
        A location within a numpy ndarray.

    Returns:
    --------
    description"""

    ref_time = s.timestamps[0]

    if loc.size > 0:
        time,pol,bls = location_specs(s,ts,parameters,loc)
        description = '(polynomial fit to {0} at time {1}, pol {2}, baseline {3}, compared to that of time {4})'.format(dtype,time,pol,bls,ref_time)

    else:
        description = '(polynomial fit to all timestamps, compared to reference timestamp for each baseline at time {0})'.format(ref_time)

    description = 'Mean reduced chi squared value of all timestamps {0}'.format(description)

    return description


def add_telstate_metric(metric,val,dtype,description,tolerance):

    """Add metric to telstate.

    Paramaters:
    -----------
    metric : str
        Short-hand metric name (e.g. 'bp' for bandpass).
    val : float
        Metric value.
    dtype : str
        Data type (e.g. 'phase').
    description : str
        Metric description.
    tolerance : float
        Metric tolerance value, below which, metric is flagged as bad."""

    ts.add('{0}_{1}_metric_val'.format(metric,dtype),val,ts=time)
    ts.add('{0}_{1}_metric_status'.format(metric,dtype),val > tolerance,ts=time)
    ts.add('{0}_{1}_metric_description'.format(metric,dtype),description,ts=time)
