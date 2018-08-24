import numpy as np
from . import plotting

def derive_metrics(vis,weights,flags,s,ts,parameters,deg=3,metric_func=np.mean,metric_tol=10,plot=False,plot_per='baseline'):

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
    metric_func : function, optional
        Function to apply to the distribution of residuals, which is np.mean by default.
    metric_tol : float, optional
        If the value of metric_func is less than this tolerance, flag the metric as bad.
    plot : bool, optional
        Plot the data and fitting?
    plot_per : str, optional
        When plot is True, save plot for all data, every baseline, or every timestamp ('all' | 'baseline' | 'time')."""

    # get good axis limits from data for plotting amplitude
    if plot:
        inc = vis.shape[1] // 4
        high = int(np.nanmedian(vis[:,0:inc,:,:].real)*1.3)
        low = int(np.nanmedian(vis[:,-inc:-1,:,:].real)*0.75)
        aylim=(low,high)
    else:
        aylim = None

    # compute bandpass data quality metrics
    amp_resid,amp_polys,fig = bandpass_metrics(vis.compute(),weights.compute(),flags.compute(),s,ts,parameters,dtype='Amp',deg=deg,plot=plot,plot_per=plot_per,ylim=aylim)
    phase_resid,phase_polys,fig = bandpass_metrics(vis.compute(),weights.compute(),flags.compute(),s,ts,parameters,dtype='Phase',deg=deg,plot=plot,plot_per=plot_per,ylim=(-10,10))

    # take single bandpass metric as worst
    BP_amp_metric = metric_func(amp_resid[:,:,:,3])
    BP_amp_metric_loc = np.where(amp_resid == BP_amp_metric)
    time,description = metric_description(s, ts, parameters, BP_amp_metric_loc)
    add_telstate_metric('bp',BP_amp_metric,'amp',description,time,metric_tol,ts)

    BP_phase_metric = metric_func(phase_resid[:,:,:,3])
    BP_phase_metric_loc = np.where(phase_resid == BP_phase_metric)
    time,description = metric_description(s, ts, parameters, BP_phase_metric_loc)
    add_telstate_metric('bp',BP_phase_metric,'phase',description,time,metric_tol,ts)

    plot_all_hist(amp_resid,'Amp',extn='png')
    plot_all_hist(phase_resid,'Phase',extn='png')

def bandpass_metrics(vis,weights,flags,s,ts,parameters,dtype='Amplitude',freqs=None,deg=3,ref_time=0,plot=False,plot_poly=True,plot_ref=True,plot_per='baseline',ylim=None):

    """Calculate the bandpass metrics, including....

    Paramaters:
    -----------
    vis : class:`np.ndarray`
        The visibilities. Assumed to have shape (ntimes, nchannels, npols, nbaselines).
    weights : class:`np.ndarray`
        The visibility weights. Assumed to have shape (ntimes, nchannels, npols, nbaselines).
    flags : class:`np.ndarray`
        The visibility flags. Assumed to have shape (ntimes, nchannels, npols, nbaselines).
    s : class: `Scan`
        The scan.
    ts : :class:`katsdptelstate.TelescopeState`
        Telescope state, scoped to the cal namespace within the capture block.
    parameters : dict
        The pipeline parameters
    dtype : str, optional
        'phase' | 'amplitude' (case-insensitive).
    freqs : class:`np.ndarray`, optional
        A list of frequencies for each channel.
    deg : int, optional
        Degree of polynomial to fit.
    ref_time : int, optional
        Visibility index of timestamp axis to use for reference timestamp.
    plot : bool, optional
        Write a matplotlib figure?
    plot_poly : bool, optional
        Overlay the polynomial on the plot?
    plot_ref : bool, optional
        Overlay the reference polynomial on the plot?
    plot_per : str, optional
        When plot is True, save plot for all data, every baseline, or every timestamp ('all' | 'baseline' | 'time').
    ylim : tuple, optional
        Limits to put on the y axis

    Returns:
    --------
    residuals : class `np.array`
        The residual metrics of the fit polynomials, with shape
        (ntimestamps, npols, nbaselines, 4 [red_chi_sq,nmad,nmad_model,red_chi_sq_poly])
    polys : class `np.array`
        Array of `np.polyfit` objects, representing the polynomials fit to each element of input data, with shape
        (ntimestamps, npols, nbaselines, deg+1).
    fig : class `matplotlib.figure`
        The figure (None if plot is False)."""

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

    if plot and plot_per.lower() == 'all':
        fig = plotting.plot_bandpass(None,None)
    else:
        fig = None

    for bline in range(vis.shape[3]):
        if plot and plot_per.lower() == 'baseline':
            fig = plotting.plot_bandpass(None,None)
        for pol in range(vis.shape[2]):
            for time in range(vis.shape[0]):

                data = vis[time,:,pol,bline]
                data_w = weights[time,:,pol,bline]
                data_f = flags[time,:,pol,bline]

                if freqs is None:
                    all_chan = np.arange(vis.shape[1])
                    xlab = 'Channel'
                else:
                    all_chan = np.arange(freq[0],freq[-1],1) / 1.0e6
                    xlab = 'Frequency (MHz)'

                if plot and plot_per.lower() == 'time':
                    fig = plotting.plot_bandpass(None,None)

                residuals[time,pol,bline],polys[time,pol,bline],fig = poly_residual(all_chan,data,weights=data_w,flags=data_f,xlab=xlab,ylab=dtype,deg=deg,ylim=ylim,logy=logy,plot=plot,fig=fig)
                red_chi_sq,fig = compare_poly(all_chan,polys[time,pol,bline],polys[ref_time,pol,bline],plot_poly=plot_poly,plot=plot,fig=fig)
                residuals[time,pol,bline,3] = red_chi_sq

                if plot:
                    loc = (np.array([time]),np.array([pol]),np.array([bline]))
                    tstamp,corr,bls = location_specs(s,ts,parameters,loc)

                    if plot_per.lower() == 'time':
                        txt = 'Baseline {0}, Time {1}, Pol {2}: '.format(bls,tstamp,corr)
                        if time == ref_time:
                            txt += 'Reference fit'
                        else:
                            txt += r'$\chi_{\rm red}^2 = %s$' % '{0:.2f}'.format(red_chi_sq)
                        plotting.save_bandpass(txt,dtype,bls,corr,tstamp)

        if plot and plot_per.lower() == 'baseline':
            txt = 'Baseline {0}, all time, all pol: '.format(bls)
            txt += r'$\chi_{\rm red,worst}^2 = %s$' % '{0:.2f}'.format(np.nanmax(residuals[:,:,bline,3]))
            plotting.save_bandpass(txt,dtype,bls,'all_pol','_all{0}'.format(np.mean(s.timestamps[0])))

    if plot and plot_per.lower() == 'all':
        txt = 'All baselines, all time, all pol: '
        txt += r'$\chi_{\rm red,worst}^2 = %s$' % '{0:.2f}'.format(np.nanmax(residuals[:,:,:,3]))
        plotting.save_bandpass(txt,dtype,'all_baselines','all_pol','all_{0}'.format(np.mean(s.timestamps[0])))

    return residuals,polys,fig


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
            Place holder for reduced chi squared metric.
    poly_paras : class `np.polyfit`
        The fitted polynomial parameters.
    fig : class `matplotlib.figure`
        The figure (None if plot is False)."""

    indices = ~np.isnan(xdata) & ~np.isnan(ydata) & (flags == 0)
    if weights is not None:
        indices = indices & ~(weights == 0)
        w = 1.0/weights[indices]

    x = xdata[indices]
    y = ydata[indices]

    if x.size > 0 and y.size > 0:
        poly_paras = np.polyfit(x,y,deg,w=w)
        poly = np.poly1d(poly_paras)

        chi_sq=np.sum((y-poly(x))**2)
        DOF=len(x)-(deg+1) #polynomial has deg+1 free parameters

        red_chi_sq = chi_sq/DOF
        nmad = np.median(np.abs(y-np.median(y)))/0.6745
        nmad_model = np.median(np.abs(y-poly(x)))/0.6745

        if plot:
            fig = plotting.plot_bandpass(x,y,xlab,ylab,xlim,ylim,fig=fig,logy=logy)

        return np.array([red_chi_sq,nmad,nmad_model,None]),poly_paras,fig
    else:
        return np.array([np.nan,np.nan,np.nan,None]),np.nan,None


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
        Overlay the reference polynomial fit on the figure?

    Returns:
    --------
    red_chi_sq : float
        Reduced chi squared between bot polynomials.
    fig : class `matplotlib.figure`
        The figure (None if plot is False)."""

    xdata = xdata[~np.isnan(xdata)]
    xlin = np.arange(np.min(xdata),np.max(xdata))

    fit1 = np.poly1d(poly1)
    fit2 = np.poly1d(poly2)

    chi_sq=np.sum((fit1(xdata)-fit2(xdata))**2)
    DOF=len(xdata)-len(poly1)-len(poly2)
    red_chi_sq = chi_sq/DOF

    if plot:
        fig = plotting.plot_polys(xlin,fit1,fit2,plot_poly,plot_ref,fig=fig)
    else:
        fig = None

    return red_chi_sq,fig

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
        Baseline string, in the format 'm001-m002'"""

    time = s.timestamps[loc[0][0]]
    pol = parameters['pol_ordering'][loc[1][0]] * 2
    bline = s.cross_ant.bls_lookup[loc[2][0]]
    bls_str = '{0}-{1}'.format(s.antennas[bline[0]].name,s.antennas[bline[1]].name)

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
    time : float
        Timestamp at location.
    description : str
        Metric description."""

    ref_time = s.timestamps[0]

    if loc[0].size > 0:
        time,pol,bls = location_specs(s,ts,parameters,loc)
        description = '(polynomial fit to {0} at time {1}, pol {2}, baseline {3}, compared to that of time {4})'.format(dtype,time,pol,bls,ref_time)

    else:
        time = ref_time
        description = '(polynomial fit to all timestamps, compared to reference timestamp for each baseline at time {0})'.format(ref_time)

    description = 'Mean reduced chi squared value of all timestamps {0}'.format(description)

    return time,description


def add_telstate_metric(metric,val,dtype,description,time,tolerance,ts):

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
    time : float
        Timestamp at location.
    tolerance : float
        Metric tolerance value, below which, metric is flagged as bad.
    ts : class: `katsdptelstate.TelescopeState`
        Telescope state, scoped to the cal namespace within the capture block."""

    ts.add('{0}_{1}_metric_val'.format(metric,dtype),val,ts=time)
    ts.add('{0}_{1}_metric_status'.format(metric,dtype),val < tolerance,ts=time)
    ts.add('{0}_{1}_metric_description'.format(metric,dtype),description,ts=time)


def plot_all_hist(data,label,extn='png',metric_axis=3,metric_func=np.max):

    """Write four plots to disc, summarising quality metrics.

    Parameters
    ----------
    data : class `np.array`
        The data to plot.
    label : string
        Label to append to beginning of file name that is written to disc.
    extn : str, optional
        The extension of the file to write to disc.
    metric_axis : int, optional
        Axis of data that contains metric distribution.
    metric_func : func
        Function to apply to metric distribution and return as single metric.

    Returns:
    --------
    metric : float
        Single metric value of whole of metric distrubtion."""

    plotting.plot_histogram(data[:,:,:,0],label + r' $\rm \chi_{red,data}^2$',upper_lim=1e2,nbins=40,figname=label+'_hist_data',extn=extn)
    plotting.plot_histogram(data[:,:,:,1],label + r' NMAD$_{\rm model}$',upper_lim=1e1,nbins=40,figname=label+'_hist_nmad_model',extn=extn)
    plotting.plot_histogram(data[:,:,:,2],label + ' NMAD',upper_lim=1e1,nbins=40,figname=label+'_hist_nmad',extn=extn)
    plotting.plot_histogram(data[:,:,:,3],label + r' $\rm \chi_{red,poly}^2$',upper_lim=1e2,nbins=40,figname=label+'_hist_poly',extn=extn)
    return metric_func(data[:,:,:,metric_axis])


def get_stats(data):

    """Return the min, max, median, mean, standard deviation, standard error and
    normalised median absolute deviation (NMAD) of the non-nan values in a list.

    Arguments:
    ----------
    data : list-like (numpy.array or pandas.Series)
        The data used to calculate the statistics.

    Returns:
    --------
    min : float
        The minimum.
    max : float
        The maximum.
    med : float
        The median.
    mean : float
        The mean.
    std : float
        The standard deviation.
    err : float
        The standard error.
    nmad : float
        The normalised mad

    See Also
    --------
    numpy.array
    pandas.Series"""

    #remove nan indices, as these affect the calculations
    values = data[~np.isnan(data)]

    min = np.min(values)
    max = np.max(values)
    med = np.median(values)
    mean = np.mean(values)
    std = np.std(values)
    sterr = std / np.sqrt(len(values))
    nmad = np.median(np.abs(values-np.median(values)))/0.6745

    return min,max,med,mean,std,sterr,nmad

