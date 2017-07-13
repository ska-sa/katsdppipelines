#!/bin/python
#Provision of a class: 'cal_solution' to read calibration solutions 
#from a katdal object, the class provides a method 'apply_solutions'
#to interpolate solutions and apply them to visibilities at given timestamps.


import numpy as np
import scipy
import multiprocessing as mp

class CalibrationReadError(RuntimeError):
    """An error occurred in loading calibration values from file"""
    pass

class ComplexInterpolate1D(object):
    """Interpolator that separates magnitude and phase of complex values.

    The phase interpolation is done by first linearly interpolating the
    complex values, then normalising. This is not perfect because the angular
    velocity changes (slower at the ends and faster in the middle), but it
    avoids the loss of amplitude that occurs without normalisation.

    The parameters are the same as for :func:`scipy.interpolate.interp1d`,
    except that fill values other than nan and "extrapolate" should not be
    used.
    """
    def __init__(self, x, y, *args, **kwargs):
        mag = np.abs(y)
        phase = y / mag
        self._mag = scipy.interpolate.interp1d(x, mag, *args, **kwargs)
        self._phase = scipy.interpolate.interp1d(x, phase, *args, **kwargs)

    def __call__(self, x):
        mag = self._mag(x)
        phase = self._phase(x)
        return phase / np.abs(phase) * mag

def interpolate_nans_1d(y, *args, **kwargs):
    nan_locs = np.isnan(y)
    if np.all(nan_locs):
        y[:] = np.nan
    else:
        X = np.nonzero(~nan_locs)[0]
        Y = y[X]
        f=ComplexInterpolate1D(X, Y, *args, **kwargs)
        y=f(range(len(y)))
    return y


class cal_solution():
    """
    Class to hold, interpolate and apply cal solutions from a katdal object.
    Code largely pilfered from the katdal_loader in katsdpimager, but using a katdal object
    rather than a filename so that things are a bit more portable.
    """
    def __init__(self,katobj):

        self._K_solns = self._load_cal_product('cal_product_K', katobj, kind='linear')
        self._B_solns = self._load_cal_product('cal_product_B', katobj, kind='zero')
        self._G_solns = self._load_cal_product('cal_product_G', katobj, kind='linear')

        self._cal_pol_ordering = self._load_cal_pol_ordering(katobj)
        self._cal_ant_ordering = self._load_cal_antlist(katobj)
        self._data_channel_freqs = katobj.channel_freqs
        cp = katobj.corr_products
        self._cp_lookup = [[(self._cal_ant_ordering.index(prod[0][:-1]), self._cal_pol_ordering[prod[0][-1]],),
                            (self._cal_ant_ordering.index(prod[1][:-1]), self._cal_pol_ordering[prod[1][-1]],)]
                            for prod in katobj.corr_products]

    def _load_cal_attribute(self, key, katobj):
        """Load a fixed attribute from file.
        If the attribute is presented as a sensor, it is checked to ensure that
        all the values are the same.
        Raises
        ------
        CalibrationReadError
            if there was a problem reading the value from file (sensor does not exist,
            does not unpickle correctly, inconsistent values etc)
        """
        try:
            value = katobj.file['TelescopeState/{}'.format(key)]['value']
            if len(value) == 0:
                raise ValueError('empty sensor')
            value = [pickle.loads(x) for x in value]
        except (NameError, SyntaxError):
            raise
        except Exception as e:
            raise CalibrationReadError('Could not read {}: {}'.format(key, e))
        if not all(np.array_equal(value[0], x) for x in value):
            raise CalibrationReadError('Could not read {}: inconsistent values'.format(key))
        return value[0]

    def _load_cal_antlist(self,katobj):
        """Load antenna list used for calibration.
        If the value does not match the antenna list in the katdal dataset,
        a :exc:`CalibrationReadError` is raised. Eventually this could be
        extended to allow for an antenna list that doesn't match by permuting
        the calibration solutions.
        """
        cal_antlist = self._load_cal_attribute('cal_antlist',katobj)
        if cal_antlist != [ant.name for ant in katobj.ants]:
            raise CalibrationReadError('cal_antlist does not match katdal antenna list')
        return cal_antlist

    def _load_cal_pol_ordering(self,katobj):
        """Load polarization ordering used by calibration solutions.

        Returns
        -------
        dict
            Keys are 'h' and 'v' and values are 0 and 1, in some order
        """
        cal_pol_ordering = self._load_cal_attribute('cal_pol_ordering',katobj)
        try:
            cal_pol_ordering = np.array(cal_pol_ordering)
        except (NameError, SyntaxError):
            raise
        except Exception as e:
            raise CalibrationReadError(str(e))
        if cal_pol_ordering.shape != (4, 2):
            raise CalibrationReadError('cal_pol_ordering does not have expected shape')
        if cal_pol_ordering[0, 0] != cal_pol_ordering[0, 1]:
            raise CalibrationReadError('cal_pol_ordering[0] is not consistent')
        if cal_pol_ordering[1, 0] != cal_pol_ordering[1, 1]:
            raise CalibrationReadError('cal_pol_ordering[1] is not consistent')
        order = [cal_pol_ordering[0, 0], cal_pol_ordering[1, 0]]
        if set(order) != set('vh'):
            raise CalibrationReadError('cal_pol_ordering does not contain h and v')
        return {order[0]: 0, order[1]: 1}

    def _load_cal_product(self, key, katobj,**kwargs):
        """Loads calibration solutions from a katdal file.

        If an error occurs while loading the data, a warning is printed and the
        return value is ``None``. Any keyword args are passed to
        :func:`scipy.interpolate.interp1d` or `ComplexInterpolate1D`.

        Solutions that contain non-finite values are discarded.

        Parameters
        ----------
        key : str
            Name of the telescope state sensor

        Returns
        -------
        interp : callable
            Interpolation function which accepts timestamps and returns
            interpolated data with shape (time, channel, pol, antenna). If the
            solution is channel-independent, that axis will be present with
            size 1.
        """
        try:
            ds = katobj.file['TelescopeState/' + key]
            timestamps = ds['timestamp']
            values = []
            good_timestamps = []
            for i, ts in enumerate(timestamps):
                solution = pickle.loads(ds['value'][i])
                if solution.ndim == 2:
                    # Insert a channel axis
                    solution = solution[np.newaxis, ...]
                elif solution.ndim == 3:
                    #Only use channels selected in h5 file
                    solution = solution[katobj.channels]
                    #Interpolate across nans in channel axis.
                    for p in range(solution.shape[1]):
                     for a in range(solution.shape[2]):
                        solution[:,p,a]=interpolate_nans_1d(solution[:,p,a], kind='linear',
                        	fill_value='extrapolate', assume_sorted=True)
                else:
                    raise ValueError('wrong number of dimensions')
                if np.all(np.isfinite(solution)):
                    good_timestamps.append(ts)
                    values.append(solution)
            if not good_timestamps:
                raise ValueError('no finite solutions')
            values = np.array(values)
            kind = kwargs.get('kind', 'linear')
            if np.iscomplexobj(values) and kind not in ['zero', 'nearest']:
                interp = ComplexInterpolate1D
            else:
                interp = scipy.interpolate.interp1d
            return interp(
                good_timestamps, values, axis=0, fill_value='extrapolate',
                assume_sorted=True, **kwargs)
        except (NameError, SyntaxError):
            raise
        except Exception as e:
            print 'Could not load %s: %s', key, e
            return None

    def apply_solutions(self, vis, timestamps, weights=None):
        """
        Apply the solutions to visibilities at provided timestamps.
        Optionally recompute the weights as well.

        Returns
        =======
        cal_vis : array of vis with solns applied.
        """
        if vis.shape[2]!=len(self._cp_lookup):
            raise ValueError('Shape mismatch between correlation products.')
        if vis.shape[1]!=len(self._data_channel_freqs):
            raise ValueError('Shape mismatch in frequency axis.')
        if vis.shape[0]!=len(timestamps):
            raise ValueError('Shape mismatch in timestamps.')
        delay_to_phase = (-2j * np.pi * self._data_channel_freqs)[np.newaxis, :, np.newaxis, np.newaxis]
        K = np.exp(self._K_solns(timestamps) * delay_to_phase)
        B = self._B_solns(timestamps)
        G = self._G_solns(timestamps)
        p=mp.Pool()
        async_results=[]
        for idx,cp in enumerate(self._cp_lookup):
             async_results.append(p.apply_async(apply_KBG,(vis[:,:,idx],
                 (K[:,:,cp[0][1],cp[0][0]],K[:,:,cp[1][1],cp[1][0]].conj()),
                 (B[:, :, cp[0][1],cp[0][0]],B[:, :, cp[1][1],cp[1][0]].conj()),
                 (G[:, :, cp[0][1],cp[0][0]],G[:, :, cp[1][1],cp[1][0]].conj()),
                 weights[:,:,idx])))
        p.close()
        p.join()
        for i,result in enumerate(async_results):
            if weights is not None:
                vis[...,i],weights[...,i]=result.get()
            else:
                vis[...,i],weights=result.get()
        return vis

def apply_KBG(vis,K,B,G,weights=None):
    """
    Apply the solutions storued in the tuples K,B,G to vis, and resscale the weights
    for a single baseline.
    The tuples should contain soln for ant 1 and conjugate of soln for ant 2 corresponding
    to the array in vis and weights.
    This is mainly useful for parallelising the application of the solution
    across the baseline axis.
    """
    
    #K
    vis *= K[0] * K[1]
    #B
    scale = B[0] * B[1]
    vis /= scale
    if weights is not None:
        weights *= scale.real**2 + scale.imag**2
    #G
    scale = G[0] * G[1]
    vis[:,:] *= np.reciprocal(scale)
    if weights is not None:
        weights *= scale.real**2 + scale.imag**2
    return vis,weights
