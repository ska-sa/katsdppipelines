"""
Pipeline procedures for MeerKAT calibration pipeline
====================================================
"""

from .report import make_cal_report

import numpy as np
import os
import shutil
import time

# for model files
import glob

import logging
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# --- General pipleline functions
# -------------------------------------------------------------------------------------------------


def clear_ts(ts):
    """
    Clear the TS.

    Inputs
    ======
    ts: Telescope State
    """
    try:
        for key in ts.keys():
            ts.delete(key)
    except AttributeError:
        # the Telescope State is empty
        pass


def init_ts(ts, param_dict, clear=False):
    """
    Initialises the telescope state from parameter dictionary.
    Parameters from the parameter dictionary are only added to the TS the
    the parameter is not already in the TS.

    Inputs
    ======
    ts : Telescope State
    param_dict : dictionary of parameters
    clear : clear ts before initialising
    """

    if clear:
        # start with empty Telescope State
        clear_ts(ts)

    # populate ts with parameters
    #   parameter only added if it is missing from the TS
    for key in param_dict.keys():
        if key not in ts:
            ts.add(key, param_dict[key])


def ts_from_file(ts, filename):
    """
    Initialises the telescope state from parameter file.
    Note: parameters will be returned as ints, floats or strings (not lists)

    Inputs
    ======
    ts : Telescope State
    filename : parameter file

    Notes
    =====
    * Parameter file uses colons (:) for delimiters
    * Parameter file uses hashes (#) for comments
    * Missing parameters are set to empty strings ''
    """
    param_list = np.genfromtxt(filename, delimiter=':', dtype=np.str, comments='#', missing_values='')
    param_dict = {}
    for key, value in param_list:
        try:
            # integer?
            param_value = int(value)
        except ValueError:
            try:
                # float?
                param_value = float(value)
            except ValueError:
                # keep as string, strip whitespace
                param_value = value.strip()

        param_dict[key.strip()] = param_value

    init_ts(ts, param_dict)


def setup_ts(ts, logger=logger):
    """
    Set up the telescope state for pipeline use.
    In general, the calibration parameters are mutable.
    Only subarray characteristics (e.g. antenna_mask) are immutable.

    Inputs
    ======
    ts : Telescope State
    logger : logger

    Notes
    =====
    Assumed starting ts entries:
    antenna_mask

    Assumed ending ts entries
    antenna_mask          - list or csv string of antennas present in the data, immutable
    cal_antlist           - list of antennas present in the data
    cal_antlist_description - list of antenna descriptions (in katpoint description string format)
    cal_array_position    - array position (in katpoint description string format)
    cal_preferred_refants - ordered list of refant preference
    cal_refant            - reference antenna
    """

    # ensure that antenna_mask is list of strings, not single csv string
    if isinstance(ts.antenna_mask, str):
        csv_to_list(ts,'antenna_mask')
    # cal_antlist
    #   this should not be pre-set (determine from antenna_mask, which is pre-set)
    ts.add('cal_antlist', ts.antenna_mask)

    # cal_antlist_description
    #    list of antenna descriptions
    description_list = [ts['{0}_observer'.format(ant,)] for ant in ts.cal_antlist]
    ts.add('cal_antlist_description', description_list, immutable=True)
    # array reference position
    if 'cal_array_position' not in ts:
        # take lat-long-alt value from first antenna in antenna list as the array reference position
        ts.add('cal_array_position','array_position, '+','.join(description_list[0].split(',')[1:-1]))

    # cal_preferred_refants
    if 'cal_preferred_refants' not in ts:
        logger.info('Preferred antenna list set to antenna mask list:')
        ts.add('cal_preferred_refants',ts.cal_antlist)
        logger.info('{0} : {1}'.format('cal_preferred_refants',ts.cal_preferred_refants))
    else:
        # change cal_preferred_refants to lists of strings (not single csv string)
        csv_to_list(ts,'cal_preferred_refants')
        # reduce the preferred antenna list to only antennas present in cal_antlist
        preferred = [ant for ant in ts.cal_preferred_refants if ant in ts.cal_antlist]
        if preferred != ts.cal_preferred_refants:
            if preferred == []:
                logger.info('No antennas from the antenna mask in the preferred antenna list')
                logger.info(' - preferred antenna list set to antenna mask list:')
                ts.delete('cal_preferred_refants')
                ts.add('cal_preferred_refants', ts.cal_antlist)
            else:
                logger.info('Preferred antenna list reduced to only include antennas in antenna mask:')
                ts.delete('cal_preferred_refants')
                ts.add('cal_preferred_refants',preferred)
            logger.info('{0} : {1}'.format('cal_preferred_refants',ts.cal_preferred_refants))

    # cal_refant
    if 'cal_refant' not in ts:
        ts.add('cal_refant',ts.cal_preferred_refants[0])
        logger.info('Reference antenna: {0}'.format(ts.cal_refant,))
    else:
        if ts.cal_refant not in ts.cal_antlist:
            ts.delete('cal_refant')
            ts.add('cal_refant',ts.cal_preferred_refants[0])
            logger.info('Requested reference antenna not present in subarray. Change to reference antenna: {0}'.format(ts.cal_refant,))


def csv_to_list(ts,keyname):
    """
    Cange Telescope State entry for immutable key from csv string to list of strings

    Inputs
    ======
    ts : Telescope State
    keyval : key to change

    """
    if isinstance(ts[keyname], str):
        keyvallist = [val.strip() for val in ts[keyname].split(',')]
        ts.delete(keyname)
        ts.add(keyname,keyvallist,immutable=True)


def get_model(name, lsm_dir_list=[]):
    """
    Get a sky model from a text file.
    The name of the text file must incorporate the name of the source.

    Inputs
    ======
    name : name of source, string
    lsm_dir : directory containing the source model txt file

    Returns
    =======
    model_components : numpy recarray of sky model component parameters
    model_file : name of model component file used
    """
    if not isinstance(lsm_dir_list, list):
        lsm_dir_list = [lsm_dir_list]
    # default to check the current directory first
    lsm_dir_list.append('.')

    # iterate through the list from the end so the model from the earliest directory in the list is used
    model_file = []
    for lsm_dir in reversed(lsm_dir_list):
        model_file_list = glob.glob('{0}/*{1}*.txt'.format(glob.os.path.abspath(lsm_dir), name))
        # ignore tilde ~ backup files
        model_file_list = [f for f in model_file_list if f[-1] != '~']

        if len(model_file_list) == 1:
            model_file = model_file_list[0]
        elif len(model_file_list) > 1:
            # if there are more than one model files for the source IN THE SAME DIRECTORY, raise an error
            raise ValueError('More than one possible sky model file for {0}: {1}'.format(name, model_file_list))

    # if there is not model file, return model components as None
    if model_file == []:
        model_components = None
    else:
        model_dtype = [('tag','S4'),('name','S16'),('RA','S24'),('dRA','S8'),('DEC','S24'),('dDEC','S8'),
        ('a0','f16'),('a1','f16'),('a2','f16'),('a3','f16'),('fq','f16'),('fu','f16'),('fv','f16')]
        model_components = np.genfromtxt(model_file, delimiter=',', dtype=model_dtype)
    return model_components, model_file


def setup_observation_logger(log_name, log_path='.'):
    """
    Set up a pipeline logger to file.

    Inputs
    ======
    log_path : str
        path in which log file will be written
    """
    log_path = os.path.abspath(log_path)

    # logging to file
    # set format
    formatter = logging.Formatter('%(asctime)s.%(msecs)03dZ %(name)-24s %(levelname)-8s %(message)s')
    formatter.datefmt='%Y-%m-%d %H:%M:%S'

    obs_log = logging.FileHandler('{0}/{1}'.format(log_path,log_name))
    obs_log.setFormatter(formatter)
    logging.getLogger('').addHandler(obs_log)
    return obs_log


def stop_observation_logger(obs_log):
    logging.getLogger('').removeHandler(obs_log)


def finalise_observation(ts, report_path='.', obs_log=None, full_log=None):
    """
    Housekeeping for end of observation:
     - create directory for observation
     - save logs in the directory
     - create pipeline report in the directory

    Parameters
    ----------
    ts          : telescope state
    report_path : path for pipeline report
    obs_log     : log for observation
    full_log    : log for the full run of the pipeline
    """

    # get observation end time
    if ts.has_key('cal_obs_end_time'):
        obs_end = ts.cal_obs_end_time
    else:
        logger.info('Unknown observation end time')
        obs_end = time.time()

    # get experiment_id and start time
    try:
        obs_params = ts.get_range('obs_params', st=0, et=obs_end, return_format='recarray')
        obs_keys = obs_params['value']
        obs_times = obs_params['time']
        # choose most recent experiment id (last entry in the list), if there are more than one
        experiment_id_string = [x for x in obs_keys if 'experiment_id' in x][-1]
        experiment_id = eval(experiment_id_string.split()[-1])
        obs_start = [t for x, t in zip(obs_keys, obs_times) if 'experiment_id' in x][-1]
    except (TypeError, KeyError, AttributeError):
        # TypeError, KeyError because this isn't properly implimented yet
        # AttributeError in case this key isnt in the telstate for whatever reason
        experiment_id = '{0}_unknown_project'.format(int(time.time()),)
        obs_start = None
    # get subarray ID
    subarray_id = ts['subarray_product_id'] if ts.has_key('subarray_product_id') else 'unknown_subarray'

    # make directory for this observation, for logs and report
    report_path = os.path.abspath(report_path)
    obs_dir = '{0}/{1}_{2}_{3}'.format(report_path, int(time.time()), subarray_id, experiment_id)
    current_obs_dir = '{0}-current'.format(obs_dir,)
    try:
        os.mkdir(current_obs_dir)
    except OSError:
        logger.warning('Experiment ID directory {} already exits'.format(current_obs_dir,))

    # create pipeline report (very basic at the moment)
    try:
        make_cal_report(ts, current_obs_dir, experiment_id, st=obs_start, et=obs_end)
    except Exception, e:
        logger.info('Report generation failed: {0}'.format(e,))

    # copy log of this observation into the report directory
    if obs_log is not None:
        shutil.move(obs_log.baseFilename, '{0}/pipeline_{1}.log'.format(current_obs_dir, experiment_id))
        stop_observation_logger(obs_log)
        if full_log is not None:
            shutil.copy(full_log, '{0}/.'.format(current_obs_dir,))

    # change report and log directory to final name for archiving
    shutil.move(current_obs_dir, obs_dir)
