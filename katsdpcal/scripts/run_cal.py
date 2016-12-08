#! /usr/bin/env python
import numpy as np
import time
import os
import signal
import manhole

from katsdptelstate import endpoint, ArgumentParser
from katsdpcal.pipelineprocs import ts_from_file, setup_ts
from katsdpcal.pipelineprocs import setup_observation_logger, finalise_observation
from katsdpcal import control, conf_dir

import logging
logger = logging.getLogger(__name__)


def print_dict(dictionary, ident='', braces=1):
    """
    Recursively prints nested dictionaries.

    Parameters
    ----------
    dictionary : dictionary to print
    ident      : indentation string
    braces     : number of braces to surround item with
    """
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            print '{0}{1}{2}{3}'.format(ident, braces*'[', key, braces*']')
            print_dict(value, ident+'  ', braces+1)
        else:
            print '{0}{1} = {2}'.format(ident, key, value)


def log_dict(dictionary, ident='', braces=1):
    """
    Recursively logs nested dictionaries.

    Parameters
    ----------
    dictionary : dictionary to print
    ident      : indentation string
    braces     : number of braces to surround item with
    """

    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            logger.info('{0}{1}{2}{3}'.format(ident, braces*'[', key, braces*']'))
            log_dict(value, ident+'  ', braces+1)
        else:
            logger.info('{0}{1} = {2}'.format(ident, key, value))


def comma_list(type_):
    """
    Return a function which splits a string on commas and converts each element to `type_`.
    """
    def convert(arg):
        return [type_(x) for x in arg.split(',')]
    return convert


def parse_opts():
    parser = ArgumentParser(description='Set up and wait for spead stream to run the pipeline.')
    parser.add_argument('--num-buffers', type=int, default=2, help='Specify the number of data buffers to use. default: 2')
    parser.add_argument('--buffer-maxsize', type=float, help='The amount of memory (in bytes) to allocate to each buffer.')
    parser.add_argument('--no-auto', action='store_true', help='Pipeline data DOESNT include autocorrelations [default: False (autocorrelations included)]')
    parser.set_defaults(no_auto=False)
    # note - the following two lines extract various parameters from the MC config
    parser.add_argument('--cbf-channels', type=int, help='The number of frequency channels in the visibility data. Default from MC config')
    parser.add_argument('--antenna-mask', type=comma_list(str), help='List of antennas in the L0 data stream. Default from MC config')
    parser.add_argument('--l0-spectral-spead', type=endpoint.endpoint_list_parser(7200, single_port=True), default=':7200', help='endpoints to listen for L0 spead stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--l1-spectral-spead', type=endpoint.endpoint_parser(7202), default='127.0.0.1:7202', help='destination for spectral L1 output. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--l1-rate', type=float, default=5e7, help='L1 spead transmission rate. For laptops, recommend rate of 5e7. Default: 5e7')
    parser.add_argument('--l1_level', default=0, help='Data to transmit to L1: 0 - none, 1 - target only, 2 - all [default: 0]')
    parser.add_argument('--notthreading', action='store_false', help='Use threading to control pipeline and accumulator [default: False (to use multiprocessing)]')
    parser.set_defaults(notthreading=True)
    parser.add_argument('--parameter-file', type=str, default='', help='Default pipeline parameter file (will be over written by TelescopeState.')
    parser.add_argument('--report-path', type=str, default='/var/kat/data', help='Path under which to save pipeline report. [default: /var/kat/data]')
    parser.add_argument('--log-path', type=str, default=os.path.abspath('.'), help='Path under which to save pipeline logs. [default: current directory]')
    return parser.parse_args()


def setup_logger(log_name,log_path='.'):
    """
    Set up the pipeline logger.
    The logger writes to a pipeline.log file and to stdout.

    Parameters
    ----------
    log_path : path in which log file will be written, string
    log_name : name of log file, string
    """
    log_path = os.path.abspath(log_path)
    log_file = '{0}/{1}'.format(log_path, log_name)

    # logging to file
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s.%(msecs)03dZ %(name)-24s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',)
    logging.Formatter.converter = time.gmtime
    logger.setLevel(logging.INFO)

    # logging to stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set format for console use
    formatter = logging.Formatter('%(asctime)s.%(msecs)03dZ - %(name)s - %(levelname)s - %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    formatter.converter = time.gmtime
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    # set log level to INFO for katsdpcal module
    logging.getLogger('katsdpcal').setLevel(logging.INFO)
    return log_file


def run_katsdpcal(ts, cbf_n_chans, antenna_mask, num_buffers=2, buffer_maxsize=None, auto=True,
           l0_endpoint=':7200', l1_endpoint='127.0.0.1:7202', l1_rate=5.0e7, l1_level=0,
           mproc=True, param_file='', report_path='', log_path='.', full_log=None):
    """
    Start the pipeline using 'num_buffers' buffers, each of size 'buffer_maxsize'.
    This will instantiate a katcp device server to run the calibration pipleine.
    The katcp server will then create num_buffers + 1 thread/processess: one for each pipeline 
    and one for the accumulator the reads data from the spead stream into each buffer.

    Parameters
    ----------
    ts             : telescope state, default: 'localhost' database 0, TelescopeState
    cbf_n_chans    : number of channels in the data stream, int
    antenna_mask   : antennas present in the data stream, list of strings or single csv string
    num_buffers    : number of buffers to use, int
    buffer_maxsize : maximum size of the buffer, float
    auto           : True for autocorrelations included in the data, False for cross-correlations only, bool
    l0_endpoint    : endpoint to listen to for L0 stream, default: ':7200', endpoint
    l1_endpoint    : destination endpoint for L1 stream, default: '127.0.0.1:7202', endpoint
    l1_rate        : rate for L1 stream transmission, default 5e7, float
    l1_level       : data to transmit to L1: 0 - none, 1 - target only, 2 - all
    mproc          : True for control via multiprocessing, False for control via threading, bool
    param_file     : file of default pipeline parameters, string
    report_path    : path under which to save pipeline report, string
    log_path       : path for pipeline logs, string
    full_log       : log file name, string
    """

    def force_exit(_signo=None, _stack_frame=None):
        logger.info("Forced exit on signal {0}".format(_signo,))
        os.kill(os.getpid(), signal.SIGKILL)

    # SIGTERM exit needed for Docker use since this process runs as PID 1
    # and does not get passed sigterm unless it has a custom listener
    signal.signal(signal.SIGTERM, force_exit)
    # SIGINT exit needed to shut down everything on keyboard interrupt
    signal.signal(signal.SIGINT, force_exit)

    logger.info('opt params: {0} {1}'.format(antenna_mask, cbf_n_chans))

    # extract data shape parameters
    #   argument parser traversed TS config to find these
    if antenna_mask is not None:
        ts.add('antenna_mask', antenna_mask, immutable=True)
    elif 'antenna_mask' not in ts:
        raise RuntimeError("No antenna_mask set.")
    if len(ts.antenna_mask) < 4:
        # if we have less than four antennas, no katsdpcal necessary
        logger.info('Only {0} antenna present - stopping katsdpcal'.format(len(ts.antenna_mask,)))
        return

    if cbf_n_chans is not None:
        ts.add('cbf_n_chans', cbf_n_chans, immutable=True)
    elif 'cbf_n_chans' not in ts:
        raise RuntimeError("No cbf_n_chans set.")

    # initialise TS from default parameter file
    #   defaults are used only for parameters missing from the TS
    if param_file == '':
        if ts.cbf_n_chans == 4096:
            param_filename = 'pipeline_parameters_meerkat_ar1_4k.txt'
            param_file = os.path.join(conf_dir, param_filename)
            logger.info('Parameter file for 4k mode: {0}'.format(param_file,))
        else:
            param_filename = 'pipeline_parameters_meerkat_ar1_32k.txt'
            param_file = os.path.join(conf_dir, param_filename)
            logger.info('Parameter file for 32k mode: {0}'.format(param_file,))
    else:
        logger.info('Parameter file: {0}'.format(param_file))
    logger.info('Inputting Telescope State parameters from parameter file.')
    ts_from_file(ts, param_file)
    # telescope state logs for debugging
    logger.info('Telescope state parameters:')
    for keyval in ts.keys():
        # don't print out the really long telescope state key values
        if keyval not in ['cbf_bls_ordering', 'cbf_channel_freqs']:
            logger.info('{0} : {1}'.format(keyval, ts[keyval]))
    logger.info('Telescope state config graph:')
    log_dict(ts.config)

    # set up TS for pipeline use
    logger.info('Setting up Telescope State parameters for pipeline.')
    setup_ts(ts)

    # consolidate spead transmission params
    spead_params = {}
    spead_params['l0_endpoint'] = l0_endpoint
    spead_params['l1_endpoint'] = l1_endpoint
    spead_params['l1_level'] = l1_level
    spead_params['l1_rate'] = l1_rate

    # consolidate information needed to set up buffers
    npol = 4
    nant = len(ts.cal_antlist)
    # number of baselines (may include autocorrelations)
    nbl = nant*(nant+1)/2 if auto else nant*(nant-1)/2
    # get buffer size
    if buffer_maxsize is not None:
        ts.add('cal_buffer_size', buffer_maxsize)
    else:
        if ts.has_key('cal_buffer_size'):
            buffer_maxsize = ts['cal_buffer_size']
        else:
            buffer_maxsize = 20.0e9
            ts.add('cal_buffer_size', buffer_maxsize)
    # buffer needs to include:
    #   visibilities, shape(time,channel,baseline,pol), type complex64 (8 bytes)
    #   flags, shape(time,channel,baseline,pol), type uint8 (1 byte)
    #   weights, shape(time,channel,baseline,pol), type float32 (4 bytes)
    #   time, shape(time), type float64 (8 bytes)
    # plus minimal extra for scan transition indices
    scale_factor = 8. + 1. + 4.  # vis + flags + weights
    time_factor = 8. + 0.1  # time + 0.1 for good measure (indiced)
    array_length = buffer_maxsize/((scale_factor*ts.cbf_n_chans*npol*nbl) + time_factor)
    array_length = np.int(np.ceil(array_length))
    logger.info('Buffer size : {0} G'.format(buffer_maxsize/1.e9,))
    logger.info('Max length of buffer array : {0}'.format(array_length,))
    # buffer shape based on data characteristics
    buffer_shape = [array_length, ts.cbf_n_chans, npol, nbl]
    logger.info('Buffer shape : {0}'.format(buffer_shape,))

    # start calibration pipeline server
    server = control.CalibrationServer('', 5000, control_method, control_task, spead_params)
    logger.info('Starting calibration pipeline server')
    server.start(ts, num_buffers, buffer_shape)
    try:
        manhole.install(oneshot_on='USR1', locals={'ts':ts, 'server':server})
    # allow remote debug connections and expose telescope state, accumulator and pipelines
    except manhole.AlreadyInstalled:
        pass

    # run calibration pipeline
    while True:
        # set up the logger specific to this observation
        observation_log = '{0}_pipeline.log'.format(int(time.time()),)
        obs_log = setup_observation_logger(observation_log, log_path)

        logger.info('Ready to receive L0 data on port {0}'.format(spead_params['l0_endpoint'].port,))
        logger.info('===========================')
        logger.info('   Starting new observation')
        server.capture_start()
        logger.info('capture done')
        server.capture_done()

        # write report, copy log of this observation into the report directory
        finalise_observation(ts, report_path, obs_log, full_log)
        logger.info('   Observation finalised')
        logger.info('===========================')
    # close down everything
    server.join()

if __name__ == '__main__':

    opts = parse_opts()

    # set up logging
    log_name = 'pipeline.log'
    log_path = os.path.abspath(opts.log_path)
    log_file = setup_logger(log_name, log_path)

    # threading or multiprocessing imports
    if opts.notthreading is True:
        logger.info("Using multiprocessing")
        import multiprocessing as control_method
        from multiprocessing import Process as control_task
    else:
        logger.info("Using threading")
        import threading as control_method
        from threading import Thread as control_task

    run_katsdpcal(opts.telstate,
           cbf_n_chans=opts.cbf_channels, antenna_mask=opts.antenna_mask,
           num_buffers=opts.num_buffers, buffer_maxsize=opts.buffer_maxsize, auto=not(opts.no_auto),
           l0_endpoint=opts.l0_spectral_spead[0], l1_endpoint=opts.l1_spectral_spead,
           l1_rate=opts.l1_rate, l1_level=opts.l1_level, mproc=opts.notthreading,
           param_file=opts.parameter_file, report_path=opts.report_path, log_path=log_path, full_log=log_file)
