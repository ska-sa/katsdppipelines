#! /usr/bin/env python
import time
import os
import signal
import logging
import manhole
import numpy as np

import trollius
from trollius import From
from tornado.platform.asyncio import AsyncIOMainLoop, to_asyncio_future
import concurrent.futures

from katsdptelstate import endpoint
import katsdpservices

from katsdpcal.control import (
    Accumulator, Pipeline, ReportWriter, shared_empty,
    CalDeviceServer, SensorReadingEvent, StopEvent)
from katsdpcal.pipelineprocs import ts_from_file, setup_ts

from katsdpcal import param_dir, rfi_dir


logger = logging.getLogger(__name__)


class TaskLogFormatter(logging.Formatter):
    """Injects the thread/process name into log messages.

    This must be subclassed to provide the TASKNAME property.
    """
    def __init__(self, fmt=None, datefmt=None):
        if fmt is None:
            fmt = '%(message)s'
        fmt = fmt.replace('%(message)s', '[%(' + self.TASKNAME + ')s] %(message)s')
        super(TaskLogFormatter, self).__init__(fmt, datefmt)


class ProcessLogFormatter(TaskLogFormatter):
    TASKNAME = 'processName'


class ThreadLogFormatter(TaskLogFormatter):
    TASKNAME = 'threadName'


def log_dict(dictionary, ident='', braces=1):
    """ Recursively logs nested dictionaries."""

    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            logger.info('%s%s%s%s', ident, braces * '[', key, braces * ']')
            log_dict(value, ident+'  ', braces+1)
        else:
            logger.info('%s%s = %s', ident, key, value)


def comma_list(type_):
    """Return a function which splits a string on commas and converts each element to
    `type_`."""

    def convert(arg):
        return [type_(x) for x in arg.split(',')]
    return convert


def parse_opts():
    parser = katsdpservices.ArgumentParser(
        description='Set up and wait for spead stream to run the pipeline.')
    parser.add_argument(
        '--num-buffers', type=int, default=2,
        help='Specify the number of data buffers to use. default: 2')
    parser.add_argument(
        '--buffer-maxsize', type=float,
        help='The amount of memory (in bytes) to allocate to each buffer.')
    parser.add_argument(
        '--no-auto', action='store_true',
        help='Pipeline data DOESNT include autocorrelations '
        + '[default: False (autocorrelations included)]')
    parser.set_defaults(no_auto=False)
    # note - the following lines extract various parameters from the MC config
    parser.add_argument(
        '--cbf-channels', type=int,
        help='The number of frequency channels in the visibility data. Default from MC config')
    parser.add_argument(
        '--cbf-pols', type=int,
        help='The number of polarisation products in the visibility data. Default from MC config')
    parser.add_argument(
        '--antenna-mask', type=comma_list(str),
        help='List of antennas in the L0 data stream. Default from MC config')
    # also need bls ordering
    parser.add_argument(
        '--l0-spectral-spead', type=endpoint.endpoint_list_parser(7200, single_port=True),
        default=':7200',
        help='endpoints to listen for L0 spead stream (including multicast IPs). '
        + '[<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument(
        '--l0-spectral-interface',
        help='interface to subscribe to for L0 spectral data. [default: auto]', metavar='INTERFACE')
    parser.add_argument(
        '--l1-spectral-spead', type=endpoint.endpoint_parser(7202), default='127.0.0.1:7202',
        help='destination for spectral L1 output. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument(
        '--l1-rate', type=float, default=5e7,
        help='L1 spead transmission rate. For laptops, recommend rate of 5e7. Default: 5e7')
    parser.add_argument(
        '--l1_level', default=0,
        help='Data to transmit to L1: 0 - none, 1 - target only, 2 - all [default: 0]')
    parser.add_argument(
        '--notthreading', action='store_false',
        help='Use threading to control pipeline and accumulator '
        + '[default: False (to use multiprocessing)]')
    parser.set_defaults(notthreading=True)
    parser.add_argument(
        '--parameter-file', type=str, default='',
        help='Default pipeline parameter file (will be over written by TelescopeState.')
    parser.add_argument(
        '--report-path', type=str, default='/var/kat/data',
        help='Path under which to save pipeline report. [default: /var/kat/data]')
    parser.add_argument(
        '--log-path', type=str, default=os.path.abspath('.'),
        help='Path under which to save pipeline logs. [default: current directory]')
    parser.add_argument(
        '--port', '-p', type=int, default=2048, help='katcp host port [%(default)s]')
    parser.add_argument(
        '--host', '-a', type=str, default='', help='katcp host address [all hosts]')
    return parser.parse_args()


def setup_logger(log_name, log_path='.'):
    """
    Set up the pipeline logger.
    The logger writes to a pipeline.log file and to stdout.

    Parameters
    ----------
    log_path : str
        path in which log file will be written
    log_name : str
        name of log file
    """
    katsdpservices.setup_logging()

    # logging to file
    log_path = os.path.abspath(log_path)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03dZ %(name)-24s %(levelname)-8s %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    formatter.converter = time.gmtime

    handler = logging.FileHandler('{0}/{1}'.format(log_path, log_name))
    handler.setFormatter(formatter)
    logging.getLogger('').addHandler(handler)


def create_buffer_arrays(buffer_shape, mproc=True):
    """
    Create empty buffer record using specified dimensions
    """
    if mproc:
        factory = shared_empty
    else:
        factory = np.empty
    data = {}
    data['vis'] = factory(buffer_shape, dtype=np.complex64)
    data['flags'] = factory(buffer_shape, dtype=np.uint8)
    data['weights'] = factory(buffer_shape, dtype=np.float32)
    data['times'] = factory(buffer_shape[0], dtype=np.float)
    data['max_index'] = factory([1], dtype=np.int32)
    data['max_index'][0] = 0
    return data


def queue_forward(mp_queue, asyncio_queue, loop):
    """Forwards from a multiprocessing.Queue to a trollius.Queue. This is run
    in a separate thread, since the multiprocessing queue accesses are blocking.
    """
    while True:
        event = mp_queue.get()
        logger.debug('Received event %r', event)
        loop.call_soon_threadsafe(asyncio_queue.put_nowait, event)
        if isinstance(event, StopEvent):
            break


@trollius.coroutine
def run_threads(ts, cbf_n_chans, cbf_n_pols, antenna_mask, host, port, num_buffers=2,
                buffer_maxsize=None, auto=True,
                l0_endpoint=':7200', l0_interface=None,
                l1_endpoint='127.0.0.1:7202', l1_rate=5.0e7, l1_level=0,
                mproc=True, param_file='', report_path='', log_path='.', full_log=None):
    """
    Start the pipeline using 'num_buffers' buffers, each of size 'buffer_maxsize'.
    This will instantiate num_buffers + 1 threads; a thread for each pipeline and an
    extra accumulator thread the reads data from the spead stream into each buffer
    seen by the pipeline.

    Inputs
    ======
    ts : TelescopeState
        The telescope state, default: 'localhost' database 0
    cbf_n_chans : int
        The number of channels in the data stream
    cbf_n_pols : int
        The number of polarisations in the data stream
    antenna_mask : list of strings
        List of antennas present in the data stream
    host : str
        Bind hostname for katcp server
    port : int
        Bind port for katcp server
    num_buffers : int
        The number of buffers to use- this will create a pipeline thread for each buffer
        and an extra accumulator thread to read the spead stream.
    buffer_maxsize : float
        The maximum size of the buffer. Memory for each buffer will be allocated at first
        and then populated by the accumulator from the spead stream.
    auto : bool
        True for autocorrelations included in the data, False for cross-correlations only
    l0_endpoint : endpoint
        Endpoint to listen to for L0 stream, default: ':7200'
    l0_interface : str
        Name of interface to subscribe to for L0, or None to let the OS decide
    l1_endpoint : endpoint
        Destination endpoint for L1 stream, default: '127.0.0.1:7202'
    l1_rate : float
        Rate for L1 stream transmission, default 5e7
    l1_level : int
        Data to transmit to L1: 0 - none, 1 - target only, 2 - all
    mproc : bool
        True for control via multiprocessing, False for control via threading
    param_file : string
        File of default pipeline parameters
    report_path : string
        Path under which to save pipeline report
    log_path : string
        Path for pipeline logs
    """

    logger.info('Pipeline system input parameters')
    logger.info('   - antenna mask: %s', antenna_mask)
    logger.info('   - number of channels: %d', cbf_n_chans)
    logger.info('   - number of polarisation products: %d', cbf_n_pols)

    # threading or multiprocessing imports
    if mproc:
        logger.info("Using multiprocessing")
        import multiprocessing
    else:
        logger.info("Using threading")
        import multiprocessing.dummy as multiprocessing

    # extract data shape parameters
    #   argument parser traversed TS config to find these
    if antenna_mask is not None:
        ts.add('antenna_mask', antenna_mask, immutable=True)
    elif 'antenna_mask' not in ts:
        raise RuntimeError("No antenna_mask set.")
    if len(ts.antenna_mask) < 4:
        # if we have less than four antennas, no katsdpcal necessary
        logger.info('Only %d antenna(s) present - stopping katsdpcal', len(ts.antenna_mask))
        return

    # deal with required input parameters
    if cbf_n_chans is not None:
        ts.add('cbf_n_chans', cbf_n_chans, immutable=True)
    elif 'cbf_n_chans' not in ts:
        raise RuntimeError("No cbf_n_chans set.")
    if cbf_n_pols is not None:
        ts.add('cbf_n_pols', cbf_n_pols, immutable=True)
    elif 'cbf_n_pols' not in ts:
        logger.warning(
            'Number of polarisation inputs cbf_n_pols not set. Setting to default value 4')
        ts.add('cbf_n_pols', 4, immutable=True)

    # initialise TS from default parameter file
    #   defaults are used only for parameters missing from the TS
    if param_file == '':
        if ts.cbf_n_chans == 4096:
            param_filename = 'pipeline_parameters_meerkat_ar1_4k.txt'
            param_file = os.path.join(param_dir, param_filename)
            logger.info('Parameter file for 4k mode: %s', param_file)
            rfi_filename = 'rfi_mask.pickle'
            rfi_file = os.path.join(rfi_dir, rfi_filename)
            logger.info('RFI mask file for 4k mode: %s', rfi_file)
        else:
            param_filename = 'pipeline_parameters_meerkat_ar1_32k.txt'
            param_file = os.path.join(param_dir, param_filename)
            logger.info('Parameter file for 32k mode: %s', param_file)
            rfi_filename = 'rfi_mask32K.pickle'
            rfi_file = os.path.join(rfi_dir, rfi_filename)
            logger.info('RFI mask file for 32k mode: %s', rfi_file)
    else:
        logger.info('Parameter file: %s', param_file)
    logger.info('Inputting Telescope State parameters from parameter file.')
    ts_from_file(ts, param_file, rfi_file)
    # telescope state logs for debugging
    logger.info('Telescope state parameters:')
    for keyval in ts.keys():
        if keyval not in ['sdp_l0_bls_ordering', 'cbf_channel_freqs']:
            logger.info('%s : %s', keyval, ts[keyval])
    logger.info('Telescope state config graph:')
    log_dict(ts.config)

    # set up TS for pipeline use
    logger.info('Setting up Telescope State parameters for pipeline.')
    setup_ts(ts)

    nant = len(ts.cal_antlist)
    # number of baselines (may include autocorrelations)
    nbl = nant*(nant+1)/2 if auto else nant*(nant-1)/2

    # get buffer size
    if buffer_maxsize is not None:
        ts.add('cal_buffer_size', buffer_maxsize)
    else:
        if 'cal_buffer_size' in ts:
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
    array_length = buffer_maxsize/((scale_factor*ts.cbf_n_chans*ts.cbf_n_pols*nbl) + time_factor)
    array_length = np.int(np.ceil(array_length))
    logger.info('Buffer size : %f GB', buffer_maxsize / 1e9)
    logger.info('Max length of buffer array : %d', array_length)

    # Set up empty buffers
    buffer_shape = [array_length, ts.cbf_n_chans, ts.cbf_n_pols, nbl]
    buffers = [create_buffer_arrays(buffer_shape, mproc=mproc) for i in range(num_buffers)]

    logger.info('Receiving L0 data from %s via %s',
                l0_endpoint, 'default interface' if l0_interface is None else l0_interface)
    l0_interface_address = katsdpservices.get_interface_address(l0_interface)

    # set up inter-task synchronisation primitives.
    # passed events to indicate buffer transfer, end-of-observation, or stop
    accum_pipeline_queues = [multiprocessing.Queue() for i in range(num_buffers)]
    # signalled when the pipeline is finished with a buffer
    pipeline_accum_sems = [multiprocessing.Semaphore(value=1) for i in range(num_buffers)]
    # signalled by pipelines when they shut down or finish an observation
    pipeline_report_queue = multiprocessing.Queue()
    # other tasks send up sensor updates
    master_queue = multiprocessing.Queue()

    # Set up the pipelines (one per buffer)
    pipelines = [Pipeline(
        multiprocessing.Process, buffers[i], buffer_shape,
        accum_pipeline_queues[i], pipeline_accum_sems[i], pipeline_report_queue, master_queue,
        i, l1_endpoint, l1_level, l1_rate, ts) for i in range(num_buffers)]
    # Set up the report writer
    report_writer = ReportWriter(
        multiprocessing.Process, pipeline_report_queue, master_queue, ts, num_buffers,
        l1_endpoint, l1_level, report_path, log_path, full_log)

    # Suppress SIGINT, so that the children inherit SIG_IGN. This ensures that
    # pressing Ctrl-C in a terminal will only deliver SIGINT to the parent.
    # There is a small window here where pressing Ctrl-C will have no effect at
    # all, which could be fixed in Python 3 with signal.pthread_sigmask.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Start the child tasks.
    for task in [report_writer] + pipelines:
        if not mproc:
            task.daemon = True    # Make sure it doesn't prevent process exit
        task.start()

    # Set up the accumulator. This is done after the other processes are
    # started, because it creates a ThreadPoolExecutor, and threads and fork()
    # don't play nicely together.
    accumulator = Accumulator(buffers, buffer_shape,
                              accum_pipeline_queues, pipeline_accum_sems,
                              l0_endpoint, l0_interface_address, ts)

    ioloop = AsyncIOMainLoop()
    ioloop.install()
    server = CalDeviceServer(accumulator, pipelines, report_writer, master_queue, host, port)
    ioloop.add_callback(server.start)

    # allow remote debug connections and expose telescope state and tasks
    manhole.install(oneshot_on='USR1', locals={
        'ts': ts,
        'accumulator': accumulator,
        'pipelines': pipelines,
        'report_writer': report_writer,
        'server': server
    })

    # Now install the signal handlers (which won't be inherited).
    loop = trollius.get_event_loop()

    def signal_handler(signum, force):
        loop.remove_signal_handler(signum)
        # If we were asked to do a graceful shutdown, the next attempt should
        # be to try a hard kill, before going back to the default handler.
        if not force:
            loop.add_signal_handler(signum, signal_handler, signum, True)
        trollius.ensure_future(server.shutdown(force=force))
    loop.add_signal_handler(signal.SIGTERM, signal_handler, signal.SIGTERM, True)
    loop.add_signal_handler(signal.SIGINT, signal_handler, signal.SIGINT, False)
    logger.info('katsdpcal started')

    # Start forwarding the events from the master queue
    for sensor in pipelines[0].get_sensors():
        server.add_sensor(sensor)
    for sensor in report_writer.get_sensors():
        server.add_sensor(sensor)
    executor = concurrent.futures.ThreadPoolExecutor(1)
    asyncio_queue = trollius.Queue()
    queue_forward_task = loop.run_in_executor(
        executor, queue_forward, master_queue, asyncio_queue, loop)

    # Process incoming events.
    try:
        while True:
            event = yield From(asyncio_queue.get())
            if isinstance(event, StopEvent):
                break
            elif isinstance(event, SensorReadingEvent):
                try:
                    sensor = server.get_sensor(event.name)
                except ValueError:
                    logger.warn('Received update for unknown sensor %s', event.name)
                else:
                    sensor.set(event.reading.timestamp,
                               event.reading.status,
                               event.reading.value)
            else:
                logger.warn('Unknown event %r', event)
    except Exception as error:
        logger.error('Unknown error: %s', error, exc_info=True)
        yield From(server.shutdown(True))       # Ensure that queue_forward_task can exit

    yield From(queue_forward_task)
    logger.info('Joined queue_forward_task')
    executor.shutdown()
    yield From(to_asyncio_future(server.stop()))
    logger.info('Server stopped')


@trollius.coroutine
def main():
    opts = parse_opts()

    # set up logging. The Formatter class is replaced so that all log messages
    # show the process/thread name.
    if opts.notthreading:
        logging.Formatter = ProcessLogFormatter
    else:
        logging.Formatter = ThreadLogFormatter
    log_name = 'pipeline.log'
    log_path = os.path.abspath(opts.log_path)
    setup_logger(log_name, log_path)

    yield From(run_threads(
        opts.telstate,
        cbf_n_chans=opts.cbf_channels, cbf_n_pols=opts.cbf_pols, antenna_mask=opts.antenna_mask,
        host=opts.host, port=opts.port,
        num_buffers=opts.num_buffers, buffer_maxsize=opts.buffer_maxsize, auto=not(opts.no_auto),
        l0_endpoint=opts.l0_spectral_spead[0], l0_interface=opts.l0_spectral_interface,
        l1_endpoint=opts.l1_spectral_spead,
        l1_rate=opts.l1_rate, l1_level=opts.l1_level, mproc=opts.notthreading,
        param_file=opts.parameter_file,
        report_path=opts.report_path, log_path=log_path, full_log=log_name))


if __name__ == '__main__':
    trollius.get_event_loop().run_until_complete(main())
    trollius.get_event_loop().close()
