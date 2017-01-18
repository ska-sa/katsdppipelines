from . import spead2
from . import send
from . import recv

from .reduction import pipeline
from . import calprocs

from katcp import DeviceServer, Sensor
import tornado.gen
import tornado.ioloop
import numpy as np
import time
import traceback

import logging
logger = logging.getLogger(__name__)


class TaskLoggingAdapter(logging.LoggerAdapter):
    """
    This example adapter expects the passed in dict-like object to have a
    'connid' key, whose value in brackets is prepended to the log message.
    """
    def process(self, msg, kwargs):
        return '[{0}] {1}'.format(self.extra['connid'], msg), kwargs


# ---------------------------------------------------------------------------------------
# Calibration system device server
# ---------------------------------------------------------------------------------------


class CalibrationServer(DeviceServer):
    """
    katcp DeviceServer for running the calibration system (accumulator and pipelines)

    Parameters
    ----------
    host           : katcp host
    port           : katcp port
    control_method : multiprocessing or threading
    control_task   : Process or Thread, corresponds to control_method
    spead_params   : SPEAD parameters, dictionary
    logger         : logger

    Attributes
    ----------
    control_method : multiprocessing or threading
    control_task   : Process or Thread, corresponds to control_method
    spead_params   : SPEAD parameters, dictionary
    num_buffers    : number of buffers, int
    run_accumulator_sensor_thread : flag for sensor polling, boolean
    logger         : logger
    """
    def __init__(self, host, port, control_method, control_task, spead_params, logger=None, *args, **kwargs):
        self.control_method = control_method
        self.control_task = control_task
        self.spead_params = spead_params
        self.num_buffers = 0
        # flag for maintaining sensor polling
        self.run_accumulator_sensor_thread = True

        self.logger = logger if logger is not None else TaskLoggingAdapter(logging.getLogger(__name__), {'connid': 'cal_server'})
        super(CalibrationServer, self).__init__(host, port, logger=self.logger, *args, **kwargs)

    def setup_sensors(self):
        """
        Set up katcp sensors

        This is a required method when subclassing DeviceServer
          for the moment we just have an accumulation index sensor
        """
        s = Sensor.string('accumulator-indices', description='Accumulator occupation indices')
        self.add_sensor(s)

    # For the moment we are polling to get the accumulator index sensor
    #  This would be better as a push notification, but at the moment this needs to work with either threading
    #  or multiprocessing possibilities. A push sensor is possible for the threading case, but for multiprocessing
    #  the accumulator index is read via shared memory and a push sensor isn't possible (I think?)
    #  ** will possibly change this later **
    @tornado.gen.coroutine
    def accumulator_sensor_sampling_loop(self, period=1.0):
        """
        Sensor polling thread

        Parameters
        ----------
        period : Polling period, seconds
        """
        while self.run_accumulator_sensor_thread:
            indices = self.accumulator.get_buffer_occupation()
            self.get_sensor('accumulator-indices').set_value(' '.join(str(i) for i in indices))
            yield tornado.gen.sleep(period)

    def _initialise(self, ts, num_buffers, data_shape):
        """
        Initialise the calibration system:
          - set up parameters
          - create empty buffers
          - set up conditions on the buffers

        Parameters
        ----------
        ts          : telescope state
        num_buffers : number of buffers
        data_shape  : shape of the vis, weight, flag data
        """
        # set up parameters
        self.num_buffers = num_buffers
        self.data_shape = data_shape
        self.ts = ts

        # set up empty buffers
        self.buffers = create_buffer_arrays(num_buffers, data_shape, self.control_method)

        # set up conditions on the buffers
        self.conditions = [self.control_method.Condition() for i in range(num_buffers)]

    @request(Str(), Str(), Str())
    @return_reply(Str())
    def request_start(self, req, ts, num_buffers, data_shape):
        """
        Start the server running and initialise the calibration system

        Parameters
        ----------
        ts          : telescope state
        num_buffers : number of buffers
        data_shape  : shape of the vis, weight, flag data
        """
        super(CalibrationServer, self).start()
        self._initialise(ts, num_buffers, data_shape)
        return ('ok','Calibration pipeline server started sucessfully')

    @request()
    @return_reply(Str())
    def request_join(self, req):
        """
        Stop sensor thread and stop server
        """
        # end accumulator sensor polling thread
        self.run_accumulator_sensor_thread = False
        super(CalibrationServer, self).join()
        return ('ok','Calibration pipeline server stopped sucessfully')

    @request()
    @return_reply(Str())
    def request_capture_start(self, req):
        """
        Start the accumulator and pipelines running, in anticipation of data flowing
        """
        # Set up the accumulator
        self.accumulator = init_accumulator_control(self.control_method, self.control_task, self.buffers, self.data_shape, self.conditions, self.spead_params['l0_endpoint'], self.ts)
        # start accumulator sensor polling thread
        self.ioloop.add_callback(self.accumulator_sensor_sampling_loop)

        # Set up the pipelines (one per buffer)
        self.pipelines = [init_pipeline_control(self.control_method, self.control_task, self.buffers[i], self.data_shape, self.conditions[i], i, \
            self.spead_params['l1_endpoint'], self.spead_params['l1_level'], self.spead_params['l1_rate'], self.ts) for i in range(self.num_buffers)]

        # Start the pipeline threads
        map(lambda x: x.start(), self.pipelines)
        # Start the accumulator thread
        self.accumulator.start()
        self.logger.info('Waiting for L0 data')

        try:
            # run tasks until the observation has ended
            while all_alive([self.accumulator]+self.pipelines) and not self.accumulator.obs_finished():
                time.sleep(0.1)
        except Exception as e:
            self.logger.exception('Unknown error: {0}'.format(e,))
            force_shutdown(self.accumulator, self.pipelines)

        return ('ok','Accumulator and pipelines started successfully')

    @request()
    @return_reply(Str())
    def request_capture_done(self, req):
        """
        Shut down calibration system
        """
        # closing steps
        #  - stop pipelines first so they recieve correct signal before accumulator acquires the condition
        map(lambda x: x.stop(), self.pipelines)
        self.logger.info('Pipelines stopped')
        # then stop accumulator (releasing conditions)
        self.accumulator.stop_release()
        self.logger.info('Accumulator stopped')

        # join tasks
        self.accumulator.join()
        self.logger.info('Accumulator task closed')
        # wait till all pipeline runs finish then join
        while any_alive(self.pipelines):
            map(lambda x: x.join(), self.pipelines)
        self.logger.info('Pipeline tasks closed')

        # send stop packet to close of L1 transmission, if necessary
        #   -- this needs to be done better later
        if self.spead_params['l1_level'] != 0:
            # send L1 stop transmission
            #   wait for a couple of secs before ending transmission
            time.sleep(2.0)
            end_transmit(self.spead_params['l1_endpoint'].host, self.spead_params['l1_endpoint'].port)
            self.logger.info('L1 stream ended')
        self.logger.info('Observation closed')

        return ('ok','Accumulator and pipelines sucessfully shut down')


def create_buffer_arrays(num_buffers, buffer_shape, control_method=None):
    """
    Create empty buffer record using specified dimensions
    """
    return [create_buffer(buffer_shape, control_method) for i in range(num_buffers)]


def create_buffer(buffer_shape, control_method=None):
    """
    Create empty buffer record using specified dimensions

    Assume threading case unless multiprocessing is specified as the control method.
    """
    if 'multiprocessing' in control_method.__name__:
        return create_buffer_arrays_multiprocessing(buffer_shape, control_method)
    else:
        return create_buffer_arrays_threading(buffer_shape)


def create_buffer_arrays_multiprocessing(buffer_shape, control_method):
    """
    Create empty buffer record using specified dimensions,
    for multiprocessing shared memory appropriate buffers
    """
    # import data types
    from ctypes import c_ubyte, c_double, c_int, c_float
    # set up buffer dictionary
    data = {}
    array_length = buffer_shape[0]
    buffer_size = reduce(lambda x, y: x*y, buffer_shape)
    data['vis'] = control_method.RawArray(c_float, buffer_size*2)  # times two for complex
    data['vis'].shape = buffer_shape
    data['flags'] = control_method.RawArray(c_ubyte, buffer_size)
    data['weights'] = control_method.RawArray(c_float, buffer_size)
    data['times'] = control_method.RawArray(c_double, array_length)
    data['current_index'] = control_method.sharedctypes.RawArray(c_int, 1)
    return data


def create_buffer_arrays_threading(buffer_shape):
    """
    Create empty buffer record using specified dimensions,
    for thread appropriat numpy buffers
    """
    # set up buffer dictionary
    data = {}
    data['vis'] = np.empty(buffer_shape, dtype=np.complex64)
    data['flags'] = np.empty(buffer_shape, dtype=np.uint8)
    data['weights'] = np.empty(buffer_shape, dtype=np.float32)
    data['times'] = np.empty(buffer_shape[0], dtype=np.float)
    data['current_index'] = np.zeros([1], dtype=np.int32)
    return data


def all_alive(process_list):
    """
    Check if all of the process in process list are alive and return True
    if they are or False if they are not.

    Inputs
    ======
    process_list:  list of multpirocessing.Process objects
    """

    alive = True
    for process in process_list:
        alive = alive and process.is_alive()
    return alive


def any_alive(process_list):
    """
    Check if any of the process in process list are alive and return True
    if any are, False otherwise.

    Inputs
    ======
    process_list:  list of multpirocessing.Process objects
    """
    alive = False
    for process in process_list:
        alive = alive or process.is_alive()
    return alive


def force_shutdown(accumulator, pipelines):
    # forces pipeline threads to shut down
    accumulator.stop()
    accumulator.join()
    # pipeline needs to be terminated, rather than stopped,
    # to end long running reduction.pipeline function
    map(lambda x: x.terminate(), pipelines)


# ---------------------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------------------


def init_accumulator_control(control_method, control_task, buffers, buffer_shape, data_conditions, rx_endpoint, telstate):

    class accumulator_control(control_task):
        """
        Task (Process or Thread) which accumutates data from SPEAD into a buffer

        Parameters
        ----------
        control_method  : pipeline control method, threading or multiprocessing
        buffers         : data buffer
        buffer_shape    : shape of vis, weights, flags
        data_conditions : conditions on the data buffers
        rx_endpoint     : SPEAD receiver endpoint
        telstate        : telescope state

        Attributes
        ----------
        buffers         : data buffers
        num_buffers     : number of suffers
        buffer_shape    : shape of vis, weights, flags
        max_length      : time dimension of buffer, maximum number of timestamps that cam be accumulated, int
        nchan           : number of frequency channels, int
        npol            : number of polarisations, int
        nbl             : number of baselines, int
        ordering        : baseline ordering
        rx_endpoint     : SPEAD receiver endpoint
        data_conditions : conditions on the data buffers
        telstate        : telescope state
        name            : 'Accumulator'
        accumulator_logger : logger
        """
        def __init__(self, control_method, buffers, buffer_shape, data_conditions, rx_endpoint, telstate):
            super(accumulator_control, self).__init__()

            self.buffers = buffers
            self.num_buffers = len(buffers)
            # Get data shape
            self.buffer_shape = buffer_shape
            self.max_length = buffer_shape[0]
            self.nchan = buffer_shape[1]
            self.npol = buffer_shape[2]
            self.nbl = buffer_shape[3]

            # baseline ordering, to use in re-ordering the data
            #   will be set when data starts to flow
            self.ordering = None

            self.data_conditions = data_conditions
            self.rx_endpoint = rx_endpoint
            self.telstate = telstate

            self.name = 'Accumulator'
            self._control_method = control_method
            self._stop = control_method.Event()
            self._obsend = control_method.Event()

            # set up logging adapter for the task
            self.accumulator_logger = TaskLoggingAdapter(logger, {'connid': self.name})

        def set_accumulation_index(self, current_buffer, index):
            """
            Set the accumulation index in the given buffer

            Parameters
            ----------
            current_buffer : buffer
            index          : index to set as current accumulation index
            """
            # set index in the buffer
            if 'multiprocessing' in str(self._control_method):
                # multiprocessing case
                current_buffer['current_index'][0] = index
            else:
                # threading case
                current_buffer['current_index'] = np.atleast_1d(np.array(index))

        def get_buffer_occupation(self):
            """
            Returns the accumulation index in each buffer
            """
            return [b['current_index'][0] for b in self.buffers]

        def run(self):
            """
            Task (Process or Thread) run method. Passes data conditions around and runs the accumulator.
            """
            # if we are usig multiprocessing, view ctypes array in numpy format
            if 'multiprocessing' in str(self._control_method): self.buffers_to_numpy()

            # Initialise SPEAD receiver
            self.accumulator_logger.info('Initializing SPEAD receiver')
            rx = recv.Stream(spead2.ThreadPool(), bug_compat=spead2.BUG_COMPAT_PYSPEAD_0_5_2)
            rx.add_udp_reader(self.rx_endpoint.port, bind_hostname=self.rx_endpoint.host, max_size=9172)

            # Increment between buffers, filling and releasing iteratively
            # Initialise current buffer counter
            current_buffer = -1
            while not self._stop.is_set():
                # Increment the current buffer
                current_buffer = (current_buffer+1)%self.num_buffers

                self.data_conditions[current_buffer].acquire()
                self.accumulator_logger.info('data_condition {0} acquired by {1}'.format(current_buffer, self.name))

                # accumulate data scan by scan into buffer arrays
                self.accumulator_logger.info('max buffer length {0}'.format(self.max_length,))
                self.accumulator_logger.info('accumulating into buffer {0}'.format(current_buffer,))
                try:
                    self.accumulate(rx, current_buffer)
                except Exception as e:
                    # if something goes wrong give up on this accumulator run
                    self.accumulator_logger.error('>>>>> stopping accumulator run <<<<<')
                    self.accumulator_logger.exception('Exception: {0}'.format(e,))
                    # print full traceback for debug purposes
                    traceback.print_exc()

                max_indx = self.buffers[current_buffer]['current_index'][0]+1
                self.accumulator_logger.info('Accumulated {0} timestamps'.format(max_indx,))

                # awaken pipeline task that was waiting for condition lock
                self.data_conditions[current_buffer].notify()
                self.accumulator_logger.info('data_condition {0} notification sent by {1}'.format(current_buffer, self.name))
                # release pipeline task that was waiting for condition lock
                self.data_conditions[current_buffer].release()
                self.accumulator_logger.info('data_condition {0} released by {1]'.format(current_buffer, self.name))

                time.sleep(0.2)

        def stop_release(self):
            """
            Stop the accumulator and release the conditions on the buffers
            """
            # stop accumulator
            self.stop()
            # close off data_conditions
            #  - necessary for closing pipeline task which may be waiting on condition
            for scan_accumulator in self.data_conditions:
                scan_accumulator.acquire()
                scan_accumulator.notify()
                scan_accumulator.release()

        def stop(self):
            """
            Stop the accumulator (also send stop packet if observation is still running)
            """
            # set stop event
            self._stop.set()
            # stop SPEAD stream receival manually if the observation is still running
            if not self.obs_finished(): self.capture_stop()

        def obs_finished(self):
            """
            Returns True if observation end event has been set
            """
            return self._obsend.is_set()

        def stopped(self):
            """
            Returns True if observation stop event has been set
            """
            return self._stop.is_set()

        def capture_stop(self):
            """
            Send stop packet to force shut down of SPEAD receiver
            """
            self.accumulator_logger.info('stop SPEAD receiver')
            # send the stop only to the local receiver
            end_transmit('127.0.0.1', self.rx_endpoint.port)

        def buffers_to_numpy(self):
            """
            View ctype data buffers as numpy arrays
            """
            for current_buffer in self.buffers:
                current_buffer['vis'] = np.frombuffer(current_buffer['vis'], dtype=np.float32).view(np.complex64)
                current_buffer['vis'].shape = self.buffer_shape
                current_buffer['flags'] = np.frombuffer(current_buffer['flags'], dtype=np.uint8)
                current_buffer['flags'].shape = self.buffer_shape
                current_buffer['weights'] = np.frombuffer(current_buffer['weights'], dtype=np.float32)
                current_buffer['weights'].shape = self.buffer_shape
                current_buffer['times'] = np.frombuffer(current_buffer['times'], dtype=np.float64)
                current_buffer['current_index'] = np.frombuffer(current_buffer['current_index'], dtype=np.int32)

        def set_ordering_parameters(self):
            """
            Determine re-ordering necessary to convert from supplied bls ordering to desired bls ordering
            """
            self.ordering, bls_order, pol_order = calprocs.get_reordering(self.telstate.antenna_mask,self.telstate.cbf_bls_ordering)
            # determine lookup list for baselines
            bls_lookup = calprocs.get_bls_lookup(self.telstate.antenna_mask, bls_order)
            # save these to the telescope state for use in the pipeline/elsewhere
            self.telstate.add('cal_bls_ordering', bls_order)
            self.telstate.add('cal_pol_ordering', pol_order)
            self.telstate.add('cal_bls_lookup', bls_lookup)

        def accumulate(self, rx, buffer_index):
            """
            Accumulates spead data into buffer, until accumulation end condition is reached:
             * case 1 -- actitivy change (unless gain cal following target)
             * case 2 -- beamformer phase up ended
             * case 3 -- buffer capacity limit reached
             * case 4 -- time limit reached (may be replaced later?)

            SPEAD item groups contain:
               correlator_data
               flags
               weights
               timestamp

            Parameters
            ----------
            rx           : SPEAD receiver
            buffer_index : index of current buffer
            """

            start_flag = True
            array_index = -1

            prev_activity = 'none'
            prev_activity_time = 0.
            prev_target_tags = 'none'
            prev_target_name = 'none'

            # set data buffer for writing
            data_buffer = self.buffers[buffer_index]

            # get names of activity and target TS keys, using TS reference antenna
            target_key = '{0}_target'.format(self.telstate.cal_refant,)
            activity_key = '{0}_activity'.format(self.telstate.cal_refant,)

            # set observation end flag for tracking observation end across multiple scans
            obs_end_flag = True
            # sensor parameters which will be set from telescope state when data starts to flow
            cbf_sync_time = None
            data_ts = None

            # receive SPEAD stream
            ig = spead2.ItemGroup()

            self.accumulator_logger.info('waiting to start accumulating data')
            for heap in rx:
                ig.update(heap)
                if len(ig.keys()) < 1:
                    self.accumulator_logger.info('==== empty stop packet received ====')
                    continue

                # get sync time from TS, if it is present (if it isn't present, don't process this dump further)
                if cbf_sync_time is None:
                    if self.telstate.has_key('cbf_sync_time'):
                        cbf_sync_time = self.telstate.cbf_sync_time
                        self.accumulator_logger.info(' - set cbf_sync_time')
                    else:
                        self.accumulator_logger.warning('cbf_sync_time absent from telescope state - ignoring dump')
                        continue

                # get activity and target tag from telescope state
                data_ts = ig['timestamp'].value + cbf_sync_time
                activity_full = []
                if self.telstate.has_key(activity_key):
                    activity_full = self.telstate.get_range(activity_key, et=data_ts, include_previous=True)
                if not activity_full:
                    self.accumulator_logger.info('no activity recorded for reference antenna {0} - ignoring dump'.format(self.telstate.cal_refant,))
                    continue
                activity, activity_time = activity_full[0]

                # if this is the first scan of the observation, set up some values
                if start_flag:
                    unsync_start_time = ig['timestamp'].value
                    prev_activity_time = activity_time

                    # when data starts to flow, set the baseline ordering parameters for re-ordering the data
                    self.set_ordering_parameters()
                    self.accumulator_logger.info(' - set pipeline data ordering parameters')

                    self.accumulator_logger.info('accumulating data from targets:')

                # get target time from telescope state, if it is present (if it isn't present, set to unknown)
                if self.telstate.has_key(target_key):
                    target = self.telstate.get_range(target_key, et=data_ts, include_previous=True)[0][0]
                    if target == '': target = 'unknown'
                else:
                    self.accumulator_logger.warning('target description {0} absent from telescope state'.format(target_key))
                    target = 'unknown'
                # extract name and tags from target description string
                target_split = target.split(',')
                target_name = target_split[0]
                target_tags = target_split[1] if len(target_split) > 1 else 'unknown'
                if (target_name != prev_target_name) or start_flag:
                    # update source list if necessary
                    target_list = self.telstate.get_range('cal_info_sources', st=0, return_format='recarray')['value'] if self.telstate.has_key('cal_info_sources') else []
                    if target_name not in target_list: self.telstate.add('cal_info_sources', target_name, ts=data_ts)

                # print name of target and activity type, if activity has changed or start of accumulator
                if start_flag or (activity_time != prev_activity_time):
                    self.accumulator_logger.info(' - {0} ({1})'.format(target_name,activity))
                start_flag = False

                # increment the index indicating the position of the data in the buffer
                array_index += 1
                # reshape data and put into relevent arrays
                data_buffer['vis'][array_index,:,:,:] = ig['correlator_data'].value[:,self.ordering].reshape([self.nchan,self.npol,self.nbl])
                data_buffer['flags'][array_index,:,:,:] = ig['flags'].value[:,self.ordering].reshape([self.nchan,self.npol,self.nbl])
                # if we are not receiving weights in the SPEAD stream, fake them
                if 'weights' in ig.keys():
                    data_buffer['weights'][array_index,:,:,:] = ig['weights'].value[:,self.ordering].reshape([self.nchan,self.npol,self.nbl])
                else:
                    data_buffer['weights'][array_index,:,:,:] = np.empty([self.nchan,self.npol,self.nbl],dtype=np.float32)
                data_buffer['times'][array_index] = data_ts
                # incrememt array counter index for buffer
                self.set_accumulation_index(data_buffer, array_index)

                # **************** ACCUMULATOR BREAK CONDITIONS ****************
                # **** THIS BREAKING NEEDS TO BE THOUGHT THROUGH CAREFULLY *****

                # CASE 1 -- break if activity has changed (i.e. the activity time has changed)
                #   unless previous scan was a target, in which case accumulate subsequent gain scan too
                ignore_states = ['slew', 'stop', 'unknown']
                if (activity_time != prev_activity_time) and not np.any([ignore in prev_activity for ignore in ignore_states]) and ('unknown' not in target_tags) and ('target' not in prev_target_tags):
                    self.accumulator_logger.info('Accumulation break - transition')
                    obs_end_flag = False
                    break

                # CASE 2 -- beamformer special case
                if (activity_time != prev_activity_time) and ('single_accumulation' in prev_target_tags):
                    self.accumulator_logger.info('Accumulation break - single scan accumulation')
                    obs_end_flag = False
                    break

                # CASE 3 -- end accumulation if maximum array size has been accumulated
                if array_index >= self.max_length - 1:
                    self.accumulator_logger.info('Accumulate break - buffer size limit')
                    obs_end_flag = False
                    break

                # CASE 4 -- temporary mock up of a natural break in the data stream
                # may ultimately be provided by some sort of sensor?
                duration = ig['timestamp'].value - unsync_start_time
                if duration > 2000000:
                    self.accumulator_logger.info('Accumulate break due to duration')
                    obs_end_flag = False
                    break

                prev_activity = activity
                prev_activity_time = activity_time
                prev_target_tags = target_tags
                prev_target_name = target_name

            self.accumulator_logger.info('Accumulation ended')

            # if we exited the loop because it was the end of the SPEAD transmission
            if obs_end_flag:
                if data_ts is not None:
                    end_time = data_ts
                else:
                    # no data_ts variable because no data has flowed
                    self.accumulator_logger.info(' --- no data flowed ---')
                    end_time = time.time()
                self.telstate.add('cal_obs_end_time', end_time, ts=end_time)
                self.accumulator_logger.info('Observation ended')
                self._obsend.set()

    return accumulator_control(control_method, buffers, buffer_shape, data_conditions, rx_endpoint, telstate)


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------


def init_pipeline_control(control_method, control_task, data, data_shape, data_condition, pipenum, tx_endpoint, \
        tx_level, tx_rate, telstate):

    class pipeline_control(control_task):
        """
        Task (Process or Thread) which runs pipeline

        Parameters
        ----------
        control_method  : pipeline control method, threading or multiprocessing
        data            : data buffer
        data_shape      : shape of the visibilities, flags, weights
        data_condition  : condition on the data buffer
        pipenum         : number identifying this pipeline
        telstate        : telescope state
        tx_endpoint     : endpoint for SPEAD data transmission
        tx_level        : level for SPEAD data transmission
        tx_rate         : rate for SPEAD data transmission

        Attributes
        ----------
        data            : data buffer
        data_shape      : shape of the visibilities, flags, weights
        data_condition  : condition on the data buffer
        name            : pipeline name, Pipeline_<pipenum>
        telstate        : telescope state
        tx_endpoint     : endpoint for SPEAD data transmission
        tx_level        : level for SPEAD data transmission
        tx_rate         : rate for SPEAD data transmission
        stop            : pipeline stop event
        pipeline_logger : logger
        """
        def __init__(self, control_method, data, data_shape, data_condition, pipenum, tx_endpoint, tx_level, tx_rate, telstate):
            super(pipeline_control, self).__init__()

            self.data = data
            self.data_shape = data_shape
            self.data_condition = data_condition
            self.name = 'Pipeline_'+str(pipenum)

            self.telstate = telstate
            self.tx_endpoint = tx_endpoint
            self.tx_level = tx_level
            self.tx_rate = tx_rate

            self._stop = control_method.Event()

            # set up logging adapter for the task
            self.pipeline_logger = TaskLoggingAdapter(logger, {'connid': self.name})

        def run(self):
            """
            Task (Process or Thread) run method. Passes data conditions around and runs the pipeline.
            """
            while not self._stop.is_set():
                # acquire condition on data
                self.pipeline_logger.info('data_condition acquire by {0}'.format(self.name,))
                self.data_condition.acquire()

                # release lock and wait for notify from accumulator
                self.pipeline_logger.info('data_condition release and wait by {0}'.format(self.name,))
                self.data_condition.wait()

                try:
                    if not self._stop.is_set():
                        # after notify from accumulator, condition lock re-aquired
                        self.pipeline_logger.info('data_condition acquire by {0}'.format(self.name,))
                        # run the pipeline
                        self.pipeline_logger.info('Pipeline run start on accumulated data')

                        # if we are usig multiprocessing, view ctypes array in numpy format
                        if 'multiprocessing' in str(control_method): self.data_to_numpy()

                        # run the pipeline
                        self.run_pipeline()
                except Exception as e:
                    # if something goes wrong give up on this pipeline run
                    self.pipeline_logger.error('>>>>> stopping pipeline run <<<<<')
                    self.pipeline_logger.exception('Exception: {0}'.format(e,))
                    # print full traceback for debug purposes
                    traceback.print_exc()

                # release condition after pipeline run finished
                self.data_condition.release()
                self.pipeline_logger.info('data_condition release by {0}'.formt(self.name,))

        def stop(self):
            """
            Set a stop event
            """
            self._stop.set()

        def stopped(self):
            """
            Returns whether stop has been set
            """
            return self._stop.is_set()

        def data_to_numpy(self):
            """
            Convert data buffer from ctypes to numpy arrays
            """
            # note - this sometimes causes a harmless (but irritating) PEP 3118 runtime warning
            self.data['vis'] = np.ctypeslib.as_array(self.data['vis']).view(np.complex64)
            self.data['vis'].shape = self.data_shape

            self.data['flags'] = np.ctypeslib.as_array(self.data['flags'])
            self.data['flags'].shape = self.data_shape

            self.data['weights'] = np.ctypeslib.as_array(self.data['weights'])
            self.data['weights'].shape = self.data_shape

            self.data['times'] = np.ctypeslib.as_array(self.data['times'])

        def run_pipeline(self):
            """
            Runs the pipeline
            """
            # run pipeline calibration, if more than zero timestamps accumulated
            target_slices = pipeline(self.data, self.telstate, task_name=self.name) if (self.data['current_index'][0] > 0) else []

            # send data to L1 SPEAD if necessary
            if self.tx_level != 0:
                config = send.StreamConfig(max_packet_size=8972, rate=self.tx_rate)
                tx = send.UdpStream(spead2.ThreadPool(), self.tx_endpoint.host, self.tx_endpoint.port, config)
                self.pipeline_logger.info('   Transmit L1 data')
                # for streaming all of the data (not target only),
                # use the highest index in the buffer that is filled with data
                transmit_slices = [slice(0, self.data['current_index'][0]+1)] if self.tx_level == 2 else target_slices
                self.data_to_SPEAD(transmit_slices, tx)
                self.pipeline_logger.info('   End transmit of L1 data')

        def data_to_SPEAD(self, target_slices, tx):
            """
            Transmits data as SPEAD stream

            Inputs
            ------
            target_slices : list of slices
                slices for target scans in the data buffer
            tx : spead transmitter

            """
            # create SPEAD item group
            flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
            ig = send.ItemGroup(flavour=flavour)
            # set up item group with items
            ig.add_item(id=None, name='correlator_data', description="Visibilities",
                 shape=self.data['vis'][0].shape, dtype=self.data['vis'][0].dtype)
            ig.add_item(id=None, name='flags', description="Flags for visibilities",
                 shape=self.data['flags'][0].shape, dtype=self.data['flags'][0].dtype)
            # for now, just transmit flags as placeholder for weights
            ig.add_item(id=None, name='weights', description="Weights for visibilities",
                 shape=self.data['flags'][0].shape, dtype=self.data['flags'][0].dtype)
            ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                 shape=(), dtype=None, format=[('f', 64)])

            # transmit data
            for data_slice in target_slices:
                # get data for this scan, from the slice
                scan_vis = self.data['vis'][data_slice]
                scan_flags = self.data['flags'][data_slice]
                # for now, just transmit flags as placeholder for weights
                scan_weights = self.data['flags'][data_slice]
                scan_times = self.data['times'][data_slice]

                # transmit data timestamp by timestamp
                for i in range(len(scan_times)):  # time axis
                    # transmit timestamps, vis, flags and weights
                    ig['correlator_data'].value = scan_vis[i]
                    ig['flags'].value = scan_flags[i]
                    ig['weights'].value = scan_weights[i]
                    ig['timestamp'].value = scan_times[i]
                    tx.send_heap(ig.get_heap())

    return pipeline_control(control_method, data, data_shape, data_condition, pipenum, tx_endpoint, tx_level, tx_rate, telstate)

# ---------------------------------------------------------------------------------------
# SPEAD helper functions
# ---------------------------------------------------------------------------------------


def end_transmit(host, port):
    """
    Transmit a SPEAD stop packet

    Parameters
    ----------
    host : SPEAD host
    port : SPEAD port
    """
    config = send.StreamConfig(max_packet_size=8972)
    tx = send.UdpStream(spead2.ThreadPool(), host, port, config)

    flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
    heap = send.Heap(flavour)
    heap.add_end()

    tx.send_heap(heap)
