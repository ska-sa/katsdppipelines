import spead64_48 as spead
import threading
import time

from katcal.reduction import pipeline
from katsdptelstate.telescope_state import TelescopeState

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ThreadLoggingAdapter(logging.LoggerAdapter):
    """
    This example adapter expects the passed in dict-like object to have a
    'connid' key, whose value in brackets is prepended to the log message.
    """
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['connid'], msg), kwargs
        
# ---------------------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------------------

class accumulator_thread(threading.Thread):
    """
    Thread which accumutates data from spead into numpy arrays
    """

    def __init__(self, buffers, scan_accumulator_conditions, spead_port, spead_ip):
        threading.Thread.__init__(self)
        
        self.buffers = buffers
        self.spead_port = int(spead_port)
        self.spead_ip = spead_ip
        self.scan_accumulator_conditions = scan_accumulator_conditions        
        self.num_buffers = len(buffers) 

        self.name = 'Accumulator_thread'
        self._stop = threading.Event()
        
        # flag for switching capture to the alternate buffer
        self._switch_buffer = False

        #Get data shape
        self.nchan = buffers[0]['vis'].shape[1]
        self.nbl = buffers[0]['vis'].shape[2]
        self.npol = buffers[0]['vis'].shape[3]
        
        # set up logging adapter for the thread
        self.accumulator_logger = ThreadLoggingAdapter(logger, {'connid': self.name})
        
    def run(self):
        """
        Thread run method. Append random vis to the vis list
        at random time.
        """
        # Initialise SPEAD stream
        self.accumulator_logger.info('Initializing SPEAD receiver')
        spead_stream = spead.TransportUDPrx(self.spead_port)

        # Iincrement between buffers, filling and releasing iteratively
        #Initialise current buffer counter
        current_buffer=-1
        while not self._stop.isSet():
            #Increment the current buffer
            current_buffer = (current_buffer+1)%self.num_buffers
            # ------------------------------------------------------------
            # Loop through the buffers and send data to pipeline thread when accumulation terminate conditions are met.

            self.scan_accumulator_conditions[current_buffer].acquire()
            self.accumulator_logger.info('scan_accumulator_condition %d acquired by %s' %(current_buffer, self.name,))
            
            # accumulate data scan by scan into buffer arrays
            buffer_size = self.accumulate(spead_stream, self.buffers[current_buffer])
        
            # awaken pipeline thread that was waiting for condition lock
            self.scan_accumulator_conditions[current_buffer].notify()
            self.accumulator_logger.info('scan_accumulator_condition %d notification sent by %s' %(current_buffer, self.name,))
            # release pipeline thread that was waiting for condition lock
            self.scan_accumulator_conditions[current_buffer].release()
            self.accumulator_logger.info('scan_accumulator_condition %d released by %s' %(current_buffer, self.name,))

            time.sleep(0.5)
   
    def stop(self):        
        # set stop event
        self._stop.set()
        # stop SPEAD stream recieval
        self.capture_stop()
        
        # close off scan_accumulator_conditions
        #  - necessary for closing pipeline thread which may be waiting on condition
        for scan_accumulator in self.scan_accumulator_conditions:
            scan_accumulator.acquire()
            scan_accumulator.notify()
            scan_accumulator.release()

    def stopped(self):
        return self._stop.isSet()
        
    def capture_stop(self):
        """
        Send stop packed to force shut down of SPEAD receiver
        """
        print 'sending stop packet'
        tx = spead.Transmitter(spead.TransportUDPtx(self.spead_ip,self.spead_port))
        tx.end()
        
    def accumulate(self, spead_stream, data_buffer):
        """
        Accumulates spead data into arrays
           till **TBD** metadata indicates scan has stopped, or
           till array reaches max buffer size 
           
        SPEAD item groups contain:
           correlator_data
           flags
           weights
           timestamp
           state        
        """

        ig = spead.ItemGroup()
        
        start_flag = True
        array_index = -1
        data_buffer['track_start_indices'] = []
        
        max_length = data_buffer['times'].shape[0]
        prev_state = 'none'
        prev_tage = 'none'
        
        # receive SPEAD stream
        print 'Got heaps: ',
        for heap in spead.iterheaps(spead_stream): 
            ig.update(heap)
            print array_index, 
            
            array_index += 1
            # accumulate list of track start time indices in the array
            #   for use in the pipeline, to index each track easily 
            if 'track' in ig['state'] and not 'track' in prev_state:
                data_buffer['track_start_indices'].append(array_index)
                
            # break if this scan is a slew that follows a track
            #   unless previous scan was a target, in which case accumulate subsequent gain scan too
            # ********** THIS BREAKING NEEDS TO BE THOUGHT THROUGH CAREFULLY ********** 
            if ('slew' in ig['state'] and 'track' in prev_state) and 'target' not in prev_tags:
                self.accumulator_logger.info('Accumulate break due to transition')
                break

            if start_flag: 
                start_time = ig['timestamp'] 
                start_flag = False

            # reshape data and put into relevent arrays
            data_buffer['vis'][array_index,:,:,:] = ig['correlator_data'].reshape([self.nchan,self.nbl,self.npol])  
            data_buffer['flags'][array_index,:,:,:] = ig['flags'].reshape([self.nchan,self.nbl,self.npol])  
            data_buffer['weights'][array_index,:,:,:] = ig['weights'].reshape([self.nchan,self.nbl,self.npol])   
            data_buffer['times'][array_index] = ig['timestamp']

            # this is a temporary mock up of a natural break in the data stream
            # will ultimately be provided by some sort of sensor
            duration = ig['timestamp']-start_time
            if duration>2000000: 
                self.accumulator_logger.info('Accumulate break due to duration')
                break
            # end accumulation if maximum array size has been accumulated
            if array_index >= max_length - 1: 
                self.accumulator_logger.info('Accumulate break due to buffer size limit')
                break
                
            prev_state = ig['state']
            prev_tags = ig['tags']
                
        data_buffer['track_start_indices'].append(array_index)
    
        return array_index
        
# ---------------------------------------------------------------------------------------
# Pipeline 
# ---------------------------------------------------------------------------------------
               
class pipeline_thread(threading.Thread):
    """
    Thread which runs pipeline
    """

    def __init__(self, data, scan_accumulator_condition, pipenum, ts_db=1, ts_ip='127.0.0.1',
           l1_port=8891, l1_ip='127.0.0.1'):
        threading.Thread.__init__(self)
        self.data = data
        self.scan_accumulator_condition = scan_accumulator_condition
        self.name = 'Pipeline_thread_'+str(pipenum)
        self._stop = threading.Event()
        self.ts_db = ts_db
        self.ts_ip = ts_ip
        self.l1_port = int(l1_port)
        self.l1_ip = l1_ip
        
        # set up logging adapter for the thread
        self.pipeline_logger = ThreadLoggingAdapter(logger, {'connid': self.name})
    
    def run(self):
        """
        Thread run method. Runs pipeline
        """
    
        # run until stop is set   
        while not self._stop.isSet():
            # acquire condition on data
            self.pipeline_logger.info('scan_accumulator_condition acquire by %s' %(self.name,))
            self.scan_accumulator_condition.acquire()            

            # release lock and wait for notify from accumulator
            self.pipeline_logger.info('scan_accumulator_condition release and wait by %s' %(self.name,))
            self.scan_accumulator_condition.wait()
            
            # after notify from accumulator, condition lock re-aquired 
            self.pipeline_logger.info('scan_accumulator_condition acquire by %s' %(self.name,))
            # run the pipeline 
            self.pipeline_logger.info('Pipeline run start on accumulated data')
            run_pipeline(self.data,self.ts_db,self.ts_ip,self.l1_port,self.l1_ip,self.name)
            
            # release condition after pipeline run finished
            self.scan_accumulator_condition.release()
            self.pipeline_logger.info('scan_accumulator_condition release by %s' %(self.name,))
        
    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()
        
def run_pipeline(data, ts_db=1, ts_ip='127.0.0.1', l1_port=8891, l1_ip='127.0.0.1', thread_name='Pipeline'):    
    # start TS
    ts = TelescopeState(endpoint=ts_ip,db=ts_db)
    # run pipeline calibration
    calibrated_data = pipeline(data,ts,thread_name=thread_name)
    # if target data was calibated in the pipeline, send to L1 spead
    if calibrated_data is not None:
        data_to_SPEAD(calibrated_data,l1_port,l1_ip)
        
# ---------------------------------------------------------------------------------------
# SPEAD transmission
# ---------------------------------------------------------------------------------------
        
def data_to_SPEAD(data,port,host):
    """
    Sends data to SPEAD stream
    
    data:
       list of: vis, flags, weights, times
    """
    
    print 'TX: L1 Stream initializing...'
    tx = spead.Transmitter(spead.TransportUDPtx(host,port))

    # transmit data
    for i in range(len(data[-1])): # time axis

        tx_vis = data[0][i]
        tx_flags = data[1][i]
        tx_weights = data[2][i]
        tx_time = data[3][i]

        # transmit timestamps, vis, flags and weights
        transmit_ts(tx, tx_time, tx_vis, tx_flags, tx_weights)
        # delay so receiver isn't overwhelmed
        time.sleep(0.05)
            
    end_transmit(tx)
    
def end_transmit(tx):
    """
    Send stop packet to spead stream tx
    
    Parameters
    ----------
    tx       : spead stream
    """
    tx.end()
    
def transmit_ts(tx, tx_time, tx_vis, tx_flags, tx_weights):
    """
    Send spead packet containing time, visibility, flags and array state
    
    Parameters
    ----------
    tx         : spead stream
    tx_time    : timestamp, float
    tx_vis     : visibilities, complex array 
    tx_flags   : flags, int array
    tx_weights : weights, float array
    """
    ig = spead.ItemGroup()

    ig.add_item(name='timestamp', description='Timestamp',
        shape=[], fmt=spead.mkfmt(('f',64)),
        init_val=tx_time)

    ig.add_item(name='correlator_data', description='Full visibility array',
        init_val=tx_vis)

    ig.add_item(name='flags', description='Flag array',
        init_val=tx_flags)
        
    ig.add_item(name='weights', description='Weight array',
        init_val=tx_weights)

    tx.send_heap(ig.get_heap())
    
