import spead64_48 as spead
import threading
import time

from katcal.reduction import pipeline
from katsdptelstate.telescope_state import TelescopeState

import logging
import socket
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import numpy as np

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

def get_ant_bls(pol_bls_ordering):
    """
    Given baseline list with polarisation information, return pure antenna list
    e.g. from [['ant0h','ant0h'],['ant0v','ant0v'],['ant0h','ant0v'],['ant0v','ant0h'],['ant1h','ant1h']...]
     to [['ant0,ant0'],['ant1','ant1']...]
    """
    
    # get antenna only names from pol-included bls orderding
    ant_bls = np.array([[a1[:-1],a2[:-1]] for a1,a2 in pol_bls_ordering])
    ant_dtype = ant_bls[0,0].dtype 
    # get list without repeats (ie no repeats for pol)
    #     start with array of empty strings, shape (num_baselines x 2)
    bls_ordering = np.empty([len(ant_bls)/4,2],dtype=ant_dtype) 

    # iterate through baseline list and only include non-repeats
    #   I know this is horribly non-pythonic. Fix later.
    bls_ordering[0] = ant_bls[0]
    bl = 0
    for c in ant_bls: 
        if not np.all(bls_ordering[bl] == c): 
            bl += 1
            bls_ordering[bl] = c
            
    return bls_ordering
    
def get_pol_bls(bls_ordering,pol):
    """
    Given baseline ordering and polarisation ordering, return full baseline-pol ordering array
    """
    pol_ant_dtype = np.array(bls_ordering[0,0]+'h').dtype 
    nbl = bls_ordering.shape[0]
    pol_bls_ordering = np.empty([nbl*4,2],dtype=pol_ant_dtype)
    for i,p in enumerate(pol):
        for b,bls in enumerate(bls_ordering):
            pol_bls_ordering[nbl*i+b] = bls[0]+p[0], bls[1]+p[1]    
    return pol_bls_ordering
    
def get_reordering(antlist,bls_ordering):
    """
    Determine reordering necessary to change given bls_ordering into desired ordering
    """
    antlist = antlist.split(',')
    nants = len(antlist)
    nbl = nants*(nants+1)/2
    
    # determined desired correlator product ordering
    #   first index
    bls_wanted_1 = np.array([])
    for a,i in enumerate(antlist[:-1]):
        bls_wanted_1 = np.hstack([bls_wanted_1,[i]*(nants-a-1)])
    bls_wanted_1 = np.hstack([bls_wanted_1,antlist])
    #   second index
    bls_wanted_2 = np.array([], dtype=np.int)
    mod_antlist = antlist[1:]
    for i in (range(0,len(mod_antlist))):
        bls_wanted_2 = np.hstack([bls_wanted_2,mod_antlist[:]])
        mod_antlist.pop(0)
    bls_wanted_2 = np.hstack([bls_wanted_2,antlist])
    #   combine into single array
    bls_wanted = np.vstack([bls_wanted_1,bls_wanted_2]).T
    #   add polarisation indices    
    pol_order = np.array([['h','h'],['v','v'],['h','v'],['v','h']])
    bls_pol_wanted = get_pol_bls(bls_wanted,pol_order)
    
    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering = np.array([np.all(bls_ordering==bls,axis=1).nonzero()[0][0] for bls in bls_pol_wanted])
    # how to use this:
    #print bls_ordering[ordering]
    #print bls_ordering[ordering].reshape([4,nbl,2])    
    return ordering

class accumulator_thread(threading.Thread):
    """
    Thread which accumutates data from spead into numpy arrays
    """

    def __init__(self, buffers, scan_accumulator_conditions, l0_endpoint, telstate):
        threading.Thread.__init__(self)
        
        self.buffers = buffers
        self.telstate = telstate
        self.l0_endpoint = l0_endpoint
        self.scan_accumulator_conditions = scan_accumulator_conditions        
        self.num_buffers = len(buffers) 

        self.name = 'Accumulator_thread'
        self._stop = threading.Event()
        
        # flag for switching capture to the alternate buffer
        self._switch_buffer = False

        #Get data shape
        self.nchan = buffers[0]['vis'].shape[1]
        self.npol = buffers[0]['vis'].shape[2]
        self.nbl = buffers[0]['vis'].shape[3]
        
        # set up logging adapter for the thread
        self.accumulator_logger = ThreadLoggingAdapter(logger, {'connid': self.name})
        
    def run(self):
        """
        Thread run method. Append random vis to the vis list
        at random time.
        """
        # Initialise SPEAD stream
        self.accumulator_logger.info('Initializing SPEAD receiver')

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.l0_endpoint.multicast_subscribe(sock):
            logger.info("Subscribing to multicast address {0}".format(self.l0_endpoint.host))

        spead_stream = spead.TransportUDPrx(self.l0_endpoint.port)
        
        # Determine imput data shape
        data_shape = get_reordering(self.telstate.antenna_mask,self.telstate.cbf_bls_ordering)

        # Increment between buffers, filling and releasing iteratively
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
        tx = spead.Transmitter(spead.TransportUDPtx(self.l0_endpoint.host,self.l0_endpoint.port))
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
        """

        ig = spead.ItemGroup()
        
        start_flag = True
        array_index = -1
        data_buffer['track_start_indices'] = []
        
        max_length = data_buffer['times'].shape[0]
        prev_activity = 'none'
        prev_tags = 'none'
        
        # get names of activity and target TS keys, using TS reference antenna
        target_key = '{0}_target'.format(self.telstate.cal_refant,)
        activity_key = '{0}_activity'.format(self.telstate.cal_refant,)
        
        # receive SPEAD stream
        print 'Got heaps: ',
        for heap in spead.iterheaps(spead_stream): 
            ig.update(heap)
            print array_index, 
            array_index += 1
            
            # get activity and target tag from TS
            activity = self.telstate[activity_key]
            target = self.telstate[target_key]
            
            # accumulate list of track start time indices in the array
            #   for use in the pipeline, to index each track easily 
            if 'track' in activity and not 'track' in prev_activity:
                data_buffer['track_start_indices'].append(array_index)
                
            # break if this scan is a slew that follows a track
            #   unless previous scan was a target, in which case accumulate subsequent gain scan too
            # ********** THIS BREAKING NEEDS TO BE THOUGHT THROUGH CAREFULLY ********** 
            if ('slew' in activity and 'track' in prev_activity) and 'target' not in prev_tags:
                self.accumulator_logger.info('Accumulate break due to transition')
                break

            if start_flag: 
                start_time = ig['timestamp'] 
                start_flag = False

            # reshape data and put into relevent arrays
            data_buffer['vis'][array_index,:,:,:] = ig['correlator_data'].reshape([self.nchan,self.nbl,self.npol]).swapaxes(-1,-2)  
            data_buffer['flags'][array_index,:,:,:] = ig['flags'].reshape([self.nchan,self.nbl,self.npol]).swapaxes(-1,-2)
            data_buffer['weights'][array_index,:,:,:] = ig['weights'].reshape([self.nchan,self.nbl,self.npol]).swapaxes(-1,-2)  
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
                
            prev_activity = activity
            # extract tags from target description string
            prev_tags = target.split(',')[1]
                
        data_buffer['track_start_indices'].append(array_index)
    
        return array_index
        
# ---------------------------------------------------------------------------------------
# Pipeline 
# ---------------------------------------------------------------------------------------
               
class pipeline_thread(threading.Thread):
    """
    Thread which runs pipeline
    """

    def __init__(self, data, scan_accumulator_condition, pipenum, l1_endpoint, telstate):
        threading.Thread.__init__(self)
        self.data = data
        self.scan_accumulator_condition = scan_accumulator_condition
        self.name = 'Pipeline_thread_'+str(pipenum)
        self._stop = threading.Event()
        self.telstate = telstate
        self.l1_endpoint = l1_endpoint
        
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
            self.run_pipeline()
            
            # release condition after pipeline run finished
            self.scan_accumulator_condition.release()
            self.pipeline_logger.info('scan_accumulator_condition release by %s' %(self.name,))
        
    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()
        
    def run_pipeline(self):    
        # run pipeline calibration
        calibrated_data = pipeline(self.data,self.telstate,thread_name=self.name)
        # if target data was calibated in the pipeline, send to L1 spead
        if calibrated_data is not None:
            self.pipeline_logger.info('Transmit L1 data')
            data_to_SPEAD(calibrated_data,self.l1_endpoint)
        
# ---------------------------------------------------------------------------------------
# SPEAD transmission
# ---------------------------------------------------------------------------------------
        
def data_to_SPEAD(data,spead_endpoint):
    """
    Sends data to SPEAD stream
    
    data:
       list of: vis, flags, weights, times
    """
    
    tx = spead.Transmitter(spead.TransportUDPtx(spead_endpoint.host,spead_endpoint.port))

    # transmit data
    for i in range(len(data[-1])): # time axis

        tx_vis = data[0][i]
        tx_flags = data[1][i]
        tx_weights = data[2][i]
        tx_time = data[3][i]

        # transmit timestamps, vis, flags and weights
        transmit_item(tx, tx_time, tx_vis, tx_flags, tx_weights)
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
    
def transmit_item(tx, tx_time, tx_vis, tx_flags, tx_weights):
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
    
