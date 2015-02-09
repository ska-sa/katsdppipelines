
import numpy as np
import optparse
import sys
import copy

import pickle
import os

from katcal import plotting
from katcal import calprocs
#from katcal.calsolution import CalSolution
from katcal import parameters
from katcal.simulator import SimData
from katcal import report
from katcal.scan import Scan

from katcal.calprocs import CalSolution

from time import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ThreadLoggingAdapter(logging.LoggerAdapter):
    """
    Logging adaptor which prepends a bracketed string to the log message.
    The string is passed via a 'connid' key.    
    """
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['connid'], msg), kwargs

def rfi():
    """
    Place holder for RFI detection algorithms to come.
    """
    #print 'Some sort of RFI flagging?!'
    pass
    
def get_nearest_from_ts(ts,key,t,dt=15.0):
    """
    Given a time, extract the value of a key in the Telescope State closest to that time.
    
    Inputs:
    =======
    ts  : TelescopeState
    key : string, key to be extracted from the ts
    t   : time that value of the key is desired 
    dt  : range of times over which to search for key value, bracketing time t
    
    Returns:
    ========
    Value of a key in the Telescope State closest to time t
    
    """    
    key_list = np.array(ts.get_range(key,st=t-dt,et=t+dt))
    time_diffs = [np.float(line[1]) - t for line in key_list]
    return key_list[np.argmin(time_diffs)] 

def pipeline(data, ts, thread_name):
    """
    Pipeline calibration
    """

    # ----------------------------------------------------------    
    # set up logging adapter for the pipeline thread
    pipeline_logger = ThreadLoggingAdapter(logger, {'connid': thread_name})
    
    # ----------------------------------------------------------
    # set up timing file
    timing_file = 'timing.txt'
    if os.path.isfile(timing_file): os.remove(timing_file)
    timing_file = open("timing.txt", "w")

    # ----------------------------------------------------------
    # some general constants
    #    will mostly be in the Telescope Model later?
    #    or combination of TM and param file?

    params = parameters.set_params()

    #per_scan_plots = params['per_scan_plots']
    #closing_plots = params['closing_plots']
    REFANT = params['refant']

    # solution intervals
    bp_solint = params['bp_solint'] #seconds
    k_solint = params['k_solint'] #seconds
    k_chan_sample = params['k_chan_sample']
    g_solint = params['g_solint'] #seconds
    dump_period = ts.dump_period

    # plots per scan
    #per_scan_plots = True

    # ----------------------------------------------------------
    # extract values we need frequently from the TM
    #nant = ts.nant
    #nbl = nant*(nant+1)/2
    #nchan = ts.nchan

    # ----------------------------------------------------------
    # set initial values for fits
    bp0_h = None
    k0_h = None
    g0_h = None
    target_h = None

    # ----------------------------------------------------------
    # iterate through the track scans accumulated into the data buffer
    #    iterate backwards in time through the scans, 
    #    for the case where a gains need to be calculated from a gain scan after a target scan,
    #    for application to the target scan
    for i in range(len(data['track_start_indices'])-1,0,-1):
        # start and end indices for this track in the data buffer
        ti0 = data['track_start_indices'][i-1]
        ti1 = data['track_start_indices'][i]
        
        # start time, end time
        t0 = data['times'][ti0]
        t1 = data['times'][ti1]

        # extract scan info from the TS
        #  target string contains: 'target name, tags, RA, DEC'
        target = get_nearest_from_ts(ts,'target',t0)[0]
        scan_state = get_nearest_from_ts(ts,'scan_state',t0)[0]
        taglist = get_nearest_from_ts(ts,'tag',t0)[0]        
        # fudge for now to add delay cal tag to bpcals
        if 'bpcal' in taglist: taglist.append('delaycal')
        
        target_name = target.split(',')[0]
        pipeline_logger.info('Target: {0}'.format(target_name,))
        pipeline_logger.info('Tags:   {0}'.format(taglist,))
        
        # set up scan
        s = Scan(data, ti0, ti1, ts.dump_period, ts.antlist, ts.corr_products)

        # initial RFI flagging
        pipeline_logger.info('Preliminary flagging')
        rfi()

        run_t0 = time()
        # perform calibration as appropriate, from scan intent tags           
        if any('delaycal' in k for k in taglist):
            # ---------------------------------------
            # preliminary G solution
            pipeline_logger.info('Solving for preliminary gain on delay calibrator {0}'.format(target.split(',')[0],))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(k_solint,g0_h,REFANT)
            g_to_apply = s.interpolate(pre_g_soln)
            
            # ---------------------------------------
            # K solution
            pipeline_logger.info('Solving for delay on delay calibrator {0}'.format(target.split(',')[0],))
            k_soln = s.k_sol(k_chan_sample,k0_h,bp0_h,REFANT,pre_apply=[g_to_apply])
            
            # ---------------------------------------
            # update TS
            pipeline_logger.info('Saving delay to Telescope State')
            ts.add(k_soln.soltype,k_soln.values,ts=time()) # fix times later XXXXXXXXXXXXXXX
            
            # ---------------------------------------
            timing_file.write("Delay cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()     
            
        if any('bpcal' in k for k in taglist):  
            # ---------------------------------------
            # get K solutions to apply and interpolate it to scan timestamps
            try:
                pipeline_logger.info('Applying delay to bandpass calibrator {0}'.format(target.split(',')[0],))
                # get most recent K
                sol, soltime = ts.get_range('K')
                k_soln = CalSolution('K', sol, soltime)
                k_to_apply = s.interpolate(k_soln)
            except KeyError:
                # TS doesn't yet contain 'K'
                pipeline_logger.info('Solving for bandpass without applying delay correction')
        
            # ---------------------------------------
            # Preliminary G solution
            pipeline_logger.info('Solving for preliminary gain on bandpass calibrator {0}'.format(target.split(',')[0],))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(bp_solint,g0_h,REFANT,pre_apply=[k_to_apply])
            g_to_apply = s.interpolate(pre_g_soln)
            
            # ---------------------------------------
            # B solution
            pipeline_logger.info('Solving for bandpass on bandpass calibrator {0}'.format(target.split(',')[0],))
            b_soln = s.b_sol(bp0_h,REFANT,pre_apply=[k_to_apply,g_to_apply])

            # ---------------------------------------
            # update TS
            pipeline_logger.info('Saving bandpass to Telescope State')
            ts.add(b_soln.soltype,b_soln.values,ts=time()) # fix times later XXXXXXXXXXXXXXX
            
            # ---------------------------------------
            timing_file.write("Bandpass cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()  
            
        if any('gaincal' in k for k in taglist):
            # ---------------------------------------
            # get K and B solutions to apply and interpolate it to scan timestamps
            try:
                pipeline_logger.info('Applying delay to gain calibrator {0}'.format(target.split(',')[0],))
                # get most recent K
                sol, soltime = ts.get_range('K')
                k_soln = CalSolution('K', sol, soltime)
                k_to_apply = s.interpolate(k_soln)
            except KeyError:
                # TS doesn't yet contain 'K'
                pipeline_logger.info('Solving for gain without applying delay correction')
                   
            try:
                pipeline_logger.info('Applying bandpass to gain calibrator {0}'.format(target.split(',')[0],))
                # get most recent B
                sol, soltime = ts.get_range('B')
                b_soln = CalSolution('B', sol, soltime)
                b_to_apply = s.interpolate(b_soln)
            except KeyError:
                # TS doesn't yet contain 'B'
                pipeline_logger.info('Solving for gain without applying bandpass correction')
        
            # ---------------------------------------
            # Preliminary G solution
            pipeline_logger.info('Solving for gain on gain calibrator {0}'.format(target.split(',')[0],))
            # set up solution interval: just solve for two intervals per G scan (ignore ts g_solint for now)
            dumps_per_solint = np.ceil((ti1-ti0)/2.0)
            g_solint = dumps_per_solint*dump_period            
            g_soln = s.g_sol(g_solint,g0_h,REFANT,pre_apply=[k_to_apply,b_to_apply])
            
            # ---------------------------------------
            # update TS
            pipeline_logger.info('Saving gain to Telescope State')
            # add gains to TS, iterating through solution times 
            for v,t in zip(g_soln.values,g_soln.times): 
                ts.add(g_soln.soltype,v,ts=t) 
            
            # ---------------------------------------
            timing_file.write("Gain cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time() 
            
        if any('target' in k for k in taglist):
            # ---------------------------------------
            # get K, B and G solutions to apply and interpolate it to scan timestamps
            try:
                pipeline_logger.info('Applying delay to target {0}'.format(target.split(',')[0],))
                # get most recent K
                sol, soltime = ts.get_range('K')
                k_soln = CalSolution('K', sol, soltime)
                k_to_apply = s.interpolate(k_soln)
            except KeyError:
                # TS doesn't yet contain 'K'
                pipeline_logger.info('No delay correction applied to target data')
                   
            try:
                pipeline_logger.info('Applying bandpass to target {0}'.format(target.split(',')[0],))
                # get most recent B
                sol, soltime = ts.get_range('B')
                b_soln = CalSolution('B', sol, soltime)
                b_to_apply = s.interpolate(b_soln)
            except KeyError:
                # TS doesn't yet contain 'B'
                pipeline_logger.info('No bandpass correction applied to target data')
                
            try:
                pipeline_logger.info('Applying gains to target {0}'.format(target.split(',')[0],))
                # get G values for an hour range on either side of target scan
                gsols = np.array(ts.get_range('G',st=t0-60.*60.,et=t1+60.*60))
                print 'G: ', gsols.shape
                g_soln = CalSolution('B', gsols[:,0], gsols[:,1])
                g_to_apply = s.interpolate(g_soln)
            except KeyError:
                # TS doesn't yet contain 'G'
                pipeline_logger.info('No gain correction applied to target data')
            
            
            
