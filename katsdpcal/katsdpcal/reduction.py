
import numpy as np
import optparse
import sys
import copy

import pickle
import os

from katsdpcal import plotting
from katsdpcal import calprocs
from katsdpcal.simulator import SimData
from katsdpcal import report
from katsdpcal.scan import Scan

from rfi import threshold_avg_flagging

from katsdpcal.calprocs import CalSolution

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

def rfi(s,thresholds,av_blocks):
    """
    Place holder for RFI detection algorithms to come.
    """
    threshold_avg_flagging(s.vis,s.flags,thresholds,blocks=av_blocks,transform=np.abs)

def get_tracks(data, ts):
    """
    Determines start and end indices of each track in the data buffer

    Inputs:
    -------
    data : data buffer, dictionary
    ts : telescope state, TelescopeState

    Returns:
    --------
    start_indx : start indices of each track scan
    stop_indx : stop indices of each track scan

    """
    max_ind = data['max_index'][0]

    activity_key = '{0}_activity'.format(ts.cal_refant,)
    activity = ts.get_range(activity_key,st=data['times'][0],et=data['times'][max_ind],include_previous=True)

    start_indx, stop_indx = [], []
    for state, time in activity:
        nearest_time_indx = np.abs(time - data['times'][0:max_ind+1]).argmin()
        if 'track' in state:
            start_indx.append(nearest_time_indx)
        elif 'slew' in state:
            stop_indx.append(nearest_time_indx-1)

    # remove first slew time from stop indices
    if stop_indx[0] == -1: stop_indx = stop_indx[1:]
    # add max index in buffer to stop indices of necessary
    if len(stop_indx) < len(start_indx): stop_indx.append(max_ind)

    return start_indx, stop_indx

def get_solns_to_apply(s,ts,sol_list,logger,time_range=[]):
    """
    For a given scan, extract and interpolate specified calibration solutions from TelescopeState

    Inputs:
    -------
    s : scan, Scan
    ts : telescope state, TelescopeState
    sol_list : list of calibration solutions to extract and interpolate, list of strings

    Returns:
    --------
    solns_to_apply : list of CalSolution solutions
    """
    solns_to_apply = []

    for X in sol_list:
        ts_solname = 'cal_product_{0}'.format(X,)
        try:
            # get most recent solution value
            sol, soltime = ts.get_range(ts_solname)[0]
            if X is not 'G':
                soln = CalSolution(X, sol, soltime)
            else:
                # get G values for an hour range on either side of target scan
                t0, t1 = time_range
                gsols = ts.get_range(ts_solname,st=t0-60.*60.,et=t1+60.*60,return_format='recarray')
                solval, soltime = gsols['value'], gsols['time']
                soln = CalSolution('G', solval, soltime)

            solns_to_apply.append(s.interpolate(soln))
            logger.info('Apply {0} solution to {1}'.format(X,s.target))

        except KeyError:
            # TS doesn't yet contain 'X'
            logger.info('No {0} correction present'.format(X,))

    return solns_to_apply

def pipeline(data, ts, task_name='pipeline'):
    """
    Pipeline calibration

    Inputs:
    -------
    data : data buffer, dictionary
    ts : telescope state, TelescopeState
    task_name : name of pipeline task (used for logging), string

    Returns:
    --------
    target_starts: start indices for target scans in the data buffer, list
    target_stops: stop indices for target scans in the buffer, list
    """

    # ----------------------------------------------------------
    # set up logging adapter for the pipeline thread/process
    pipeline_logger = ThreadLoggingAdapter(logger, {'connid': task_name})

    # ----------------------------------------------------------
    # set up timing file
    # at the moment this is re-made every scan! fix later!
    timing_file = 'timing.txt'
    #print timing_file
    #if os.path.isfile(timing_file): os.remove(timing_file)
    #timing_file = open("timing.txt", "w")

    # ----------------------------------------------------------
    # extract some some commonly used constants from the TS

    # solution intervals
    bp_solint = ts.cal_bp_solint #seconds
    k_solint = ts.cal_k_solint #seconds
    k_chan_sample = ts.cal_k_chan_sample
    g_solint = ts.cal_g_solint #seconds

    dump_period = ts.sdp_l0_int_time
    n_ants = len(ts.antenna_mask.split(','))

    antlist = ts.antenna_mask.split(',')
    # refant index number in the antenna list
    refant_ind = antlist.index(ts.cal_refant)

    # get names of activity and target TS keys, using TS reference antenna
    target_key = '{0}_target'.format(ts.cal_refant,)
    activity_key = '{0}_activity'.format(ts.cal_refant,)

    # ----------------------------------------------------------
    # set initial values for fits
    bp0_h = None
    k0_h = None
    g0_h = None
    target_h = None

    # ----------------------------------------------------------
    # iterate through the track scans accumulated into the data buffer
    #    first extract track scan indices from the buffer
    #    iterate backwards in time through the scans,
    #    for the case where a gains need to be calculated from a gain scan after a target scan,
    #    for application to the target scan
    track_starts, track_stops = get_tracks(data,ts)

    target_starts, target_stops = [], []

    for ti0, ti1 in reversed(zip(track_starts, track_stops)):
        # start time, end time
        t0 = data['times'][ti0]
        t1 = data['times'][ti1]

        # extract scan info from the TS
        #  target string contains: 'target name, tags, RA, DEC'
        # The large 300s window here is to account for a target that may have been set prior to a
        # slew, prior to the current scan.
        target = ts.get_range(target_key,et=t0,include_previous=300.)[0][0]
        # add the dump period here to account for scan start and activity change being closely timed
        scan_state = ts.get_range(activity_key,et=t0+dump_period,include_previous=300.)[0][0]
        taglist = target.split(',')[1].split()
        # fudge for now to add delay cal tag to bpcals
        if 'bpcal' in taglist: taglist.append('delaycal')

        target_name = target.split(',')[0]
        pipeline_logger.info('Target: {0}'.format(target_name,))
        pipeline_logger.info('Tags:   {0}'.format(taglist,))

        # update source list
        try:
            target_list = ts.get_range('cal_info_sources',st=0,return_format='recarray')['value']
        except KeyError:
            target_list = []
        if not target in target_list: ts.add('cal_info_sources',target)

        # set up scan
        s = Scan(data, ti0, ti1, dump_period, n_ants, ts.cal_bls_lookup, target_name, chans=ts.cbf_channel_freqs)

        # initial RFI flagging
        pipeline_logger.info('Preliminary flagging')
        total_size = np.multiply.reduce(s.flags.shape)/100.
        pipeline_logger.info('  Start flags: {0:.3f}%'.format(np.sum(s.flags.view(np.bool))/total_size,))
        rfi(s,[3.0,3.0,2.0,1.6],[[3,1],[3,5],[3,8]])
        pipeline_logger.info('  New flags:   {0:.3f}%'.format(np.sum(s.flags.view(np.bool))/total_size,))

        run_t0 = time()
        # perform calibration as appropriate, from scan intent tags
        if any('delaycal' in k for k in taglist):
            # ---------------------------------------
            # preliminary G solution
            pipeline_logger.info('Solving for preliminary G on delay calibrator {0}'.format(target_name,))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(k_solint,g0_h,refant_ind)
            g_to_apply = s.interpolate(pre_g_soln)

            # ---------------------------------------
            # K solution
            pipeline_logger.info('Solving for K on delay calibrator {0}'.format(target_name,))
            k_soln = s.k_sol(k_chan_sample,k0_h,bp0_h,refant_ind,pre_apply=[g_to_apply])

            # ---------------------------------------
            # update TS
            pipeline_logger.info('Saving K to Telescope State')
            ts.add(k_soln.ts_solname,k_soln.values,ts=k_soln.times)

            # ---------------------------------------
            #timing_file.write("K cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()

        if any('bpcal' in k for k in taglist):
            # ---------------------------------------
            # get K solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K'],pipeline_logger)

            # ---------------------------------------
            # Preliminary G solution
            pipeline_logger.info('Solving for preliminary G on bandpass calibrator {0}'.format(target.split(',')[0],))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(bp_solint,g0_h,refant_ind,pre_apply=solns_to_apply)
            g_to_apply = s.interpolate(pre_g_soln)

            # ---------------------------------------
            # B solution
            pipeline_logger.info('Solving for B on bandpass calibrator {0}'.format(target.split(',')[0],))
            solns_to_apply.append(g_to_apply)
            b_soln = s.b_sol(bp0_h,refant_ind,pre_apply=solns_to_apply)

            # ---------------------------------------
            # update TS
            pipeline_logger.info('Saving B to Telescope State')
            ts.add(b_soln.ts_solname,b_soln.values,ts=b_soln.times)

            # ---------------------------------------
            #timing_file.write("B cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()

        if any('gaincal' in k for k in taglist):
            # ---------------------------------------
            # get K and B solutions to apply and interpolate them to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K','B'],pipeline_logger)

            # ---------------------------------------
            # G solution
            pipeline_logger.info('Solving for G on gain calibrator {0}'.format(target.split(',')[0],))
            # set up solution interval: just solve for two intervals per G scan (ignore ts g_solint for now)
            dumps_per_solint = np.ceil((ti1-ti0)/2.0)
            g_solint = dumps_per_solint*dump_period
            g_soln = s.g_sol(g_solint,g0_h,refant_ind,pre_apply=solns_to_apply)

            # ---------------------------------------
            # update TS
            pipeline_logger.info('Saving G to Telescope State')
            # add gains to TS, iterating through solution times
            for v,t in zip(g_soln.values,g_soln.times):
                ts.add(g_soln.ts_solname,v,ts=t)

            # ---------------------------------------
            #timing_file.write("G cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()

        if any('target' in k for k in taglist):
            # ---------------------------------------
            # get K, B and G solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K','B','G'],pipeline_logger,time_range=[t0,t1])
            # apply solutions
            for soln in solns_to_apply:
                s.apply(soln, inplace=True)

            # accumulate list of target scans to be streamed to L1
            target_starts.append(ti0)
            target_stops.append(ti1)
        
    return target_starts, target_stops



