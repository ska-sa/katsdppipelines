# Template project parameter file EVLAPipe
# Generate parameter file using EVLACal.EVLAMakeParmFile
#
# Substitutions surrounded by 'at' characters
# PROJECT     Project name (up to 12 char)
# SESSION     Session code
# BAND        Band code
# VLAFREQ     Frequency in Hz
# SPANBW      Spanned Frequency in Hz
# VLACFG      VLA configuraton ("A", "B", "CnD"...)
# DATAROOT    Root of archive data directory
# CONFIG      Configuration
# SELCHAN     Number of channels for selected configuration
# BPCAL       Bandpass calibrator list
# PHSCAL      Phase calibrator list
# AMPCAL      Amplitude calibrator list
# DLYCAL      Delay calibrator list
# PINCAL      Instrumental polarization calibrator list
# PINCAL      Instrumental polarization calibrator list
# PRLDCAL     R-L phase and delay calibrator list, for each
#             (name, R-L phase (deg at 1 GHz), RM (rad/m**2)
# REFANT      Reference antenna
# PLOTSRC     Diagnostic plot source name or None
# PLOTTIIME   Diagnostic plot timerange
# TARGET      List of target sources
#--------------------------------------------------------------------------------------------
# Project specific parameter values for EVLAPipeline
parms["seq"]           = 1                        # Sequence number for AIPS files
parms["project"]       = nam                      # Project name (12 char or less, used as AIPS Name)
parms["session"]       = ""                       # Project session code
parms["band"]          = "L"                      # Observing band
parms["dataClass"]     = cls                      # AIPS class of raw uv data
parms["fluxModel"]     = "PERLEY_BUTLER_2013.csv" # Filename of flux calibrator model (in FITS)
parms["VLAFreq"]       = obsdata["centerfreq"]    # Representive frequency
parms["VLACfg"]        = "D "                     # VLA configuraton ("A", "B", "CnD"...)

# Archive parameters
parms["doLoadArchive"] = True  # Load from archive?
parms["archRoot"]      = "NOT VLA DATA" # Root of ASDM/BDF data
parms["selBand"]       = "L"   # Selected band, def = first  
parms["selConfig"]     = 1     # Selected frequency config, def = first  
parms["selNIF"]        = 1     # Selected number of IFs, def = first  
parms["selChan"]       = obsdata["numchan"]  # Selected number of channels, def = first  

# Calibration sources/models
bpcal=obsdata['bpcal']
gaincal=obsdata['gaincal']
source=obsdata['source']

parms["BPCal"]=[]
for cal in bpcal:
    parms["BPCal"].append(cal.name)      # Bandpass calibrator

parms["doFD1"]       = True         # Do initial frequency domain flagging
parms["FD1widMW"]    = 255          # Width of the initial FD median window
parms["FD1maxRes"]   = 8.0          # Clipping level in sigma
parms["FD1TimeAvg"]  = 1.0          # time averaging in min. for initial FD flagging

parms["doMedn"]      = True         # Median editing?
parms["mednSigma"]   = 8.0          # Median sigma clipping level
parms["timeWind"]    = 1.0          # Median window width in min for median flagging
parms["avgTime"]     = 10.0/60.     # Averaging time in min
parms["avgFreq"]     = 1            # 1=>avg chAvg chans, 2=>avg all chan, 3=> avg chan and IFs
parms["chAvg"]       = 8            # number of channels to average
parms["IClip"]       = [50.0,0.1]   # IPol clipping

# Elevation flagging
parms["minElev"]    = 15.0         # Elevation flagging minimum

# Post calibration autoflag
parms["doAFFD"]      = True         # do AutoFlag frequency domain flag
parms["FDwidMW"]     = 255          # Width of the median window
parms["FDmaxRMS"]    = [50.,0.1]    # Channel RMS limits (Jy)
parms["FDmaxRes"]    = 6.           # Max. residual flux in sigma
parms["FDmaxResBL"]  = 6.           # Max. baseline residual
parms["FDbaseSel"]   = [100,375,1,0] # Channels for baseline fit
parms["FDmaxAmp"]    = 50.         # Maximum average amplitude (Jy)
parms["FDmaxV"]      = 1000000.0   # Maximum average VPol amp (Jy)

parms["doRMSAvg"]    = True         # Edit calibrators by RMSAvg?
parms["RMSAvg"]      = 3.0          # AutoFlag Max RMS/Avg for time domain RMS filtering
parms["RMSTimeAvg"]  = 1.0          # AutoFlag time averaging in min.

from KATCal import EVLACalModel,EVLAStdModel
freq = parms["VLAFreq"]
# Bandpass Calibration
calist = bpcal
BPCals = []
for cal in calist:
    BPCals.append(EVLACalModel(cal.name))
# Check for standard model
EVLAStdModel(BPCals, freq)
parms["BPCals"]          = BPCals   # Bandpass calibrator(s)

# Amp/phase calibration
calist = gaincal
PCals = []
tcals = []
for cal in calist:
    if not cal in tcals:
        PCals.append(EVLACalModel(cal.name))
        tcals.append(cal.name)
# Check for standard model
EVLAStdModel(PCals, freq)
parms["PCals"]          = PCals   # Phase calibrator(s)

calist = bpcal
ACals = []
tcals = []
# Get flux calibrator flux from model file
fluxcals = katpoint.Catalogue(file(ffdirs[0]+"/"+parms["fluxModel"]))
for cal in calist:
    fluxcal,offset=fluxcals.closest_to(cal)
    if offset*3600. < 1.5:       # Arbritray 1.5" position offset 
        cal.flux_model = fluxcal.flux_model
        if not cal in tcals:
            calflux=float(cal.flux_density(freq/1e6))
            ACals.append(EVLACalModel(cal.name,CalFlux=calflux,CalModelFlux=calflux))
            tcals.append(cal.name)

parms["ACals"]           = ACals    # Amplitude calibrators

calist = bpcal+gaincal
DCals = []
tcals = []
for cal in calist:
    if not cal in tcals:
        DCals.append(EVLACalModel(cal.name))
        tcals.append(cal.name)
# Check for standard model
EVLAStdModel(DCals, freq)
parms["DCals"]          = DCals      # delay calibrators
parms["delayZeroPhs"]   = False      # Keep phases

parms["refAnt"]        = -1   # Reference antenna

# Sample spectra
parms["plotSource"]    = parms["BPCal"][0]   # Source name or None
parms["plotTime"]      = [0.0, 5.0] # timerange

# Instrumental Poln  Cal
PClist                 = bpcal  # List of instrumental poln calibrators
parms["PCInsCals"]     = []
# Remove redundancies 
tcals = []
for cal in PClist:
    if not cal in tcals:
        parms["PCInsCals"].append(cal.name)
        tcals.append(cal.name)
parms["doPolCal"]      = len(parms["PCInsCals"])>0  # Do polarization calibration?
parms["doPol"]         = False #parms["doPolCal"]

# R-L phase/delay calibration
parms["RLPCal"]    = None         # Polarization angle (R-L phase) calibrator, IF based
parms["PCRLPhase"] = None         # R-L phase difference for RLPCal, IF based
parms["RM"]        = None         # rotation measure (rad/m^2) for RLPCal, IF based
parms["RLDCal"]    = [('3C286', 66.0, 0.0)]    #  R-L delay calibrator list, R-L phase, RM
parms["rlrefAnt"]  = -1     # Reference antenna for R-L cal, defaults to refAnt
parms["doRLDelay"] = parms["RLDCal"][0][0]!=None  # Determine R-L delay? If calibrator given
parms["doRLCal"]   = parms["RLDCal"][0][0]!=None  # Determine  R-L bandpass? If calibrator given

# Imaging

imtargets = bpcal + gaincal + source
parms["targets"]=[]
for targ in imtargets:
    parms["targets"].append(targ.name)
parms["Stokes"]  = "I"          # Stokes to image
parms["FOV"]     = 2.00         # FOV calculated for 25 m antenna not 12
parms["MBmaxFBW"]= 0.02         # max. MB fractional bandwidth
parms["PBCor"]   = False        # no PB correction
parms["antSize"] = 12.          # Antenna size for PB correction
parms["Niter"]   = 1000         # Number of iterations
parms["minFluxPSC"]  = 0.025    # Min flux density peak for phase self cal
parms["xCells"] = 15.0          # Cell size to use in y-axis
parms["yCells"] = 15.0          # Cell size to use in x-axis
# Multi frequency or narrow band?
#SpanBW = 1214000000.0
#if SpanBW<=1498000000.0*parms["MBmaxFBW"]*1.5:
#    parms["doMB"] = False
parms["doMB"]        = True
parms["BLFact"]      = 1.01          # Baseline dependent time averaging
parms["BLchAvg"]     = False         # Baseline dependent frequency averaging

# Customization
parms["IClip"]         = [30.,0.1]
parms["XClip"]         = None
parms["VClip"]         = None
# Special editing list
parms["doEditList"]  = True         # Edit using editList?
parms["editFG"]      = 2            # Table to apply edit list to
# Channel numbers after Hanning if any
parms["editList"] = [
    #{"timer":("0/00:00:0.0","5/00:00:0.0"),"Ant":[ 0,0],"IFs":[1,1],"Chans":[1,100],     "Stokes":'1111',"Reason":"No Gain"},
    #{"timer":("0/00:00:0.0","5/00:00:0.0"),"Ant":[ 0,0],"IFs":[1,1],"Chans":[375,0],     "Stokes":'1111',"Reason":"No Gain"},
    #{"timer":("0/00:00:0.0","5/00:00:0.0"),"Ant":[ 0,0],"IFs":[1,1],"Chans":[159,165],   "Stokes":'1111',"Reason":"RFI"},
    {"timer":("0/00:00:0.0","5/00:00:0.0"),"Ant":[ 0,0],"IFs":[1,1],"Chans":[1,100],     "Stokes":'1111',"Reason":"No Gain"},
    {"timer":("0/00:00:0.0","5/00:00:0.0"),"Ant":[ 0,0],"IFs":[1,1],"Chans":[375,0],     "Stokes":'1111',"Reason":"No Gain"},
    {"timer":("0/00:00:0.0","5/00:00:0.0"),"Ant":[ 0,0],"IFs":[1,1],"Chans":[242,245],   "Stokes":'1111',"Reason":"Bad Ch"},
    #{"timer":("0/00:00:0.0","5/00:00:0.0"),"Ant":[ 0,0],"IFs":[1,1],"Chans":[157,170],   "Stokes":'1111',"Reason":"RFI"},
    ]

# Average 
parms["CalAvgTime"] =    10./60.    # Time for averaging calibrated uv data (min)
parms["CABChan"] =       100        # First Channel to copy
parms["CAEChan"] =       375        # Highest Channel to copy
# No Hann
# No Hannparms["CABChan"] =       200        # First Channel to copy
# No Hannparms["CAEChan"] =       750        # Highest Channel to copy

# Control, mark items as F to disable
T   = True
F   = False
check                  = parms["check"]     # Only check script, don't execute tasks
debug                  = F #parms["debug"]     # run tasks debug
parms["doLoadArchive"] = False        # Load from archive?
parms["doHann"]        = T #parms["doHann"]     # Apply Hanning?
parms["doClearTab"]    = T        # Clear cal/edit tables
parms["doCopyFG"]      = T        # Copy FG 1 to FG 2
parms["doQuack"]       = T        # Quack data?
parms["doShad"]        = F    #parms["doShad"] # Flag shadowed data?
parms["doElev"]        = F
parms["doMedn"]        = T        # Median editing?
parms["doFD1"]         = T        # Do initial frequency domain flagging
parms["doRMSAvg"]      = T        # Do RMS/Mean editing for calibrators
parms["doPACor"]       = False    # Polarization angle correction?
parms["doDelayCal"]    = F        # Group Delay calibration?
parms["doBPCal"]       = T        # Determine Bandpass calibration
#debug                 = T
parms["doAmpPhaseCal"] = T        # Amplitude/phase calibration
parms["doAutoFlag"]    = T        # Autoflag editing after final calibration?
parms["doRecal"]       = T        # Redo calibration after editing
parms["doDelayCal2"]   = F        # Group Delay calibration of averaged data?, 2nd pass
parms["doBPCal2"]      = T        # Determine Bandpass calibration, 2nd pass
parms["doAmpPhaseCal2"]= T        # Amplitude/phase calibration, 2nd pass
parms["doAutoFlag2"]   = T        # Autoflag editing after final calibration?
parms["doCalAvg"]      = T        # calibrate and average data
parms["doRLDelay"]     = False #parms["doRLDelay"] # Determine R-L delay?
parms["doPolCal"]      = False #parms["doPolCal"]  # Do polarization calibration?
parms["doRLCal"]       = False #parms["doRLCal"]   # Determine  R-L bandpass?
parms["doImage"]       = T        # Image targets
parms["doSaveImg"]     = T        # Save results to FITS
parms["doSaveUV"]      = T        # Save calibrated UV data to FITS
parms["doSaveTab"]     = T        # Save UV tables to FITS
parms["doKntrPlots"]   = T        # Contour plots
parms["doDiagPlots"]   = T        # Source diagnostic plots
parms["doMetadata"]    = T        # Generate metadata dictionaries
parms["doHTML"]        = T        # Generate HTML report
parms["doVOTable"]     = T        # VOTable report
parms["doCleanup"]     = T    # Destroy AIPS files

# diagnostics
parms["doSNPlot"]      = T        # Plot SN tables etc
parms["doReport"]      = T        # Individual source report
parms["doRawSpecPlot"] = T #'3C286'!='None'  # Plot Raw spectrum
parms["doSpecPlot"]    = T #'3C286'!='None'  # Plot spectrum at various stages of processing
