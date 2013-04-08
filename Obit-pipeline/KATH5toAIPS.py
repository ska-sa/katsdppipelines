""" Utility to convert KAT-7 HDF format data to AIPS (or FITS)

This module requires katfile and katpoint and their dependencies
"""
# $Id: KATH5toAIPS.py 430 2012-11-02 02:00:09Z bill.cotton $
#-----------------------------------------------------------------------
#  Copyright (C) 2012
#  Associated Universities, Inc. Washington DC, USA.
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License as
#  published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public
#  License along with this program; if not, write to the Free
#  Software Foundation, Inc., 675 Massachusetts Ave, Cambridge,
#  MA 02139, USA.
#
#  Correspondence concerning this software should be addressed as follows:
#         Internet email: bcotton@nrao.edu.
#         Postal address: William Cotton
#                         National Radio Astronomy Observatory
#                         520 Edgemont Road
#                         Charlottesville, VA 22903-2475 USA
#-----------------------------------------------------------------------
try:
    import katfile
except Exception, exception:
    print exception
    print "KAT software not available"
    raise  RuntimeError, "KAT software unavailable"
else:
    pass
import time,math
import UV, UVVis, OErr, UVDesc, Table, History
from OTObit import day2dhms
import numpy      
from numpy import numarray

def KAT2AIPS (h5datafile, outUV, err, \
              calInt=1.0):
    """ 
    Convert KAT-7 HDF 5 data set to an Obit UV
    
    This module requires katfile and katpoint and their dependencies
     contact Ludwig Schwardt <schwardt@ska.ac.za> for details
    * h5datafile  = input KAT data file
    * outUV       = Obit UV object, shoud be a KAT template for the
                    appropriate number of IFs and poln.
    * err         = Obit error/message stack
    * calInt      = Calibration interval in min.
    """
    ################################################################
    # get interface
    OK = False
    try:
        katdata = katfile.open(h5datafile)
        OK = True
    except Exception, exception:
        print exception
    else:
        pass
    if not OK:
        OErr.PSet(err)
        OErr.PLog(err, OErr.Fatal, "Unable to read KAT HDF5 data in "+h5datafile)
    if err.isErr:
        OErr.printErrMsg(err, "Error with h5 file")
    # Extract metadata
    meta = GetKATMeta(katdata, err)
    # Update descriptor
    UpdateDescriptor (outUV, meta, err)
    # Write AN table
    WriteANTable (outUV, meta, err)
    # Write FQ table
    WriteFQTable (outUV, meta, err)
    # Write SU table
    WriteSUTable (outUV, meta, err)

    # Convert data
    ConvertKATData(outUV, katdata, meta, err)

    # Index data
    OErr.PLog(err, OErr.Info, "Indexing data")
    OErr.printErr(err)
    UV.PUtilIndex (outUV, err)

    # Open/close UV to update header
    outUV.Open(UV.READONLY,err)
    outUV.Close(err)
    if err.isErr:
        OErr.printErrMsg(err, message="Update UV header failed")

    # initial CL table
    OErr.PLog(err, OErr.Info, "Create Initial CL table")
    OErr.printErr(err)
    UV.PTableCLGetDummy(outUV, outUV, 1, err, solInt=calInt)
    outUV.Open(UV.READONLY,err)
    outUV.Close(err)
    if err.isErr:
        OErr.printErrMsg(err, message="Update UV header failed")

    # History
    outHistory = History.History("outhistory", outUV.List, err)
    outHistory.Open(History.READWRITE, err)
    outHistory.TimeStamp("Convert KAT7 HDF 5 data to Obit", err)
    outHistory.WriteRec(-1,"datafile = "+h5datafile, err)
    outHistory.WriteRec(-1,"calInt   = "+str(calInt), err)
    outHistory.Close(err)
    outUV.Open(UV.READONLY,err)
    outUV.Close(err)
    if err.isErr:
        OErr.printErrMsg(err, message="Update UV header failed")
    # Return the metadata for the pipeline
    return meta
   # end KAT2AIPS

def GetKATMeta(katdata, err):
    """
    Get KAT metadata and return as a dictionary

     * katdata  = input KAT dataset
     * err      = Python Obit Error/message stack to init
     returns dictionary:
     "spw"     Spectral window array as tuple (nchan, freq0, chinc)
               nchan=no. channels, freq0 = freq of channel 0,
               chinc=signed channel increment, one tuple per SPW
     "targets" Array of target tuples:
               (index, name, ra2000, dec2000, raapp, decapp)
     "bpcal"   List of source indices of Bandpass calibrators
     "gaincal" List of source indices of Gain calibrators
     "source" List of source indices of imaging targets
     "targLookup" dict indexed by source number with source index
     "tinteg"   Integration time in seconds
     "obsdate"  First day as YYYY-MM-DD
     "observer" name of observer
     "ants"     Array of antenna tuples (index, name, X, Y, Z, diameter)
     "nstokes"  Number of stokes parameters
     "products" Tuple per data product (ant1, ant2, offset)
                where offset is the index on the Stokes axis (XX=0...)
    """
    ################################################################
    # Checks
    out = {}
    # Spectral windows
    sw = []
    for s in katdata.spectral_windows:
        sw.append((s.num_chans, s.channel_freqs[0], s.channel_freqs[1]-s.channel_freqs[0]))
    out["spw"] = sw
    # targets
    tl = []
    tb = []
    tg = []
    tt = []
    td = {}
    i = 0
    for t in  katdata.catalogue.targets:
        name = (t.name.replace(' ','_')+"                ")[0:16]
        ras, decs = t.radec()
        dec = UVDesc.PDMS2Dec(str(decs).replace(':',' '))
        ra  = UVDesc.PHMS2RA(str(ras).replace(':',' '))
        # Apparent posn
        ras, decs = t.apparent_radec()
        deca = UVDesc.PDMS2Dec(str(decs).replace(':',' '))
        raa  = UVDesc.PHMS2RA(str(ras).replace(':',' '))
        i += 1
        tl.append((i, name, ra, dec, raa, deca))
        if len(t.tags)>1:
            if t.tags[1]=='bpcal':
                tb.append(t)
            if t.tags[1]=='gaincal':
                tg.append(t)
            if t.tags[1]=='target':
                tt.append(t)
        td[name.rstrip()] = i
    out["targets"] = tl
    out["targLookup"] = td
    out["bpcal"] = tb
    out["gaincal"] = tg
    out["source"] = tt
    # Antennas
    al = []
    i = 0
    for a in  katdata.ants:
        name  = a.name
        x,y,z = a.position_ecef
        diam  = a.diameter
        i += 1
        al.append((i, name, x, y, z, diam))
    out["ants"] = al
    # Data products
    dl = []
    nstokes = 1
    for d in  katdata.corr_products:
        a1 = int(d[0][3:4])
        a2 = int(d[1][3:4])
        if d[0][4:]=='h' and d[1][4:]=='h':
            dp = 0
        elif d[0][4:]=='v' and d[1][4:]=='v':
            dp = 1
        elif d[0][4:]=='h' and d[1][4:]=='v':
            dp = 2
        else:
            dp = 3
        dl.append((a1, a2, dp))
        nstokes = max (nstokes,dp+1)
    out["products"] = dl
    out["nstokes"]  = nstokes
    # integration time
    out["tinteg"] = katdata.dump_period
    # observing date
    start=time.gmtime(katdata.timestamps[0])
    out["obsdate"] = time.strftime('%Y-%m-%d', start)
    # Observer's name
    out["observer"] = katdata.observer
    # Number of channels
    numchan = len(katdata.channels)
    out["numchan"] = numchan
    # Correlator mode (assuming 1 spectral window KAT-7)
    out["corrmode"] = katdata.spectral_windows[0].mode
    # Central frequency (in Hz)
    out["centerfreq"] = katdata.channel_freqs[numchan //2]
    # Expose all KAT-METADATA to calling script
    out["katdata"] = katdata
    return out
    # end GetKATMeta

def UpdateDescriptor (outUV, meta, err):
    """
    Update information in data descriptor

    NB: Cannot change geometry of visibilities
    * outUV    = Obit UV object
    * meta     = dict with data meta data
    * err      = Python Obit Error/message stack to init
    """
    ################################################################
    chinc   =  meta["spw"][0][2]   # Frequency increment
    reffreq =  meta["spw"][0][1]   # reference frequency
    nchan   =  meta["spw"][0][0]   # number of channels
    nif     = len(meta["spw"])     # Number of IFs
    nstok   = meta["nstokes"]      # Number of Stokes products
    desc = outUV.Desc.Dict
    outUV.Desc.Dict = desc
    desc['obsdat']   = meta["obsdate"]
    desc['observer'] = meta["observer"]
    desc['JDObs']    = UVDesc.PDate2JD(meta["obsdate"])
    desc['naxis']    = 6
    desc['inaxes']   = [3,nstok,nchan,nif,1,1,0]
    desc['cdelt']    = [1.0,-1.0,chinc, 1.0, 0.0, 0.0, 0.0]
    desc['crval']    = [1.0, -5.0,reffreq, 1.0, 0.0, 0.0, 0.0]
    desc['crota']    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    outUV.Desc.Dict = desc
    outUV.UpdateDesc(err)
    if err.isErr:
        OErr.printErrMsg(err, "Error updating UV descriptor")
    # end UpdateDescriptor

def WriteANTable(outUV, meta, err):
    """
    Write data in meta to AN table

     * outUV    = Obit UV object
     * meta     = dict with data meta data
     * err      = Python Obit Error/message stack to init
    """
    ################################################################
    antab = outUV.NewTable(Table.READWRITE, "AIPS AN",1,err)
    if err.isErr:
        OErr.printErrMsg(err, "Error with AN table")
    antab.Open(Table.READWRITE, err)
    if err.isErr:
        OErr.printErrMsg(err, "Error opening AN table")
    # Update header
    antab.keys['RefDate'] = meta["obsdate"]
    antab.keys['Freq']    = meta["spw"][0][1]
    JD                    = UVDesc.PDate2JD(meta["obsdate"])
    antab.keys['GSTIA0']  = UVDesc.GST0(JD)*15.0
    antab.keys['DEGPDY']  = UVDesc.ERate(JD)*360.0
    Table.PDirty(antab),
    # Force update
    row = antab.ReadRow(1,err)
    if err.isErr:
        OErr.printErrMsg(err, "Error reading AN table")
    OErr.printErr(err)
    irow = 0
    for ant in meta["ants"]:
        irow += 1
        row['NOSTA']    = [ant[0]]
        row['ANNAME']   = [ant[1]+"    "]
        row['STABXYZ']  = [ant[2],ant[3],ant[4]]
        row['DIAMETER'] = [ant[5]]
        row['POLAA']    = [90.0]
        antab.WriteRow(irow, row,err)
        if err.isErr:
            OErr.printErrMsg(err, "Error writing AN table")
    antab.Close(err)
    if err.isErr:
        OErr.printErrMsg(err, "Error closing AN table")
    # end WriteANTable

def WriteFQTable(outUV, meta, err):
    """
    Write data in meta to FQ table
    An old FQ table is deleted

     * outUV    = Obit UV object
     * meta     = dict with data meta data
     * err      = Python Obit Error/message stack to init
    """
    ################################################################
    # If an old table exists, delete it
    if outUV.GetHighVer("AIPS FQ")>0:
        zz = outUV.ZapTable("AIPS FQ", 1, err)
        if err.isErr:
            OErr.printErrMsg(err, "Error zapping old FQ table")
    reffreq =  meta["spw"][0][1]   # reference frequency
    noif    = len(meta["spw"])     # Number of IFs
    fqtab = outUV.NewTable(Table.READWRITE, "AIPS FQ",1,err,numIF=noif)
    if err.isErr:
        OErr.printErrMsg(err, "Error with FQ table")
    fqtab.Open(Table.READWRITE, err)
    if err.isErr:
        OErr.printErrMsg(err, "Error opening FQ table")
    # Update header
    fqtab.keys['NO_IF'] = len(meta["spw"])  # Structural so no effect
    Table.PDirty(fqtab)  # Force update
    # Create row
    row = {'FRQSEL': [1], 'CH WIDTH': [0.0], 'TOTAL BANDWIDTH': [0.0], \
           'RXCODE': ['L'], 'SIDEBAND': [-1], 'NumFields': 7, 'Table name': 'AIPS FQ', \
           '_status': [0], 'IF FREQ': [0.0]}
    if err.isErr:
        OErr.printErrMsg(err, "Error reading FQ table")
    OErr.printErr(err)
    irow = 0
    for sw in meta["spw"]:
        irow += 1
        row['FRQSEL']    = [irow]
        row['IF FREQ']   = [sw[1] - reffreq]
        row['CH WIDTH']  = [sw[2]]
        row['TOTAL BANDWIDTH']  = [abs(sw[2])*sw[0]]
        row['RXCODE']  = ['L']
        if sw[2]>0.0:
            row['SIDEBAND']  = [1]
        else:
            row['SIDEBAND']  = [-1]
        fqtab.WriteRow(irow, row,err)
        if err.isErr:
            OErr.printErrMsg(err, "Error writing FQ table")
    fqtab.Close(err)
    if err.isErr:
        OErr.printErrMsg(err, "Error closing FQ table")
    # end WriteFQTable

def WriteSUTable(outUV, meta, err):
    """
    Write data in meta to SU table

     * outUV    = Obit UV object
     * meta     = dict with data meta data
     * err      = Python Obit Error/message stack to init
    """
    ################################################################
    sutab = outUV.NewTable(Table.READWRITE, "AIPS SU",1,err)
    if err.isErr:
        OErr.printErrMsg(err, "Error with SU table")
    sutab.Open(Table.READWRITE, err)
    if err.isErr:
        OErr.printErrMsg(err, "Error opening SU table")
    # Update header
    sutab.keys['RefDate'] = meta["obsdate"]
    sutab.keys['Freq']    = meta["spw"][0][1]
    Table.PDirty(sutab)  # Force update
    row = sutab.ReadRow(1,err)
    if err.isErr:
        OErr.printErrMsg(err, "Error reading SU table")
    OErr.printErr(err)
    irow = 0
    for tar in meta["targets"]:
        irow += 1
        row['ID. NO.']   = [tar[0]]
        row['SOURCE']    = [tar[1]]
        row['RAEPO']     = [tar[2]]
        row['DECEPO']    = [tar[3]]
        row['RAOBS']     = [tar[2]]
        row['DECOBS']    = [tar[3]]
        row['EPOCH']     = [2000.0]
        row['RAAPP']     = [tar[4]]
        row['DECAPP']    = [tar[5]]
        row['BANDWIDTH'] = [meta["spw"][0][2]]
        sutab.WriteRow(irow, row,err)
        if err.isErr:
            OErr.printErrMsg(err, "Error writing SU table")
    sutab.Close(err)
    if err.isErr:
        OErr.printErrMsg(err, "Error closing SU table")
    # end WriteSUtable

def ConvertKATData(outUV, katdata, meta, err):
    """
    Read KAT HDF data and write Obit UV

     * outUV    = Obit UV object
     * katdata  = input KAT dataset
     * meta     = dict with data meta data
     * err      = Python Obit Error/message stack to init
    """
    ################################################################
    reffreq =  meta["spw"][0][1]    # reference frequency
    lamb    = 2.997924562e8/reffreq # wavelength of reference freq
    nchan   =  meta["spw"][0][0]    # number of channels
    nif     = len(meta["spw"])      # Number of IFs
    nstok   = meta["nstokes"]       # Number of Stokes products
    p       = meta["products"]      # baseline stokes indices
    nprod   = len(p)                # number of correlations/baselines
    # work out Start time in unix sec
    tm = katdata.timestamps[1:2]
    tx = time.gmtime(tm[0])
    time0   = tm[0] - tx[3]*3600.0 - tx[4]*60.0 - tx[5]
    
    # Set data to read one vis per IO
    outUV.List.set("nVisPIO", 1)
    
    # Open data
    zz = outUV.Open(UV.READWRITE, err)
    if err.isErr:
        OErr.printErrMsg(err, "Error opening output UV")
    # visibility record offsets
    d = outUV.Desc.Dict
    ilocu   = d['ilocu']
    ilocv   = d['ilocv']
    ilocw   = d['ilocw']
    iloct   = d['iloct']
    ilocb   = d['ilocb']
    ilocsu  = d['ilocsu']
    nrparm  = d['nrparm']
    jlocc   = d['jlocc']
    jlocs   = d['jlocs']
    jlocf   = d['jlocf']
    jlocif  = d['jlocif']
    naxes   = d['inaxes']
    count = 0.0
    visno = 0
  
    # Get IO buffers as numpy arrays
    shape = len(outUV.VisBuf) / 4
    buffer =  numarray.array(sequence=outUV.VisBuf,
                             type=numarray.Float32, shape=shape)

    # Template vis
    vis = outUV.ReadVis(err, firstVis=1)
    first = True
    firstVis = 1
    for scan, state, target in katdata.scans():
        name=target.name.replace(' ','_')
        if state!="track":
            continue                    # Only on source data
        # Fetch data
        tm = katdata.timestamps[:]
        nint = len(tm)
        el=target.azel(tm[int(nint/2)])[1]*180./math.pi
        if el<15.:   # Throw away scans at low elevation
            msg = "Rejecting Scan on %s Start %s: Elevation %4.1f deg."%(name,day2dhms((tm[0]-time0)/86400.0)[0:12],el)
            OErr.PLog(err, OErr.Info, msg)
            OErr.printErr(err)
            continue
        vs = katdata.vis[:]
        wt = katdata.weights()[:]
        uu = katdata.u
        vv = katdata.v
        ww = katdata.w
        suid = meta["targLookup"][name]
        # Number of integrations
        msg = "Scan %4d Int %16s Start %s"%(nint,name,day2dhms((tm[0]-time0)/86400.0)[0:12])
        OErr.PLog(err, OErr.Info, msg);
        OErr.printErr(err)
        
        # Loop over integrations
        for iint in range(0,nint):
            # loop over data products/baselines
            for iprod in range(0,nprod):
                # Copy slices
                indx = nrparm+(p[iprod][2])*3
                buffer[indx:indx+(nchan+1)*nstok*3:nstok*3] = vs[iint:iint+1,:,iprod:iprod+1].real.flatten()
                indx += 1
                buffer[indx:indx+(nchan+1)*nstok*3:nstok*3] = vs[iint:iint+1,:,iprod:iprod+1].imag.flatten()
                indx += 1
                buffer[indx:indx+(nchan+1)*nstok*3:nstok*3] = wt[iint:iint+1,:,iprod:iprod+1].flatten()
                # Write if Stokes index >= next or the last
                if (iprod==nprod-1) or (p[iprod][2]>=p[iprod+1][2]):
                    # Random parameters
                    buffer[ilocu]  = uu[iint][iprod]/lamb
                    buffer[ilocv]  = vv[iint][iprod]/lamb
                    buffer[ilocw]  = ww[iint][iprod]/lamb
                    buffer[iloct]  = (tm[iint]-time0)/86400.0 # Time in days
                    buffer[ilocb]  = p[iprod][0]*256.0 + p[iprod][1]
                    buffer[ilocsu] = suid
                    outUV.Write(err, firstVis=visno)
                    visno += 1
                    buffer[3]= -3.14159
                    #print visno,buffer[0:5]
                    firstVis = None  # Only once
                    # initialize visibility
                    first = True
            # end loop over integrations
            if err.isErr:
                OErr.printErrMsg(err, "Error writing data")
    # end loop over scan
    outUV.Close(err)
    if err.isErr:
        OErr.printErrMsg(err, "Error closing data")
    # end ConvertKATData

