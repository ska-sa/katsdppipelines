# Define AIPS and FITS disks

import os,shutil
import AIPSLite

# Define OBIT_EXEC for access to Obit Software (None means it's in path)
OBIT_EXEC    = "/home/tmauch/ObitInstall/ObitSystem/Obit"

############################# Initialize AIPS ##########################################
# Is aips installed and LOGIN.CSH defined??
user             = 10
cwd              = os.getcwd()
aipsversion      = '31DEC13'

# Get the local AIPS version for the pipeline.
path = cwd
AIPSLite.get_aips(cwd,version=aipsversion)

# Make aips Disk in /tmp
DA00         = cwd+'/da00'
AIPS_DISK    = cwd+'/aipsdisk'

if os.path.exists(DA00):
    shutil.rmtree(DA00)
if os.path.exists(AIPS_DISK):
    shutil.rmtree(AIPS_DISK)

# Create the aips disk and the AIPS environment for the disks.
AIPSLite.make_disk(disk_path=AIPS_DISK)
AIPSLite.filaip(force=True,data_dir=AIPS_DISK)
AIPSLite.make_da00(da00_path=DA00)

# Get the set up AIPS environment.
AIPS_ROOT    = os.environ['AIPS_ROOT']
AIPS_VERSION = os.environ['AIPS_VERSION']

adirs = [(None, AIPS_DISK)]
fdirs = [(None, path+'/FITS')]

############################# Initialize OBIT ##########################################
err     = OErr.OErr()
ffdirs = []
for f in fdirs:
    ffdirs.append(f[1])
aadirs = []
for a in adirs:
    aadirs.append(a[1])

ObitSys = OSystem.OSystem ("Pipeline", 1, user, len(aadirs), aadirs, \
                         len(ffdirs), ffdirs, True, False, err)
OErr.printErrMsg(err, "Error with Obit startup")

# Setup AIPS, FITS
AIPS.userno = user

# setup environment
ObitTalkUtil.SetEnviron(AIPS_ROOT=AIPS_ROOT, AIPS_VERSION=AIPS_VERSION, \
                        OBIT_EXEC=OBIT_EXEC, DA00=DA00, ARCH=AIPSLite.arch(), \
                        aipsdirs=adirs, fitsdirs=fdirs)

# List directories
ObitTalkUtil.ListAIPSDirs()
ObitTalkUtil.ListFITSDirs()

# Disks to avoid
noScrat     = [0]          # AIPS disks to avoid 
nThreads    = 4            # Number of threads allowed
disk        = 1            # AIPS disk number
