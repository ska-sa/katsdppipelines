import argparse
import logging
import os
from os.path import join as pjoin
import shutil

import katsdpcontim
from katsdpcontim.configuration import get_config

logging.getLogger('').handlers = [] # Remove handlers on the root logger

def create_logger():
    """ Create a logger """
    log = logging.getLogger("cfg_aips_disks")
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    log.addHandler(sh)
    return log

log = create_logger()

def create_parser():
    """ Create an argument parser """
    parser = argparse.ArgumentParser()
    parser.add_argument("aipsdir",
                help="Location of AIPS installation, "
                     "e.g. /usr/local/AIPS")
    return parser

def rewrite_dadevs(args, cfg):
    """
    Rewrite `args.aipsdir/DA00/DADEVS.LIST` to reference
    the AIPS directories specified in our configuration
    """
    dadevs_list = pjoin(args.aipsdir, 'DA00', 'DADEVS.LIST')
    backup = pjoin(args.aipsdir, 'DA00', '.DADEVS.LIST.BAK')

    if not os.path.exists(dadevs_list):
        raise ValueError("Could not find '{}' for modification. "
                         "Check your AIPS directory root '{}'."
                            .format(dadevs_list, args.aipsdir))

    # Make a copy of the original
    shutil.copy(dadevs_list, backup)

    # Read the copy, writing back to the original
    with open(backup, "r") as rf, open(dadevs_list, "w") as wf:
        # Retain comments
        for line in rf:
            if line.startswith('#'):
                wf.write(line)

        # Write out AIPS directories
        for aipsdir in cfg.obit.aipsdirs:
            log.info("Adding AIPS Disk '{}' to '{}'".format(aipsdir, dadevs_list))
            wf.write("-  " + aipsdir + '\n')

    # Remove the copy
    os.remove(backup)

def rewrite_netsp(args, cfg):
    """
    Rewrite `args.aipsdir/DA00/NETSP` to reference
    the AIPS directories specified in our configuration
    """
    netsp = pjoin(args.aipsdir, 'DA00', 'NETSP')
    backup = pjoin(args.aipsdir, 'DA00', '.NETSP.BAK')

    if not os.path.exists(netsp):
        raise ValueError("Could not find '{}' for modification. "
                         "Check your AIPS directory root '{}'."
                            .format(netsp, args.aipsdir))

    # Make a copy of the original
    shutil.copy(netsp, backup)

    # Read the copy, writing back to the original
    with open(backup, "r") as rf, open(netsp, "w") as wf:
        # Retain comments
        for line in rf:
            if line.startswith('#'):
                wf.write(line)

        # Write out AIPS Directory parameters
        for aipsdir in cfg.obit.aipsdirs:
            log.info("Adding AIPS Disk '{}' to '{}'".format(aipsdir, netsp))
            wf.write(aipsdir + ' 365.0    0    0    0    0    0    0    0    0\n')

    # Remove the copy
    os.remove(backup)

def setup_aips_disks(args, cfg):
    """
    Ensure that each AIPS disk (directory) exists.
    Creates a SPACE file within the disk.
    """

    for aipsdir in cfg.obit.aipsdirs:
        # Create directory if it doesn't exist
        if not os.path.exists(aipsdir):
            log.warn("AIPS Disk '{}' does not exist "
                     "and will be created".format(aipsdir))
            os.makedirs(aipsdir)

        # Create SPACE file
        space = pjoin(aipsdir, 'SPACE')

        with open(space, 'a'):
            os.utime(space, None)

if __name__ == "__main__":
    args = create_parser().parse_args()
    cfg = get_config()

    setup_aips_disks(args, cfg)
    rewrite_dadevs(args, cfg)
    rewrite_netsp(args, cfg)
