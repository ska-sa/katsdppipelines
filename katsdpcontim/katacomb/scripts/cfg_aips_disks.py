#!/usr/bin/env python
import glob
import logging
import os
from os.path import join as pjoin
import shutil

import katacomb
from katacomb.configuration import get_config

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

def rewrite_dadevs(cfg):
    """
    Rewrite ``cfg.aips.aipsroot/DA00/DADEVS.LIST`` to reference
    the AIPS directories specified in our configuration
    """
    dadevs_list = pjoin(cfg.aips.da00, 'DADEVS.LIST')
    backup = pjoin(cfg.aips.da00, '.DADEVS.LIST.BAK')

    if not os.path.exists(dadevs_list):
        log.warn("Could not find '%s' for modification. "
                         "Check your AIPS directory root '%s'.",
                                dadevs_list, cfg.aips.aipsroot)

        return

    # Make a copy of the original
    shutil.copy(dadevs_list, backup)

    # Read the copy, writing back to the original
    with open(backup, "r") as rf, open(dadevs_list, "w") as wf:
        # Retain comments
        for line in rf:
            if line.startswith('#'):
                wf.write(line)

        # Write out AIPS directories
        for url, aipsdir in cfg.obit.aipsdirs:
            log.info("Adding AIPS Disk '%s' to '%s'", aipsdir, dadevs_list)
            wf.write("-  " + aipsdir + '\n')

    # Remove the copy
    os.remove(backup)

def rewrite_netsp(cfg):
    """
    Rewrite `cfg.aips.aipsroot.da00/DA00/NETSP` to reference
    the AIPS directories specified in our configuration
    """
    netsp = pjoin(cfg.aips.da00, 'NETSP')
    backup = pjoin(cfg.aips.da00, '.NETSP.BAK')

    if not os.path.exists(netsp):
        log.warn("Could not find '%s' for modification. "
                 "Check your AIPS directory root '%s'.",
                            netsp, cfg.aips.aipsroot)
        return

    # Make a copy of the original
    shutil.copy(netsp, backup)

    # Read the copy, writing back to the original
    with open(backup, "r") as rf, open(netsp, "w") as wf:
        # Retain comments
        for line in rf:
            if line.startswith('#'):
                wf.write(line)

        # Write out AIPS Directory parameters
        for url, aipsdir in cfg.obit.aipsdirs:
            log.info("Adding AIPS Disk '%s to '%s'", aipsdir, netsp)
            wf.write(aipsdir + ' 365.0    0    0    0    0    0    0    0    0\n')

    # Remove the copy
    os.remove(backup)

def setup_aips_disks(cfg):
    """
    Ensure that each AIPS disk (directory) exists.
    Creates a SPACE file within the disk.
    """

    for url, aipsdir in cfg.obit.aipsdirs + cfg.obit.fitsdirs:
        # Create directory if it doesn't exist
        if not os.path.exists(aipsdir):
            log.info("Creating AIPS Disk '%s'", aipsdir)
            os.makedirs(aipsdir)

        # Create SPACE file
        space = pjoin(aipsdir, 'SPACE')

        with open(space, 'a'):
            os.utime(space, None)

def link_obit_data(cfg):
    """
    Creates soft links to Obit data files within FITS directories
    """

    # Directory in which Obit data file are located
    obit_data_glob = pjoin(cfg.obit.obitroot, 'ObitSystem', 'Obit',
                                                'share', 'data', '*')
    # Data files we wish to symlink
    data_files = glob.glob(obit_data_glob)

    # Separate filename from full path
    filenames = [os.path.split(f)[1] for f in data_files]

    # In each FITS dir, create a link to each data file
    for url, fitsdir in cfg.obit.fitsdirs:
        # Fully expand link paths
        link_names = [pjoin(fitsdir, f) for f in filenames]

        # Remove any prior symlinks and then symlink
        for data_file, link_name in zip(data_files, link_names):
            if os.path.exists(link_name):
                os.remove(link_name)

            try:
                os.symlink(data_file, link_name)
            except OSError as e:
                log.warn("Unable to link '{}' to '{}'\n"
                         "{}".format(link_name, data_file, e))

if __name__ == "__main__":
    cfg = get_config()

    setup_aips_disks(cfg)
    rewrite_dadevs(cfg)
    rewrite_netsp(cfg)
    link_obit_data(cfg)
