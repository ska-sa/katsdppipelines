#!/usr/bin/env python
import argparse
import glob
import logging
import os
from os.path import join as pjoin
import shutil

from katsdpservices import setup_logging
import katacomb.configuration as kc
from katacomb.util import setup_aips_disks

log = logging.getLogger('katacomb')

def rewrite_dadevs():
    """
    Rewrite ``cfg.aips.aipsroot/DA00/DADEVS.LIST`` to reference
    the AIPS directories specified in our configuration
    """
    cfg = kc.get_config()
    dadevs_list = pjoin(cfg['da00'], 'DADEVS.LIST')
    backup = pjoin(cfg['da00'], '.DADEVS.LIST.BAK')

    if not os.path.exists(dadevs_list):
        log.warn("Could not find '%s' for modification. "
                         "Check your AIPS directory root '%s'.",
                                dadevs_list, cfg['aipsroot'])

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
        for url, aipsdir in cfg['aipsdirs']:
            log.info("Adding AIPS Disk '%s' to '%s'", aipsdir, dadevs_list)
            wf.write("-  " + aipsdir + '\n')

    # Remove the copy
    os.remove(backup)

def rewrite_netsp():
    """
    Rewrite `cfg.aips.aipsroot.da00/DA00/NETSP` to reference
    the AIPS directories specified in our configuration
    """
    cfg = kc.get_config()
    netsp = pjoin(cfg['da00'], 'NETSP')
    backup = pjoin(cfg['da00'], '.NETSP.BAK')

    if not os.path.exists(netsp):
        log.warn("Could not find '%s' for modification. "
                 "Check your AIPS directory root '%s'.",
                            netsp, cfg['aipsroot'])
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
        for url, aipsdir in cfg['aipsdirs']:
            log.info("Adding AIPS Disk '%s to '%s'", aipsdir, netsp)
            wf.write(aipsdir + ' 365.0    0    0    0    0    0    0    0    0\n')

    # Remove the copy
    os.remove(backup)

def link_obit_data():
    """
    Creates soft links to Obit data files within FITS directories
    """

    cfg = kc.get_config()
    # Directory in which Obit data file are located
    obit_data_glob = pjoin(cfg['obitroot'], 'ObitSystem',
                           'Obit', 'share', 'data', '*')
    # Data files we wish to symlink
    data_files = glob.glob(obit_data_glob)

    # Separate filename from full path
    filenames = [os.path.split(f)[1] for f in data_files]

    # In each FITS dir, create a link to each data file
    for url, fitsdir in cfg['fitsdirs']:
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


def create_parser():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)

    parser.add_argument("-a", "--aipsdisks",
                        default=None,
                        type=lambda s: [(None, ds.strip()) for ds in s.split(',')],
                        help="Comma separated list of paths to aipsdisks.")

    parser.add_argument("-f", "--fitsdisks",
                        default=None,
                        type=lambda s: [(None, ds.strip()) for ds in s.split(',')],
                        help="Comma separated list of paths to fitsdisks.")

    return parser


setup_logging()

args = create_parser().parse_args()

kc.set_config(aipsdirs=args.aipsdisks, fitsdirs=args.fitsdisks)
setup_aips_disks()
rewrite_dadevs()
rewrite_netsp()
link_obit_data()
