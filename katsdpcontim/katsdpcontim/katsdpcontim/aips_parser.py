from collections import OrderedDict
import logging

import six

import InfoList
import ParserUtil

from katsdpcontim import obit_err, handle_obit_err

log = logging.getLogger('katsdpcontim')

def parse_aips_config(aips_cfg_file):
    """
    Parses an AIPS config file into a
    dictionary with schema
    :code:`{ option: [type, dimensions, value]}`

    :code:`type_` is an enum. Look at ObitTypes.h
    to figure it out.

    :code:`dims` indicate dimensionality of input
    For scalar types :code:`[64,1,1,1,1]` indicates
    a 1D array of length 64 floats.

    String dims need to be handled slightly differently
    First dimension indicates string length so for e.g.
    :code:`["obit", "    ", "abcd"]` has dims :code:`[4,3,1,1,1]`
    So a :code:`[4,1,1,1,1]` implies one string of length 4

    :code:`value` will always be a list and is probably
    nested if :code:`dims is setup appropriately.

    Parameters
    ----------
    aips_cfg_file : str
        AIPS configuration file

    Returns
    -------
    dict
        A dictionary of AIPS configuration options
    """
    err = obit_err()
    info_list = InfoList.InfoList()
    ParserUtil.PParse(aips_cfg_file, info_list, err)
    handle_obit_err("Error parsing Obit configuration file '{}'"
                        .format(aips_cfg_file), err)


    return InfoList.PGetDict(info_list)

def process_aips_config(aips_cfg_file):
    """
    Extract key-values from AIPS configuration file
    into a { option: value } dictionary.

    Processes the configuration so that the values are
    suitable to apply to ObitTask objects, converting
    singleton list to objects, for example.

    Parameters
    ----------
    aips_cfg_file : str
        AIPS configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """

    def _massage(option):
        """
        Massage values into structures suitable for
        setting on Obit Task objects
        """

        # Split into type, dimensions and value
        type_, dims, value = option

        # Booleans integers must be cast to python bools
        if type_ == 15:
            value = [bool(v) for v in value]

        # Check second dimension to test singletons
        # if we're handling strings else the first dim
        check_dim = 1 if type_ == 14 else 0

        # Return element zero from singleton lists
        if dims[check_dim] == 1 and len(value) == 1:
            return value[0]

        return value

    return { k: _massage(o) for k, o
            in parse_aips_config(aips_cfg_file).iteritems()}



from AIPS import AIPS
from FITS import FITS

def aips_disk_config(infile, fitsdir, aipsdir):
    """
    Returns an AIPS disk configuration suitable for
    use with Obit
    """

    import os

    if infile is None:
        infile = ""
        log.warn("No 'infile' was provided, setting to '%s'" % infile)

    if fitsdir is None:
        fitsdir = FITS.disks[aipsdisk].dirname
        log.warn("No 'fitsdir' was provided, setting to '%s'" % fitsdir)

    if aipsdir is None:
        aipsdir = AIPS.disks[fitsdisk].dirname
        log.warn("No 'aipsdir' was provided, setting to '%s'" % aipsdir)

    # Override AIPS disk configuration options
    cfg = OrderedDict([
        ("DataType", "FITS"),
        ("FITSdirs", [fitsdir]),
        ("AIPSdirs", [aipsdir]),
        ("inFile", infile),
        ("inDisk", 0),
        ("inSeq", 0),
        ("outFile", ".out.fits"),
        ("out2File", ".out.fits")])

    # Set output disk options to input disk options
    cfg.update([
        ("outDType", cfg["DataType"]),
        ("outDisk", cfg["inDisk"]),
        ("outSeq", cfg["inSeq"]),
        ("out2Disk", cfg["inDisk"]),
        ("out2Seq", cfg["inSeq"]),
    ])

    return cfg

def aips_user():
    """ Get the AIPS user """
    import OSystem

    try:
        return OSystem.PGetAIPSuser()
    except Exception as e:
        log.exception("Exception getting AIPS User. "
                          "Returning 105 instead")
        return 105

def aips_cfg(aips_cfg_file, infile=None, fitsdir=None, aipsdir=None):
    """ Construct a usable AIPS configuration """

    # Parse the configuration file
    cfg = process_aips_config(aips_cfg_file)

    # Set the user file
    cfg['userno'] = aips_user()

    # Set up some reasonable AIPS disk config
    cfg.update(aips_disk_config(infile, fitsdir, aipsdir))

    # These don't work with OBIT for some reason
    for k in ('nFITS','nAIPS', 'AIPSuser'):
        cfg.pop(k, None)

    return cfg

def apply_cfg_to_task(task, cfg):
    """
    Applies supplied configuration to task
    by setting attributes on the task object.

    Will warn if the attribute does not exist,
    but will continue
    """
    for k, v in six.iteritems(cfg):
        try:
            setattr(task, k, v)
        except AttributeError as e:
            attr_err = "ObitTask instance has no attribute '{}'".format(k)
            if attr_err in e.message:
                log.warn("Key '{}' is not valid for this "
                             "task and will be ignored".format(k))
            else:
                raise
