import ast
import logging

from pretty import pretty
import six

from katsdpcontim.aips_parser import obit_config_from_aips
from katsdpcontim.configuration import get_config

import ObitTask

log = logging.getLogger('katsdpcontim')


def parse_python_assigns(assign_str):
    """
    Parses a string, containing assign statements
    into a dictionary.

    .. code-block:: python

        h5 = katdal.open('123456789.h5')
        kwargs = parse_python_assigns("spw=3; scans=[1,2];
                                      targets='bpcal,radec'")
        h5.select(**kwargs)

    Parameters
    ----------
    assign_str: str
        Assignment string. Should only contain
        assignment statements assigning
        python literal values to names,
        separated by semi-colons.

    Returns
    -------
    dict
        Dictionary { name: value } containing
        assignment results.
    """

    if not assign_str:
        return {}

    try:
        return {target.id: ast.literal_eval(stmt.value)
                for stmt in ast.parse(assign_str, mode='single').body
                for target in stmt.targets}
    except SyntaxError as e:
        log.exception("Exception parsing assignment string "
                      "'{}'".format(assign_str))
        raise e


def task_factory(name, aips_cfg_file=None, **kwargs):
    """
    Creates Obit tasks, applying any configuration options
    supplied in an AIPS configuration file and keyword arguments.

    The following creates an `MFImage` task, loading configuration
    options from `mfimage.in` and setting the name, class and
    sequence number of the input AIPS file on disk 1.

    .. code-block:: python

        mfimage = task_factory("MFImage", "mfimage.in",
                            DataType="AIPS", inName="DEEP2",
                            inClass="raw", inSeq=1, inDisk=1)
        mfimage.go()

    Invalid configuration options will generate warnings,
    but will not be applied to the task.

    Parameters
    ----------
    name: string
        Obit task name. "MFImage" for example.
    aips_cfg_file (optional): string
        AIPS config file
    **kwargs (optional): :obj:
        Any task options that should be applied
        to the returned task object.

    Returns
    -------
    :class:`ObitTask.ObitTask`
        Obit task object
    """
    # Load any supplied file configuration first,
    # then apply kwargs on top of this
    if aips_cfg_file is not None:
        file_kwargs = obit_config_from_aips(aips_cfg_file)
        file_kwargs.update(kwargs)
        kwargs = file_kwargs

    # Override user number and directories
    # from global configuration
    global_cfg = get_config()
    kwargs['user'] = global_cfg.aips.userno
    kwargs['AIPSdirs'] = [dir for url, dir in global_cfg.obit.aipsdirs]
    kwargs['FITSdirs'] = [dir for url, dir in global_cfg.obit.fitsdirs]

    # Obit ignores these options
    for k in ('nFITS', 'nAIPS', 'AIPSuser'):
        kwargs.pop(k, None)

    # Create the Obit Task
    task = ObitTask.ObitTask(name)

    # Apply configuration options to the task
    for k, v in six.iteritems(kwargs):
        try:
            setattr(task, k, v)
        except AttributeError as e:
            attr_err = "ObitTask instance has no attribute '{}'".format(k)
            if attr_err in e.message:
                log.warn("Key '{}' is not valid for this "
                         "task and will be ignored".format(k))
            else:
                raise

    return task

def fractional_bandwidth(uv_desc):
    """
    Returns the fractional bandwidth, given a uv descriptor dictionary

    Parameters
    ----------
    uv_desc : dict
        UV descriptor

    Returns
    -------
    float
        fractional bandwidth
    """
    try:
        ctypes = uv_desc['ctype']
    except KeyError:
        raise KeyError("The following UV descriptor is missing "
                        "a 'ctype': %s" % pretty(uv_desc))

    ctypes = [ct.strip() for ct in ctypes]

    try:
        freq_idx = ctypes.index('FREQ')
    except ValueError:
        raise ValueError("The following UV descriptor is missing "
                        "FREQ in it's 'ctype' field: %s" % pretty(uv_desc))

    freq_crval = uv_desc['crval'][freq_idx]
    freq_cdelt = uv_desc['cdelt'][freq_idx]
    freq_naxis = uv_desc['inaxes'][freq_idx]

    f1 = freq_crval
    f2 = freq_crval + freq_naxis*freq_cdelt

    return 2.0*(f2 - f1) / (f2 + f1)

def fmt_bytes(nbytes):
    """ Returns a human readable string, given the number of bytes """
    for x in ['B','KB','MB','GB']:
        if nbytes < 1024.0:
            return "%3.1f%s" % (nbytes, x)
        nbytes /= 1024.0

    return "%.1f%s" % (nbytes, 'TB')
