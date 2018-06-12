import ast
import functools
import logging

from pretty import pretty
import six

from katacomb.aips_parser import obit_config_from_aips
from katacomb.configuration import get_config

import ObitTask

log = logging.getLogger('katacomb')

def post_process_args(args, kat_adapter):
    """
    Perform post-processing on command line arguments.

    1. Capture Block ID set to katdal experiment ID if not present.

    Parameters
    ----------
    args : object
        Arguments created by :meth:`argparse.ArgumentParser.parse_args()`
    kat_adapter : :class:`katacomb.KatdalAdapter`
        Katdal Adapter

    Returns
    -------
    object
        Modified arguments
    """
    # Set capture block ID to experiment ID if not set
    if args.capture_block_id is None:
        file_cb_id = kat_adapter.obs_params.get('capture_block_id')
        if file_cb_id is None:
            args.capture_block_id = kat_adapter.experiment_id

            log.warn("No capture block ID was specified. "
                    "Using experiment_id '%s' instead.", kat_adapter.experiment_id)
        else:
            args.capture_block_id = file_cb_id

    return args

import __builtin__

# builtin function whitelist
_BUILTIN_WHITELIST = frozenset(['slice'])
_missing = _BUILTIN_WHITELIST.difference(dir(__builtin__))
if len(_missing) > 0:
    raise ValueError("'%s' are not valid builtin functions.'" % list(_missing))


def parse_python_assigns(assign_str):
    """
    Parses a string, containing assign statements
    into a dictionary.

    .. code-block:: python

        h5 = katdal.open('123456789.h5')
        kwargs = parse_python_assigns("spw=3; scans=[1,2];"
                                      "targets='bpcal,radec';"
                                      "channels=slice(0,2048)")
        h5.select(**kwargs)

    Parameters
    ----------
    assign_str: str
        Assignment string. Should only contain assignment statements
        assigning python literals or builtin function calls, to variable names.
        Multiple assignment statements should be separated by semi-colons.

    Returns
    -------
    dict
        Dictionary { name: value } containing
        assignment results.
    """


    if not assign_str:
        return {}

    def _eval_value(stmt_value):
        # If the statement value is a call to a builtin, try evaluate it
        if isinstance(stmt_value, ast.Call):
            func_name = stmt_value.func.id

            if func_name not in _BUILTIN_WHITELIST:
                raise ValueError("Function '%s' in '%s' is not builtin. "
                                 "Available builtins: '%s'"
                                    % (func_name, assign_str, list(_BUILTIN_WHITELIST)))

            # Recursively pass arguments through this same function
            if stmt_value.args is not None:
                args = tuple(_eval_value(a) for a in stmt_value.args)
            else:
                args = ()

            # Recursively pass keyword arguments through this same function
            if stmt_value.keywords is not None:
                kwargs = {kw.arg : _eval_value(kw.value) for kw
                                          in stmt_value.keywords}
            else:
                kwargs = {}

            return getattr(__builtin__, func_name)(*args, **kwargs)
        # Try a literal eval
        else:
            return ast.literal_eval(stmt_value)

    # Variable dictionary
    variables = {}

    # Parse the assignment string
    stmts = ast.parse(assign_str, mode='single').body

    for i, stmt in enumerate(stmts):
        if not isinstance(stmt, ast.Assign):
            raise ValueError("Statement %d in '%s' is not a "
                             "variable assignment." % (i, assign_str))

        # Evaluate assignment lhs
        values = _eval_value(stmt.value)

        # "a = b = c" => targets 'a' and 'b' with 'c' as result
        for target in stmt.targets:
            # a = 2
            if isinstance(target, ast.Name):
                variables[target.id] = values

            # Tuple/List unpacking case
            # (a, b) = 2
            elif isinstance(target, (ast.Tuple, ast.List)):
                # Require all tuple/list elements to be variable names,
                # although anything else is probably a syntax error
                if not all(isinstance(e, ast.Name) for e in target.elts):
                    raise ValueError("Tuple unpacking in assignment %d "
                                     "in expression '%s' failed as not all "
                                     "tuple contents are variable names.")

                # Promote for zip and length checking
                if not isinstance(values, (tuple, list)):
                    elements = (values,)
                else:
                    elements = values

                if not len(target.elts) == len(elements):
                    raise ValueError("Unpacking '%s' into a tuple/list in "
                                     "assignment %d of expression '%s' failed. "
                                     "The number of tuple elements did not match "
                                     "the number of values."
                                        % (values, i, assign_str))

                # Unpack
                for variable, value in zip(target.elts, elements):
                    variables[variable.id] = value
            else:
                raise TypeError("'%s' types are not supported"
                                "as assignment targets." % type(target))


    return variables

def log_exception(logger):
    """ Decorator that wraps the passed log object and logs exceptions """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                logger.exception("Exception in '%s'", func.__name__)
                raise

        return wrapper

    return decorator

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
                log.warn("Key '%s' is not valid for this "
                         "task and will be ignored", k)
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
