import ast
import logging

from katsdpcontim.configuration import get_config

log = logging.getLogger('katsdpcontim')

def parse_katdal_select(select_str):
    """
    Parses a string, containing assign statements
    that will turned be into kwargs suitable for
    passing to :meth:`katdal.DataSet.select`.

    .. code-block:: python

        h5 = katdal.open('123456789.h5')
        kwargs = parse_katdal_select("spw=3; scans=[1,2];
                                      targets='bpcal,radec'")
        h5.select(**kwargs)

    Parameters
    ----------
    select_str: str
        Selection string. Should only contain
        assignment statements assigning
        python literal values to names,
        separated by semi-colons.

    Returns
    -------
    dict
        Dictionary { name: value } containing
        assignment results.
    """

    if not select_str:
        return {}

    try:
        return { target.id: ast.literal_eval(stmt.value)
                for stmt in ast.parse(select_str, mode='single').body
                for target in stmt.targets}
    except SyntaxError as e:
        log.exception("Exception parsing katdal selection string "
                    "'{}'".format(select_str))
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

    import ObitTask
    from katsdpcontim.aips_parser import (obit_config_from_aips,
                                        apply_cfg_to_task)

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
    apply_cfg_to_task(task, kwargs)

    return task
