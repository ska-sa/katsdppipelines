import logging

from contextlib import contextmanager

import AIPS
import ObitTalkUtil

import OErr
import OSystem

logging.basicConfig(level=logging.INFO)

# Single obit context class
__obit_context = None

class ObitContext(object):
    """
    Small wrapper class encapsulating
    the Obit error stack and Obit System
    """
    def __init__(self):
        """
        Constructor

        Largely derived from
        https://github.com/bill-cotton/Obit/blob/master/ObitSystem/Obit/share/scripts/AIPSSetup.py
        """

        # TODO: Configuration object should be passed.
        # Generate default configuration for now
        from configuration import get_config
        cfg = get_config()

        self.err = err = OErr.OErr()
        self.obitsys =  OSystem.OSystem("Pipeline", 1, cfg.obit.userno,
                                            0, [" "], 0, [" "],
                                            True, False, err)
        OErr.printErrMsg(err, "Error starting Obit System")

        # Setup AIPS userno
        AIPS.userno = cfg.obit.userno

        # (url, dir) tuples, where None means localhost
        aipsdirs = [(None, d) for d in cfg.obit.aipsdirs]
        fitsdirs = [(None, d) for d in cfg.obit.fitsdirs]

        # Setup Obit Environment
        ObitTalkUtil.SetEnviron(AIPS_ROOT=cfg.obit.aipsroot,
                                AIPS_VERSION=cfg.obit.aipsversion,
                                OBIT_EXEC=cfg.obit.obitexec,
                                DA00=cfg.obit.da00,
                                ARCH="LINUX",
                                aipsdirs=aipsdirs,
                                fitsdirs=fitsdirs)

    def close(self):
        """
        Shutdown the Obit System, logging any errors on the error stack
        """
        if self.err.isErr:
            OErr.printErr(self.err)

        OSystem.Shutdown(self.obitsys)

@contextmanager
def obit_context():
    """
    Creates a global Obit Context, initialising the AIPS system
    and creating error stacks.

    .. code-block:: python

        with obit_context():
            err = obit_err()
            handle_obit_err("An error occured", err)
            ...
    """
    global __obit_context

    try:
        if __obit_context is not None:
            raise ValueError("Obit Context already exists")

        logging.info("Creating Obit Context")
        __obit_context = ObitContext()

        yield
    finally:
        if __obit_context is not None:
            logging.info("Shutting Down Obit Context")
            __obit_context.close()
            __obit_context = None

def handle_obit_err(msg="", err=None):
    """
    If the Obit error stack is in an error state,
    print it via :code:`OErr.printErrMsg` and raise
    an :code:`Exception(msg)`.

    Parameters
    ----------
    msg (optional): str
        Message describing the context in which the
        error occurred. Defaults to "".
    err (optional): OErr
        Obit error stack to handle. If None, the default
        error stack will be obtained from :code:`obit_err()`

    Raises
    ------
    Exception
        Raised by Obit if error stack is in an error state
    """
    if err is None:
        err = obit_err()

    # OErr.printErrMsg raises an Exception
    if err.isErr:
        err.printErrMsg(err, msg)

def obit_err():
    """ Return the Obit Context error stack """
    try:
        return __obit_context.err
    except AttributeError as e:
        if 'NoneType' in e.message:
            raise ValueError("Create a valid Obit context with obit_context()")

def obit_sys():
    """ Return the Obit Context system """
    try:
        return __obit_context.obitsys
    except AttributeError as e:
        if 'NoneType' in e.message:
            raise ValueError("Create a valid Obit context with obit_context()")