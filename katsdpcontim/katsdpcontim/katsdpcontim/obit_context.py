import logging

from contextlib import contextmanager

from katim import AIPSSetup

import OErr
import OSystem

logging.basicConfig(level=logging.INFO)

__obit_context = None

class ObitContext(object):
    """
    Small wrapper class encapsulating
    the Obit error stack and Obit System
    """
    def __init__(self):
        self.err = OErr.OErr()
        self.obitsys = AIPSSetup.AIPSSetup(self.err)

    def close(self):
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

def handle_obit_err(message=None, err=None):
    """
    Performs Obit error log handling if any errors are present

    Parameters
    ----------
    message (optional): str
        Additional message to log, along with the error
    err (optional): OErr
        Obit error stack to handle. If None, the default
        error stack will be obtained from :code:`obit_err()`

    Raises
    ------
    Exception
        exceptions may be thrown by thrown by Obit on error.
    """
    if err is None:
        err = obit_err()

    if err.isErr:
        if msg is None:
            OErr.printErr(err)
        else:
            OErr.printMsg(err, message=msg)


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