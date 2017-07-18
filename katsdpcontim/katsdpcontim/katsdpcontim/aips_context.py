import logging

from contextlib import contextmanager

from katim import AIPSSetup

import OErr
import OSystem

logging.basicConfig(level=logging.INFO)

@contextmanager
def aips_context():
    """ Context for creating and shutting down an AIPS context """
    try:
        logging.info("Creating AIPS Context")
        err = OErr.OErr()
        ObitSys = AIPSSetup.AIPSSetup(err)
        yield ObitSys, err
    finally:
        logging.info("Shutting Down AIPS Context")
        OErr.printErr(err)
        OSystem.Shutdown(ObitSys)