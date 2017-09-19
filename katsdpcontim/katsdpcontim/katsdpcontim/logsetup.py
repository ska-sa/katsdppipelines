import logging
import logging.handlers


def get_logger():
    # Console formatter, mention name
    cfmt = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(cfmt)

    logger = logging.getLogger('katsdpcontim')
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.propagate = False

    return logger
