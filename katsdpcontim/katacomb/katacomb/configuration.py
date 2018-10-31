import logging
import os
import threading
from os.path import join as pjoin

log = logging.getLogger('katacomb')


def config_validator():
    """
    Returns
    -------
    :py:class:`cerberus.Validator`
        Cerberus validator, suitable for validating the configuration
    """

    from cerberus import Validator

    # Create the schema and defaults
    obitroot = pjoin(os.sep, 'home', 'kat', 'Obit')
    obitexec = pjoin(obitroot, 'ObitSystem', 'Obit')
    # (url, dir) tuples, where None means localhost
    aipsroot = pjoin(os.sep, 'home', 'kat', 'AIPS')
    fitsdirs = [(None, pjoin(aipsroot, 'FITS'))]
    aipsdirs = [(None, pjoin(aipsroot, 'DATA', 'LOCALHOST_1')),
                (None, pjoin(aipsroot, 'DATA', 'LOCALHOST_2'))]

    schema = {
        'obitroot': {'type': 'string', 'default': obitroot},
        'obitexec': {'type': 'string', 'default': obitexec},
        'fitsdirs': {'type': 'list', 'default': fitsdirs},
        'aipsdirs': {'type': 'list', 'default': aipsdirs},
        'aipsroot': {'type': 'string', 'default': aipsroot},
        'aipsversion': {'type': 'string', 'default': '31DEC16'},
        'da00': {'type': 'string', 'default': pjoin(aipsroot, 'DA00')},
        'userno': {'type': 'integer', 'default': 105},
    }

    return Validator(schema)


__cfg_lock = threading.Lock()
__default_cfg = config_validator().validated({})
__active_cfg = {}


def set_config(cfg={}, **kwargs):
    # Set active configuration.
    from katacomb.util import recursive_merge

    global __active_cfg

    with __cfg_lock:
        __active_cfg = __default_cfg.copy()
        recursive_merge(cfg, __active_cfg)
        recursive_merge(kwargs, __active_cfg)
        __active_cfg = config_validator().validated(__active_cfg)


def get_config():
    # Get active configuration
    with __cfg_lock:
        return __active_cfg


def reset_config():
    # Reset active configuration to defaults
    set_config(cfg={})
