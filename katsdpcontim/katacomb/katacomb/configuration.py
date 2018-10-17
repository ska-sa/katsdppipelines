import collections

import os
from os.path import join as pjoin


def config_validator():
    """
    Returns
    -------
    :py:class:`cerberus.Validator`
        Cerberus validator, suitable for validating the configuration
    """

    from cerberus import Validator

    # Create the aips schema and defaults
    aipsroot = pjoin(os.sep, 'home', 'kat', 'AIPS')

    aips_schema = {
        'aipsroot': {'type': 'string', 'default': aipsroot},
        'aipsversion': {'type': 'string', 'default': '31DEC16'},
        'da00': {'type': 'string', 'default': pjoin(aipsroot, 'DA00')},
        'userno': {'type': 'integer', 'default': 105},
    }

    # Create the obit schema and defaults
    obitroot = pjoin(os.sep, 'home', 'kat', 'Obit')
    obitexec = pjoin(obitroot, 'ObitSystem', 'Obit')
    # (url, dir) tuples, where None means localhost
    fitsdirs = [(None, pjoin(aipsroot, 'FITS'))]
    aipsdirs = [(None, pjoin(aipsroot, 'DATA', 'LOCALHOST_1')),
                (None, pjoin(aipsroot, 'DATA', 'LOCALHOST_2'))]

    obit_schema = {
        'obitroot': {'type': 'string', 'default': obitroot},
        'obitexec': {'type': 'string', 'default': obitexec},
        'fitsdirs': {'type': 'list', 'default': fitsdirs},
        'aipsdirs': {'type': 'list', 'default': aipsdirs},
    }

    schema = {
        'aips': {
            'type': 'dict',
            'schema': aips_schema,
            'default': Validator(aips_schema).validated({}),
        },
        'obit': {
            'type': 'dict',
            'schema': obit_schema,
            'default': Validator(obit_schema).validated({}),
        }
    }

    return Validator(schema)


def validate_configuration(configuration):
    """
    Validate the supplied configuration dictionary,
    and return an object containing the configuration

    Parameters
    ----------
    configuration: dict
        User supplied configuration dictionary

    Returns
    -------
    object
        Returns an object with attributes created
        from the key value pairs on `configuration`.
        This process is applied recursively to any
        dictionaries within `configuration`.
    """
    import attr

    cfg = config_validator().validated(configuration)

    def _dicts_to_attrs(key, value):
        """ Recursively convert any dictionaries to attr classes """

        # Handle dictionaries
        if isinstance(value, collections.Mapping):
            # Make an attribute class and instance for this key,
            # with value's keys as attributes
            skey = "{}_section".format(key)
            klass = attr.make_class(skey, value.keys(), frozen=True)
            return klass(*(_dicts_to_attrs(k, v) for k, v in value.items()))

        return value

    return _dicts_to_attrs("main", cfg)


def get_config(aipsdirs=None, fitsdirs=None):
    """
    Get a configuration, optionally specify lists
    of aipsdisks and fitsdisks.

    Parameters
    ----------
    aipsdirs: list (optional)
        list of path locations of aipsdisks
    fitsdirs: list (optional)
        list of path loactions of fitsdisks
    """
    cfg_dict = {'obit': {}, 'aips': {}}
    if aipsdirs:
        cfg_dict['obit']['aipsdirs'] = [(None, disk) for disk in aipsdirs]
    if fitsdirs:
        cfg_dict['obit']['fitsdirs'] = [(None, disk) for disk in fitsdirs]
    return validate_configuration(cfg_dict)
