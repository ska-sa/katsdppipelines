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

    aipsroot = pjoin(os.sep, 'usr', 'local', 'AIPS')
    obitroot = pjoin(os.sep, 'usr', 'local', 'Obit')
    obitexec = pjoin(obitroot, 'ObitSystem', 'Obit')
    fitsdirs = [pjoin(aipsroot, 'FITS'),
                pjoin(aipsroot, 'FITS2')]
    aipsdirs = [pjoin(aipsroot, 'DATA', 'LOCALHOST_1'),
                pjoin(aipsroot, 'DATA', 'LOCALHOST_2')]
    da00 = pjoin(aipsroot, 'DA00')

    obit_schema = {
        'aipsroot' : {'type': 'string', 'default': aipsroot },
        'aipsversion' : {'type': 'string', 'default': '31DEC16' },
        'obitroot': {'type': 'string', 'default': obitroot },
        'obitexec': {'type': 'string', 'default': obitexec },
        'fitsdirs' : {'type': 'list', 'default': fitsdirs },
        'aipsdirs' : {'type': 'list', 'default': aipsdirs },
        'userno': {'type': 'integer', 'default': 105 },
        'da00': {'type':'string', 'default': da00 },
    }

    schema = {
        'obit' : {
            'type' : 'dict',
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

