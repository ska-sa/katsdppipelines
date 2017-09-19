from collections import OrderedDict
import logging

import InfoList
import ParserUtil

import katsdpcontim
from katsdpcontim import obit_err, handle_obit_err
from katsdpcontim.obit_types import OBIT_TYPE_ENUM, OBIT_TYPE

log = logging.getLogger('katsdpcontim')


def parse_aips_config(aips_cfg_file):
    """
    Parses an AIPS config file into a
    dictionary with schema
    :code:`{ option: [type, dimensions, value]}`

    :code:`type_` is an enum. Look at ObitTypes.h
    to figure it out.

    :code:`dims` indicate dimensionality of input
    For scalar types :code:`[64,1,1,1,1]` indicates
    a 1D array of length 64 floats.

    String dims need to be handled slightly differently
    First dimension indicates string length so for e.g.
    :code:`["obit", "    ", "abcd"]` has dims :code:`[4,3,1,1,1]`
    So a :code:`[4,1,1,1,1]` implies one string of length 4

    :code:`value` will always be a list and is probably
    nested if :code:`dims is setup appropriately.

    Parameters
    ----------
    aips_cfg_file : str
        AIPS configuration file

    Returns
    -------
    dict
        A dictionary of AIPS configuration options
    """
    err = obit_err()
    info_list = InfoList.InfoList()
    ParserUtil.PParse(aips_cfg_file, info_list, err)
    handle_obit_err("Error parsing Obit configuration file '{}'"
                    .format(aips_cfg_file), err)

    return InfoList.PGetDict(info_list)


def obit_config_from_aips(aips_cfg_file):
    """
    Extract key-values from AIPS configuration file
    into a { option: value } dictionary.

    Processes the configuration so that the values are
    suitable to apply to ObitTask objects, converting
    singleton list to objects, for example.

    Parameters
    ----------
    aips_cfg_file : str
        AIPS configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """

    def _massage(option):
        """
        Massage values into structures suitable for
        setting on Obit Task objects
        """

        # Split into type, dimensions and value
        type_, dims, value = option

        assert isinstance(value, list)

        enum = OBIT_TYPE_ENUM[type_]

        is_str = enum.name == "string"

        # Coerce values into their python equivalents
        if is_str:
            value = [enum.coerce(v).ljust(dims[0], ' ') for v in value]
        else:
            value = [enum.coerce(v) for v in value]

        # Check second dimension to test singletons
        # if we're handling strings else the first dim
        check_dim = 1 if is_str else 0

        # Return first element from singleton lists
        if dims[check_dim] == 1 and len(value) == 1:
            return value[0]

        return value

    return {k: _massage(o) for k, o
            in parse_aips_config(aips_cfg_file).iteritems()}
